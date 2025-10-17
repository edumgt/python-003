# app_feedback_training.py
# ------------------------------------------------------------
# FastAPI + SQLite + APScheduler
# - 감정(휴리스틱) → 스타일 매핑 → 캘리그래피 이미지 생성
# - 결과에 대해 만족/불만족 라벨 저장
# - 주기적으로(10분) 피드백을 학습하여 "스타일 튜닝 파라미터" 업데이트
# 실행:
#   uvicorn app_feedback_training:app --host 0.0.0.0 --port 8000 --reload
# ------------------------------------------------------------

from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from apscheduler.schedulers.background import BackgroundScheduler

import io, os, random, textwrap, json, sqlite3, base64, time, datetime
import numpy as np

# ----------------------------
# 경로/리소스 설정
# ----------------------------
DB_PATH      = "feedback.db"
OUT_DIR      = "outputs"
MODEL_DIR    = "models"
TUNING_JSON  = os.path.join(MODEL_DIR, "style_tuning.json")

FONT_PATH    = "fonts/NanumBrush.ttf"         # 한글 지원 폰트 필수
HANJI_PATH   = "static/hanji_texture.jpg"     # 없으면 내부 노이즈로 대체
SEAL_PATH    = "static/seal_red.png"          # (옵션) 낙관 PNG

CANVAS_W, CANVAS_H = 1600, 900
MARGIN = 120
BASE_FONT_SIZE = 150
LINE_SPACING = 1.45

# ----------------------------
# FastAPI & 정적파일
# ----------------------------
app = FastAPI(title="Feedback-Driven Calligraphy")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

# ----------------------------
# SQLite 초기화
# ----------------------------
def db_conn():
    return sqlite3.connect(DB_PATH, check_same_thread=False)

def init_db():
    con = db_conn()
    cur = con.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS generations (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        text TEXT NOT NULL,
        emo_json TEXT NOT NULL,
        style_json TEXT NOT NULL,
        filename TEXT NOT NULL,
        created_at TEXT NOT NULL
    );
    """)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS feedback (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        generation_id INTEGER NOT NULL,
        satisfied INTEGER NOT NULL,    -- 1 or 0
        created_at TEXT NOT NULL,
        FOREIGN KEY (generation_id) REFERENCES generations (id)
    );
    """)
    con.commit()
    con.close()

init_db()

# ----------------------------
# 감정 분석(간단 휴리스틱) & 스타일 매핑
# ----------------------------
EMO_LABELS = ["기쁨", "슬픔", "분노", "평온", "열정", "냉정"]

def analyze_emotion(text: str):
    score = {k: 0.0 for k in EMO_LABELS}
    pos_words    = ["기쁨","행복","설렘","빛","봄","환희","웃음","햇살","희망"]
    sad_words    = ["슬픔","눈물","비","쓸쓸","고독","그리움","밤","겨울"]
    anger_words  = ["분노","불꽃","타오르","폭발","격렬","울분"]
    calm_words   = ["고요","평온","잔잔","바람","호수","은은","담담"]
    passion_words= ["열정","붉","뜨겁","불","강렬","타오르"]
    cold_words   = ["냉정","차갑","서늘","무심","청명"]

    def bump(keys, label, w=1.0):
        for kw in keys:
            if kw in text:
                score[label] += w

    bump(pos_words,"기쁨",1.2)
    bump(sad_words,"슬픔",1.2)
    bump(anger_words,"분노",1.2)
    bump(calm_words,"평온",1.2)
    bump(passion_words,"열정",1.2)
    bump(cold_words,"냉정",1.2)

    if sum(score.values()) == 0:
        score["평온"] = 0.6
        score["기쁨"] = 0.4

    s = sum(score.values())
    for k in score:
        score[k] /= s
    return score

# 기본 스타일(튜닝 전)
BASE_STYLE = {
    "ink_base": 20,     # 먹 진하기(낮을수록 진함, 0~255)
    "blur": 1.6,        # 번짐 강도
    "noise": 25,        # 먹 질감 노이즈
    "angle_amp": 8,     # 글자 기울기 범위
    "scale_low": 0.85,  # 글자 크기 랜덤 하한
    "scale_high": 1.35, # 글자 크기 랜덤 상한
    "jitter": 6,        # 글자 위치 흔들림
    "hanji_strength": 1.0,
    "stroke_width": 2,  # 획 굵기(외곽선)
}

# 튜닝 파라미터(학습으로 갱신)
def load_tuning():
    if os.path.exists(TUNING_JSON):
        with open(TUNING_JSON, "r", encoding="utf-8") as f:
            return json.load(f)
    # 초기값: 감정별 가중치 1.0
    t = {emo: 1.0 for emo in EMO_LABELS}
    with open(TUNING_JSON, "w", encoding="utf-8") as f:
        json.dump(t, f, ensure_ascii=False, indent=2)
    return t

def save_tuning(tuning: dict):
    with open(TUNING_JSON, "w", encoding="utf-8") as f:
        json.dump(tuning, f, ensure_ascii=False, indent=2)

TUNING = load_tuning()

def emotion_to_style(emo_scores: dict, tuning: dict):
    # 감정 점수에 튜닝 가중치를 곱해 "효과적인 강도"를 반영
    # (예) 특정 감정에서 만족도가 높으면 해당 감정의 영향력을 키움
    adj = {k: emo_scores[k] * float(tuning.get(k, 1.0)) for k in EMO_LABELS}

    joy    = adj.get("기쁨",0)
    sad    = adj.get("슬픔",0)
    anger  = adj.get("분노",0)
    calm   = adj.get("평온",0)
    passion= adj.get("열정",0)
    cold   = adj.get("냉정",0)

    intense = anger + passion   # 강렬 감정
    cool    = sad + calm + cold # 차분/냉정

    style = dict(BASE_STYLE)

    style["ink_base"]   = int(max(0, 20 - 10 * intense))
    style["blur"]       = 1.2 + 1.0 * intense
    style["noise"]      = int(20 + 30 * intense)
    style["angle_amp"]  = int(6 + 10 * intense)
    style["scale_low"]  = 0.85 - 0.1 * intense
    style["scale_high"] = 1.35 + 0.1 * intense
    style["jitter"]     = int(6 + 6 * intense)

    # 획 굵기(감정 기반)
    base_thickness = 2 + 4 * intense - 2 * cool
    style["stroke_width"] = int(np.clip(base_thickness, 1, 6))

    # 평온: 안정화
    if calm > 0:
        style["angle_amp"] = int(style["angle_amp"] * (1.0 - 0.3 * calm))
        style["jitter"]    = int(style["jitter"] * (1.0 - 0.3 * calm))
        style["blur"]      = max(1.0, style["blur"] - 0.3 * calm)

    # 기쁨: 명료감 증가
    if joy > 0:
        style["blur"]  = max(1.0, style["blur"] - 0.2 * joy)
        style["noise"] = int(max(10, style["noise"] - 8 * joy))

    return style

# ----------------------------
# 렌더링 유틸
# ----------------------------
def get_font(size): 
    if not os.path.exists(FONT_PATH):
        raise FileNotFoundError(f"폰트가 없습니다: {FONT_PATH}")
    return ImageFont.truetype(FONT_PATH, size=size)

def make_hanji_background(strength=1.0) -> Image.Image:
    if os.path.exists(HANJI_PATH):
        base = Image.open(HANJI_PATH).convert("RGB").resize((CANVAS_W, CANVAS_H))
    else:
        base = Image.new("RGB", (CANVAS_W, CANVAS_H), (250, 248, 240))
        noise = np.random.normal(0, 12 * strength, (CANVAS_H, CANVAS_W, 3))
        arr = np.clip(np.array(base).astype("float32") + noise, 0, 255).astype("uint8")
        base = Image.fromarray(arr)
    return base

def add_ink_effect(img: Image.Image, blur=1.6, noise=25) -> Image.Image:
    arr = np.array(img).astype("float32")
    arr = np.clip((arr - 40) * 1.4, 0, 255)  # 대비↑
    img = Image.fromarray(arr.astype("uint8"))
    if blur > 0:
        img = img.filter(ImageFilter.GaussianBlur(radius=float(blur)))
    if noise > 0:
        noise_arr = np.random.normal(0, float(noise), arr.shape)
        arr2 = np.clip(arr + noise_arr, 0, 255).astype("uint8")
        img = Image.fromarray(arr2)
    return img

def apply_seal(canvas: Image.Image):
    if not os.path.exists(SEAL_PATH):
        return canvas
    seal = Image.open(SEAL_PATH).convert("RGBA").resize((180, 180))
    x = CANVAS_W - seal.width - 100
    y = CANVAS_H - seal.height - 80
    canvas.paste(seal, (x, y), seal)
    return canvas

def render_calligraphy(text: str, style: dict) -> Image.Image:
    canvas = make_hanji_background(style.get("hanji_strength", 1.0)).convert("RGBA")
    draw = ImageDraw.Draw(canvas)

    avg_font = get_font(BASE_FONT_SIZE)
    max_text_width = CANVAS_W - 2 * MARGIN
    avg_char = max(1, draw.textlength("한", font=avg_font))
    max_chars_per_line = max(1, int(max_text_width / avg_char))
    wrapped = []
    for line in text.splitlines():
        wrapped.extend(textwrap.wrap(line, width=max_chars_per_line) or [" "])

    total_h = sum(draw.textbbox((0, 0), line, font=avg_font)[3] * LINE_SPACING for line in wrapped)
    y = int((CANVAS_H - total_h) / 2)

    ink_base  = int(style["ink_base"])
    angle_amp = int(style["angle_amp"])
    scale_low, scale_high = float(style["scale_low"]), float(style["scale_high"])
    jitter = int(style["jitter"])
    stroke_w = int(style["stroke_width"])

    for line in wrapped:
        x = MARGIN
        for ch in line:
            scale = random.uniform(scale_low, scale_high)
            angle = random.uniform(-angle_amp, angle_amp)
            ox, oy = random.randint(-jitter, jitter), random.randint(-jitter, jitter)
            font = get_font(int(BASE_FONT_SIZE * scale))

            temp = Image.new("RGBA", (420, 420), (255, 255, 255, 0))
            dtemp = ImageDraw.Draw(temp)
            ink = random.randint(0, max(0, ink_base))
            dtemp.text(
                (50, 100),
                ch,
                font=font,
                fill=(ink, ink, ink, 255),
                stroke_width=stroke_w,
                stroke_fill=(ink, ink, ink, 255),
            )

            rotated = temp.rotate(angle, resample=Image.BICUBIC, expand=True)
            bbox = rotated.getbbox()
            if bbox:
                char_img = rotated.crop(bbox)
                canvas.alpha_composite(char_img, (x + ox, y + oy))

            w = draw.textbbox((0, 0), ch, font=font)[2]
            x += int(w * random.uniform(0.9, 1.1))

        _, _, _, h = draw.textbbox((0, 0), line, font=avg_font)
        y += int(h * LINE_SPACING)

    canvas = add_ink_effect(canvas, blur=style["blur"], noise=style["noise"])
    canvas = apply_seal(canvas)
    return canvas.convert("RGB")

# ----------------------------
# 저장/불러오기 유틸
# ----------------------------
def save_generation_record(text: str, emo: dict, style: dict, filename: str) -> int:
    con = db_conn(); cur = con.cursor()
    cur.execute(
        "INSERT INTO generations(text, emo_json, style_json, filename, created_at) VALUES (?,?,?,?,?)",
        (text, json.dumps(emo, ensure_ascii=False), json.dumps(style, ensure_ascii=False),
         filename, datetime.datetime.utcnow().isoformat())
    )
    gen_id = cur.lastrowid
    con.commit(); con.close()
    return gen_id

def save_feedback(generation_id: int, satisfied: int):
    con = db_conn(); cur = con.cursor()
    cur.execute(
        "INSERT INTO feedback(generation_id, satisfied, created_at) VALUES (?,?,?)",
        (generation_id, int(satisfied), datetime.datetime.utcnow().isoformat())
    )
    con.commit(); con.close()

def b64_of_image(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")

# ----------------------------
# 학습(튜닝) 작업: 10분마다
# ----------------------------
def training_job():
    """
    feedback + generations를 조인해서 감정별 만족률을 계산,
    튜닝 가중치(TUNING_JSON)를 0.9 ~ 1.1 사이로 서서히 이동(EMA 느낌)
    """
    con = db_conn(); cur = con.cursor()
    cur.execute("""
        SELECT g.emo_json, f.satisfied
        FROM feedback f
        JOIN generations g ON g.id = f.generation_id
    """)
    rows = cur.fetchall()
    con.close()

    if not rows:
        return

    # 각 감정별 평균 만족도 계산
    emo_sum = {k: 0.0 for k in EMO_LABELS}
    emo_cnt = {k: 0.0 for k in EMO_LABELS}

    for emo_json, sat in rows:
        emo = json.loads(emo_json)
        # 해당 결과에 "주요 감정 상위1~2개"만 가중치로 본다(잡음 완화)
        top = sorted(emo.items(), key=lambda x: x[1], reverse=True)[:2]
        for k, v in top:
            emo_sum[k] += (v * float(sat))   # 만족(1)일수록 +, 불만(0)은 0
            emo_cnt[k] += v

    # 비율(만족률) -> 0~1
    sat_rate = {}
    for k in EMO_LABELS:
        if emo_cnt[k] > 0:
            sat_rate[k] = emo_sum[k] / emo_cnt[k]
        else:
            sat_rate[k] = 0.5  # 데이터 없으면 중립

    # 현 튜닝 불러와서 0.9~1.1 범위로 서서히 이동
    tuning = load_tuning()
    for k in EMO_LABELS:
        target = 0.9 + 0.2 * sat_rate[k]  # 0(불만족)→0.9, 1(만족)→1.1
        # 지수이동평균(EMA)식 업데이트
        tuning[k] = float(0.8 * tuning.get(k, 1.0) + 0.2 * target)

    save_tuning(tuning)
    print("[trainer] updated tuning:", tuning)

# 스케줄러 시작
scheduler = BackgroundScheduler()
scheduler.add_job(training_job, "interval", minutes=10, id="trainer")
scheduler.start()

# ----------------------------
# 라우트
# ----------------------------
@app.get("/", response_class=HTMLResponse)
def index():
    return """
    <html>
    <head>
      <meta charset="utf-8"/>
      <title>Feedback-Driven Calligraphy</title>
      <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
      <style>
        .wrap{max-width:960px}
        img{max-width:100%}
      </style>
    </head>
    <body class="bg-light">
      <div class="container py-5 wrap">
        <h1 class="mb-4 text-center">🖋️ 감성 캘리그래피 (피드백 학습형)</h1>
        <form method="post" action="/render" class="card p-4 shadow-sm">
          <label class="form-label">문장/단어</label>
          <textarea name="text" class="form-control mb-3" rows="3" placeholder="예: 타오르는 열정, 그리고 고요한 마음"></textarea>
          <button type="submit" class="btn btn-dark">이미지 생성</button>
        </form>
        <p class="text-muted mt-3">※ 만족/불만족 피드백이 10분 간격으로 반영되어 스타일이 점진적으로 개선됩니다.</p>
      </div>
    </body>
    </html>
    """

@app.post("/render", response_class=HTMLResponse)
def render_endpoint(text: str = Form(...)):
    try:
        emo = analyze_emotion(text)
        tuning = load_tuning()
        style = emotion_to_style(emo, tuning)
        img = render_calligraphy(text, style)

        # 파일로 저장 + DB 기록
        ts = time.strftime("%Y%m%d_%H%M%S")
        fname = f"{ts}.png"
        fpath = os.path.join(OUT_DIR, fname)
        img.save(fpath)

        gen_id = save_generation_record(text, emo, style, fname)

        # 응답 HTML (이미지 미리보기 + 피드백 버튼)
        b64 = b64_of_image(img)
        # 간단한 감정/튜닝 가시화
        emo_rows = "".join([f"<tr><td>{k}</td><td>{emo[k]:.2f}</td><td>{tuning.get(k,1.0):.2f}</td></tr>" for k in EMO_LABELS])

        return f"""
        <html>
        <head>
          <meta charset="utf-8"/>
          <title>결과</title>
          <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
          <style>.wrap{{max-width:960px}}</style>
        </head>
        <body class="bg-light">
          <div class="container py-5 wrap">
            <a class="btn btn-link" href="/">← 다시 생성</a>
            <h3 class="mt-2">결과 미리보기</h3>
            <img class="img-fluid border rounded" src="data:image/png;base64,{b64}"/>
            <div class="mt-3 d-flex gap-2">
              <form method="post" action="/feedback">
                <input type="hidden" name="generation_id" value="{gen_id}"/>
                <input type="hidden" name="satisfied" value="1"/>
                <button class="btn btn-success">😊 만족</button>
              </form>
              <form method="post" action="/feedback">
                <input type="hidden" name="generation_id" value="{gen_id}"/>
                <input type="hidden" name="satisfied" value="0"/>
                <button class="btn btn-danger">😞 불만족</button>
              </form>
            </div>

            <div class="card mt-4">
              <div class="card-header">감정 추정 & 튜닝 가중치</div>
              <div class="card-body">
                <table class="table table-sm">
                  <thead><tr><th>감정</th><th>확률</th><th>튜닝 가중치</th></tr></thead>
                  <tbody>{emo_rows}</tbody>
                </table>
              </div>
            </div>

            <p class="text-muted mt-3">저장 파일: <code>{fpath}</code></p>
          </div>
        </body>
        </html>
        """
    except Exception as e:
        return HTMLResponse(f"<pre>{e}</pre>", status_code=500)

@app.post("/feedback")
def feedback_endpoint(generation_id: int = Form(...), satisfied: int = Form(...)):
    try:
        save_feedback(generation_id, int(satisfied))
        return JSONResponse({"ok": True, "generation_id": generation_id, "satisfied": int(satisfied)})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

# (선택) 최근 통계 확인용 간단 관리자 페이지
@app.get("/admin", response_class=HTMLResponse)
def admin():
    con = db_conn(); cur = con.cursor()
    cur.execute("SELECT COUNT(*), SUM(satisfied) FROM feedback")
    n, s = cur.fetchone()
    sat_rate = (s or 0) / n if n else 0.0
    cur.execute("""
        SELECT g.id, g.text, g.filename, g.created_at,
               (SELECT satisfied FROM feedback f WHERE f.generation_id=g.id ORDER BY id DESC LIMIT 1) as last_fb
        FROM generations g ORDER BY g.id DESC LIMIT 30
    """)
    rows = cur.fetchall()
    con.close()

    trs = ""
    for rid, text, fname, ts, last in rows:
        tag = "✅" if last == 1 else ("❌" if last == 0 else "—")
        trs += f"<tr><td>{rid}</td><td>{text}</td><td>{fname}</td><td>{ts}</td><td>{tag}</td></tr>"

    return f"""
    <html>
    <head>
      <meta charset="utf-8"/>
      <title>관리</title>
      <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    </head>
    <body class="bg-light">
      <div class="container py-5">
        <h3>피드백 통계</h3>
        <p>총 피드백 수: <b>{n or 0}</b> / 만족률: <b>{sat_rate:.2%}</b></p>
        <table class="table table-striped table-sm">
          <thead><tr><th>ID</th><th>텍스트</th><th>파일</th><th>생성시각</th><th>최근 피드백</th></tr></thead>
          <tbody>{trs}</tbody>
        </table>
        <a href="/" class="btn btn-secondary">← 생성 페이지</a>
      </div>
    </body>
    </html>
    """
