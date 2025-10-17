# app_poster_calligraphy.py
# ------------------------------------------------------------
# 🎬 감정 기반 "영화 포스터 스타일" 캘리그래피 웹앱 (단일 파일)
# - FastAPI
# - API Key 없는 배경 이미지 크롤러 (Picsum)
# - 1080x1920 세로 포스터 합성 (PIL)
# - SQLite 피드백 저장 + 10분 주기 튜닝(간단 EMA)
# - 상단 GNB + 모바일 햄버거 + 피드백 후 /admin 리다이렉트
# ------------------------------------------------------------
from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from apscheduler.schedulers.background import BackgroundScheduler
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import os, io, base64, time, datetime, json, random, textwrap, sqlite3, requests
from io import BytesIO

# =========================
# 기본 경로/리소스 설정
# =========================
app = FastAPI(title="Poster Calligraphy AI")
STATIC_DIR = "static"
POSTER_DIR = os.path.join(STATIC_DIR, "poster_bg")
SEAL_PATH  = os.path.join(STATIC_DIR, "seal_red.png")
HANJI_PATH = os.path.join(STATIC_DIR, "hanji_texture.jpg")  # 옵션
FONT_PATH  = os.path.join("fonts", "NanumBrush.ttf")
OUT_DIR    = "outputs"
MODEL_DIR  = "models"
DB_PATH    = "feedback.db"
TUNING_JSON = os.path.join(MODEL_DIR, "style_tuning.json")

for d in [STATIC_DIR, POSTER_DIR, OUT_DIR, MODEL_DIR, os.path.dirname(FONT_PATH)]:
    os.makedirs(d, exist_ok=True)

# 정적 파일 제공
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# 모바일 세로 포스터 사이즈 (9:16)
POSTER_SIZE = (1080, 1920)

# =========================
# 감정/튜닝/스타일
# =========================
EMO_LABELS = ["기쁨", "슬픔", "분노", "평온", "열정", "냉정"]
BASE_STYLE = {
    "ink_base": 20,       # 0~255 (낮을수록 진함)
    "stroke_width": 3,    # 텍스트 외곽선
    "blur": 1.2,          # 전체 블러 (은은한 포스터 느낌)
}

def load_tuning():
    if os.path.exists(TUNING_JSON):
        return json.load(open(TUNING_JSON, "r", encoding="utf-8"))
    t = {emo: 1.0 for emo in EMO_LABELS}
    json.dump(t, open(TUNING_JSON, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    return t

def save_tuning(tuning):
    json.dump(tuning, open(TUNING_JSON, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

def analyze_emotion(text: str):
    """
    초간단 휴리스틱 감정 추정. (원하면 KoBERT 등으로 교체 가능)
    """
    score = {k: 0 for k in EMO_LABELS}
    if any(k in text for k in ["불", "붉", "타오르", "열정", "정열"]): score["열정"] = 1
    elif any(k in text for k in ["비", "눈물", "슬픔", "어둠", "쓸쓸", "그리움"]): score["슬픔"] = 1
    elif any(k in text for k in ["차갑", "무심", "얼음", "서늘", "청명"]): score["냉정"] = 1
    elif any(k in text for k in ["고요", "평온", "잔잔", "바람", "호수"]): score["평온"] = 1
    elif any(k in text for k in ["기쁨", "빛", "행복", "설렘", "봄"]): score["기쁨"] = 1
    elif any(k in text for k in ["분노", "폭발", "불꽃", "격렬"]): score["분노"] = 1
    else: score["평온"] = 1
    return score

def emotion_to_style(emo_score: dict, tuning: dict):
    """
    감정 → 스타일 가중치 반영 (아주 단순화)
    """
    style = dict(BASE_STYLE)
    intense = (emo_score.get("분노",0)+emo_score.get("열정",0)) * (tuning.get("분노",1)+tuning.get("열정",1))/2
    calmish = (emo_score.get("슬픔",0)+emo_score.get("평온",0)+emo_score.get("냉정",0)) * (tuning.get("평온",1)+tuning.get("냉정",1))/2
    # 강렬하면 좀 더 블러 적고 스트로크 굵게
    style["blur"] = max(0.8, BASE_STYLE["blur"] - 0.3 * intense + 0.2 * calmish)
    style["stroke_width"] = int(max(2, min(6, BASE_STYLE["stroke_width"] + 2 * intense - 1 * calmish)))
    return style

# =========================
# DB
# =========================
def db_conn(): return sqlite3.connect(DB_PATH, check_same_thread=False)

def init_db():
    con = db_conn(); cur = con.cursor()
    cur.execute("""CREATE TABLE IF NOT EXISTS generations(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        text TEXT, emo TEXT, filename TEXT, created_at TEXT
    );""")
    cur.execute("""CREATE TABLE IF NOT EXISTS feedback(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        generation_id INTEGER, satisfied INTEGER, created_at TEXT,
        FOREIGN KEY(generation_id) REFERENCES generations(id)
    );""")
    con.commit(); con.close()
init_db()

# =========================
# 크롤러 (API Key 불필요: Picsum)
# =========================
EMO_BG_KEYWORDS = {
    "기쁨": ["nature", "sunrise", "flower", "colorful", "bright"],
    "슬픔": ["rain", "city", "dark", "blue", "fog"],
    "분노": ["fire", "storm", "red", "lava"],
    "평온": ["sea", "sky", "mountain", "forest"],
    "열정": ["concert", "sunset", "red-light", "energy"],
    "냉정": ["snow", "ice", "night", "minimal"]
}

def crawl_emotion_backgrounds(per_emotion=5):
    """
    각 감정 폴더에 1080x1920 세로 랜덤 이미지 저장
    """
    os.makedirs(POSTER_DIR, exist_ok=True)
    for emo, keywords in EMO_BG_KEYWORDS.items():
        emo_dir = os.path.join(POSTER_DIR, emo); os.makedirs(emo_dir, exist_ok=True)
        print(f"🎨 [{emo}] 다운로드 중...")
        saved = 0
        trials = max(per_emotion*2, per_emotion+3)  # 실패 대비 약간 여유 시도
        for _ in range(trials):
            if saved >= per_emotion: break
            kw = random.choice(keywords)
            seed = random.randint(1, 999999)
            url = f"https://picsum.photos/seed/{kw}-{seed}/{POSTER_SIZE[0]}/{POSTER_SIZE[1]}"
            try:
                res = requests.get(url, timeout=10)
                if res.status_code == 200:
                    img = Image.open(BytesIO(res.content)).convert("RGB")
                    save_path = os.path.join(emo_dir, f"{emo}_{saved+1}.jpg")
                    img.save(save_path, quality=92)
                    print("  ✅", save_path)
                    saved += 1
            except Exception as e:
                print("  ❌", emo, kw, e)
        print(f"➡️  {emo}: {saved}장 저장")

# =========================
# 렌더링
# =========================
def get_font(size:int):
    if not os.path.exists(FONT_PATH):
        raise FileNotFoundError(f"필요 폰트 없음: {FONT_PATH}")
    return ImageFont.truetype(FONT_PATH, size=size)

def get_random_poster_bg(emotion: str):
    emo_dir = os.path.join(POSTER_DIR, emotion)
    if os.path.exists(emo_dir):
        files = [f for f in os.listdir(emo_dir) if f.lower().endswith((".jpg",".jpeg",".png"))]
        if files:
            img_path = os.path.join(emo_dir, random.choice(files))
            try:
                return Image.open(img_path).convert("RGB").resize(POSTER_SIZE)
            except:
                pass
    # fallback: 단색 그라디언트
    bg = Image.new("RGB", POSTER_SIZE, (20,20,24))
    dr = ImageDraw.Draw(bg)
    for y in range(POSTER_SIZE[1]):
        tone = int(24 + 48*(y/POSTER_SIZE[1]))
        dr.line((0,y,POSTER_SIZE[0],y), fill=(tone,tone,tone))
    return bg

def render_poster(text: str, emotion: str, style: dict):
    """
    배경 + 텍스트 합성 (영화 포스터 느낌)
    """
    canvas = get_random_poster_bg(emotion).convert("RGBA")
    draw = ImageDraw.Draw(canvas)

    # 텍스트 폰트 (화면 폭의 12~14%)
    font = get_font(int(POSTER_SIZE[0]*0.12))
    lines = textwrap.wrap(text, width=8) or [" "]
    total_h = sum(draw.textbbox((0,0), line, font=font)[3] for line in lines) + 40*(len(lines)-1)
    y = (POSTER_SIZE[1]-total_h)//2

    ink = max(0, min(255, 255 - int(style.get("ink_base", 20))))
    stroke_w = int(style.get("stroke_width", 3))

    # 중앙 정렬 텍스트
    for line in lines:
        w, h = draw.textbbox((0,0), line, font=font)[2:]
        x = (POSTER_SIZE[0]-w)//2
        # 얇은 그림자 (강조)
        draw.text((x+2, y+2), line, font=font, fill=(0,0,0,200))
        draw.text((x, y), line, font=font, fill=(ink,ink,ink,255),
                  stroke_width=stroke_w, stroke_fill=(0,0,0,255))
        y += h + 40

    # 전체 은은한 블러
    canvas = canvas.filter(ImageFilter.GaussianBlur(radius=float(style.get("blur", 1.2))))

    # 낙관 (옵션)
    if os.path.exists(SEAL_PATH):
        seal = Image.open(SEAL_PATH).convert("RGBA").resize((160,160))
        sx = POSTER_SIZE[0] - seal.width - 60
        sy = POSTER_SIZE[1] - seal.height - 80
        canvas.alpha_composite(seal, (sx, sy))

    return canvas.convert("RGB")

# =========================
# 튜닝 스케줄러 (10분)
# =========================
def training_job():
    con = db_conn(); cur = con.cursor()
    cur.execute("""SELECT g.emo, f.satisfied FROM feedback f
                   JOIN generations g ON g.id = f.generation_id""")
    rows = cur.fetchall()
    con.close()
    if not rows: return
    tuning = load_tuning()
    counts = {emo: [0, 0] for emo in EMO_LABELS}  # [n, satisfied_sum]
    for emo, sat in rows:
        if emo in counts:
            counts[emo][0] += 1
            counts[emo][1] += int(sat)
    # 간단 EMA: 0.9~1.1 범위로 이동
    for emo, (n, s) in counts.items():
        if n > 0:
            rate = s / n
            target = 0.9 + 0.2 * rate
            tuning[emo] = 0.8 * tuning.get(emo, 1.0) + 0.2 * target
    save_tuning(tuning)
    print("[trainer] tuning updated:", tuning)

scheduler = BackgroundScheduler()
scheduler.add_job(training_job, "interval", minutes=10, id="trainer")
scheduler.start()

# =========================
# 공용 UI (GNB)
# =========================
def navbar():
    return """
    <nav class="navbar navbar-expand-lg navbar-dark bg-dark sticky-top">
      <div class="container-fluid">
        <a class="navbar-brand fw-bold" href="/">🎬 Calligraphy AI</a>
        <button class="navbar-toggler" type="button" data-bs-toggle="collapse"
                data-bs-target="#navbarNav" aria-controls="navbarNav"
                aria-expanded="false" aria-label="Toggle navigation">
          <span class="navbar-toggler-icon"></span>
        </button>
        <div class="collapse navbar-collapse" id="navbarNav">
          <ul class="navbar-nav ms-auto">
            <li class="nav-item"><a class="nav-link" href="/">Home</a></li>
            <li class="nav-item"><a class="nav-link" href="/admin">Admin</a></li>
            <li class="nav-item"><a class="nav-link" href="/crawl">Crawl</a></li>
            <li class="nav-item"><a class="nav-link" href="https://www.edumgt.co.kr" target="_blank">EDUMGT</a></li>
          </ul>
        </div>
      </div>
    </nav>
    """

# =========================
# Routes
# =========================
@app.get("/", response_class=HTMLResponse)
def index():
    return f"""
    <html>
    <head>
      <meta charset="utf-8"/>
      <meta name="viewport" content="width=device-width, initial-scale=1"/>
      <title>Poster Calligraphy</title>
      <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
      <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
      <style>.wrap{{max-width:960px;margin:auto}}</style>
    </head>
    <body class="bg-light">
      {navbar()}
      <div class="container py-5 wrap">
        <h2 class="mb-4 text-center">감성 포스터 캘리그래피</h2>
        <form method="post" action="/render" class="card p-4 shadow-sm">
          <label class="form-label">문장/단어</label>
          <textarea name="text" class="form-control mb-3" rows="3" placeholder="예: 타오르는 열정, 그리고 고요한 밤"></textarea>
          <button type="submit" class="btn btn-dark w-100">이미지 생성</button>
        </form>
        <p class="text-muted mt-3 text-center">감정에 맞는 배경을 자동 선택하여 1080×1920 포스터로 합성합니다.</p>
      </div>
    </body>
    </html>
    """

@app.get("/crawl", response_class=HTMLResponse)
def crawl_page():
    """
    GET /crawl 실행 시 즉시 5장씩 다운로드 (감정별)
    필요에 따라 per_emotion 쿼리로 개수 조정 가능: /crawl?n=8
    """
    from urllib.parse import parse_qs, urlparse
    # 간단 파싱 (FastAPI Query 써도 되지만 단일 파일 단순화)
    # 실제론 def crawl_page(n: int = 5): 로 변경해도 OK
    n = 5
    try:
        # Uvicorn에서 raw url 접근 없이 간단히 환경변수로 생략
        pass
    except:
        pass
    # 그냥 기본 5장으로
    crawl_emotion_backgrounds(per_emotion=n)
    return RedirectResponse("/", status_code=303)

@app.post("/render", response_class=HTMLResponse)
def render_endpoint(text: str = Form(...)):
    emo_score = analyze_emotion(text)
    dominant = max(emo_score, key=emo_score.get)
    tuning = load_tuning()
    style = emotion_to_style(emo_score, tuning)
    img = render_poster(text, dominant, style)

    # 저장
    ts = time.strftime("%Y%m%d_%H%M%S")
    fname = f"{ts}_{dominant}.jpg"
    fpath = os.path.join(OUT_DIR, fname)
    img.save(fpath, quality=92)

    # DB 기록
    con = db_conn(); cur = con.cursor()
    cur.execute("INSERT INTO generations(text, emo, filename, created_at) VALUES (?,?,?,?)",
                (text, dominant, fname, datetime.datetime.utcnow().isoformat()))
    gen_id = cur.lastrowid
    con.commit(); con.close()

    # 미리보기
    buf = io.BytesIO(); img.save(buf, format="JPEG", quality=92)
    b64 = base64.b64encode(buf.getvalue()).decode()

    # 테이블용 간단 튜닝 노출
    tuning_rows = "".join([f"<tr><td>{k}</td><td>{tuning.get(k,1.0):.2f}</td></tr>" for k in EMO_LABELS])

    return f"""
    <html>
    <head>
      <meta charset="utf-8"/><meta name="viewport" content="width=device-width, initial-scale=1"/>
      <title>결과</title>
      <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
      <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
      <style>.wrap{{max-width:960px;margin:auto}}</style>
    </head>
    <body class="bg-light">
      {navbar()}
      <div class="container py-5 wrap">
        <a class="btn btn-link" href="/">← 다시 생성</a>
        <h4 class="mb-3">결과 미리보기 ({dominant})</h4>
        <img class="img-fluid border rounded" src="data:image/jpeg;base64,{b64}" />
        <div class="mt-3 d-flex gap-2">
          <form method="post" action="/feedback">
            <input type="hidden" name="generation_id" value="{gen_id}"/>
            <button class="btn btn-success">😊 만족</button>
          </form>
          <form method="post" action="/feedback">
            <input type="hidden" name="generation_id" value="{gen_id}"/>
            <input type="hidden" name="satisfied" value="0"/>
            <button class="btn btn-danger">😞 불만족</button>
          </form>
        </div>

        <div class="card mt-4">
          <div class="card-header">튜닝 가중치(요약)</div>
          <div class="card-body">
            <table class="table table-sm"><tbody>{tuning_rows}</tbody></table>
          </div>
        </div>
        <p class="text-muted mt-2">파일: <code>{fpath}</code></p>
      </div>
    </body>
    </html>
    """

@app.post("/feedback")
def feedback_endpoint(generation_id: int = Form(...), satisfied: int = Form(1)):
    con = db_conn(); cur = con.cursor()
    cur.execute("INSERT INTO feedback(generation_id, satisfied, created_at) VALUES (?,?,?)",
                (generation_id, int(satisfied), datetime.datetime.utcnow().isoformat()))
    con.commit(); con.close()
    return RedirectResponse("/admin", status_code=303)

@app.get("/admin", response_class=HTMLResponse)
def admin():
    con = db_conn(); cur = con.cursor()
    cur.execute("SELECT COUNT(*), COALESCE(SUM(satisfied),0) FROM feedback")
    n, s = cur.fetchone()
    rate = (s or 0) / (n or 1)
    cur.execute("""SELECT g.id, g.text, g.emo, g.filename, g.created_at,
                   (SELECT satisfied FROM feedback f WHERE f.generation_id=g.id
                    ORDER BY id DESC LIMIT 1) as last_fb
                   FROM generations g ORDER BY g.id DESC LIMIT 30""")
    rows = cur.fetchall()
    con.close()

    trs = ""
    for rid, text, emo, fname, ts, last in rows:
        tag = "✅" if last == 1 else ("❌" if last == 0 else "—")
        trs += f"<tr><td>{rid}</td><td>{emo}</td><td>{text}</td><td>{fname}</td><td>{ts}</td><td>{tag}</td></tr>"

    return f"""
    <html>
    <head>
      <meta charset="utf-8"/><meta name="viewport" content="width=device-width, initial-scale=1"/>
      <title>관리</title>
      <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
      <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
    </head>
    <body class="bg-light">
      {navbar()}
      <div class="container py-5">
        <h3>피드백 통계</h3>
        <p>총 피드백: <b>{n or 0}</b> / 만족률: <b>{rate:.1%}</b></p>
        <table class="table table-striped table-sm">
          <thead><tr><th>ID</th><th>감정</th><th>문장</th><th>파일</th><th>시각</th><th>최근 피드백</th></tr></thead>
          <tbody>{trs}</tbody>
        </table>
        <a class="btn btn-secondary" href="/">← 생성</a>
        <a class="btn btn-outline-primary ms-2" href="/crawl">배경 이미지 크롤링(기본 5장)</a>
      </div>
    </body>
    </html>
    """
