# ------------------------------------------------------------
# app_poster_calligraphy.py
# FastAPI + SQLite + Emotion-based Calligraphy Poster Generator
# ------------------------------------------------------------


from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from apscheduler.schedulers.background import BackgroundScheduler
from PIL import (
    Image, ImageDraw, ImageFont, ImageFilter,
    ImageChops, ImageOps, ImageEnhance
)
import os, io, json, random, sqlite3, time, textwrap, datetime, requests
import base64

# ------------------------------------------------------------
# 기본 설정
# ------------------------------------------------------------
DB_PATH = "feedback.db"
STATIC_DIR = "static"
POSTER_DIR = os.path.join(STATIC_DIR, "poster_bg")
FONT_PATH = "fonts/NanumBrush.ttf"
SEAL_PATH = os.path.join(STATIC_DIR, "seal_red.png")
POSTER_SIZE = (720, 1280)  # 모바일 세로 비율
os.makedirs(STATIC_DIR, exist_ok=True)
os.makedirs(POSTER_DIR, exist_ok=True)
os.makedirs("outputs", exist_ok=True)

EMO_LABELS = ["기쁨", "슬픔", "분노", "평온", "열정", "냉정"]

# ------------------------------------------------------------
# FastAPI 초기화
# ------------------------------------------------------------
app = FastAPI(title="AI Calligraphy Poster")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# ------------------------------------------------------------
# SQLite 초기화
# ------------------------------------------------------------
def db_conn():
    return sqlite3.connect(DB_PATH, check_same_thread=False)

def init_db():
    con = db_conn()
    cur = con.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS generations(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        text TEXT,
        emo TEXT,
        filename TEXT,
        created_at TEXT
    )
    """)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS feedback(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        generation_id INTEGER,
        satisfied INTEGER,
        created_at TEXT,
        FOREIGN KEY (generation_id) REFERENCES generations(id)
    )
    """)
    con.commit(); con.close()
init_db()

# ------------------------------------------------------------
# 감정 분석 (간단 휴리스틱)
# ------------------------------------------------------------
def analyze_emotion(text: str):
    emo = {k: 0.0 for k in EMO_LABELS}
    pos = ["행복", "기쁨", "빛", "웃음", "희망"]
    sad = ["슬픔", "눈물", "그리움", "쓸쓸", "비"]
    anger = ["분노", "타오르", "격렬", "불꽃"]
    calm = ["평온", "고요", "잔잔", "바람"]
    passion = ["열정", "붉", "뜨겁", "강렬"]
    cold = ["냉정", "차갑", "서늘", "무심"]
    for w in pos:     emo["기쁨"] += text.count(w)
    for w in sad:     emo["슬픔"] += text.count(w)
    for w in anger:   emo["분노"] += text.count(w)
    for w in calm:    emo["평온"] += text.count(w)
    for w in passion: emo["열정"] += text.count(w)
    for w in cold:    emo["냉정"] += text.count(w)
    if sum(emo.values()) == 0: emo["평온"] = 1.0
    s = sum(emo.values()); return {k: v/s for k,v in emo.items()}

# ------------------------------------------------------------
# 튜닝 파라미터 관리
# ------------------------------------------------------------
MODEL_DIR = "models"; os.makedirs(MODEL_DIR, exist_ok=True)
TUNING_JSON = os.path.join(MODEL_DIR, "style_tuning.json")

def load_tuning():
    if os.path.exists(TUNING_JSON):
        return json.load(open(TUNING_JSON, encoding="utf-8"))
    t = {emo: 1.0 for emo in EMO_LABELS}
    json.dump(t, open(TUNING_JSON,"w",encoding="utf-8"), ensure_ascii=False, indent=2)
    return t

def save_tuning(tuning: dict):
    json.dump(tuning, open(TUNING_JSON,"w",encoding="utf-8"), ensure_ascii=False, indent=2)

# ------------------------------------------------------------
# 감정별 스타일 매핑
# ------------------------------------------------------------
BASE_STYLE = {"ink_base":20,"blur":1.5}

def emotion_to_style(emo_score: dict, tuning: dict):
    style = dict(BASE_STYLE)
    emo = max(emo_score, key=emo_score.get)
    weight = tuning.get(emo, 1.0)
    # 감정별 시그니처
    if emo == "기쁨": style.update({"ink_base":10,"blur":1.0})
    elif emo == "슬픔": style.update({"ink_base":40,"blur":2.0})
    elif emo == "분노": style.update({"ink_base":5,"blur":0.8})
    elif emo == "평온": style.update({"ink_base":25,"blur":1.2})
    elif emo == "열정": style.update({"ink_base":8,"blur":1.0})
    elif emo == "냉정": style.update({"ink_base":30,"blur":1.4})
    for k in style: style[k]*=weight
    return style

# ------------------------------------------------------------
# 랜덤 배경 이미지 선택
# ------------------------------------------------------------
def get_random_poster_bg(emotion: str):
    emo_dir = os.path.join(POSTER_DIR, emotion)
    if os.path.exists(emo_dir):
        files = [f for f in os.listdir(emo_dir) if f.lower().endswith((".jpg",".png"))]
        if files:
            img = Image.open(os.path.join(emo_dir, random.choice(files))).resize(POSTER_SIZE).convert("RGB")
            overlay = Image.new("RGBA", POSTER_SIZE, (0,0,0,120))
            img = img.convert("RGBA"); img.alpha_composite(overlay)
            return img.convert("RGB")
    return Image.new("RGB", POSTER_SIZE, (10,10,15))

# ------------------------------------------------------------
# 폰트 헬퍼
# ------------------------------------------------------------
def get_font(size): return ImageFont.truetype(FONT_PATH, size=size)

# ------------------------------------------------------------
# 붓터치 + 한지 질감 렌더링
# ------------------------------------------------------------
def render_poster(text: str, emotion: str, style: dict):
    bg = get_random_poster_bg(emotion).convert("RGB")

    # 한지 질감
    hanji_path = os.path.join(STATIC_DIR, "hanji_texture.jpg")
    if os.path.exists(hanji_path):
        hanji = Image.open(hanji_path).convert("L").resize(POSTER_SIZE)
        hanji = ImageOps.autocontrast(hanji)
        bg = ImageChops.multiply(bg, Image.merge("RGB",(hanji,hanji,hanji)))
    else:
        noise = Image.effect_noise(POSTER_SIZE, 20).convert("L")
        noise_rgb = Image.merge("RGB",(noise,noise,noise))
        bg = ImageChops.multiply(bg, noise_rgb)

    canvas = bg.convert("RGBA")
    draw = ImageDraw.Draw(canvas)
    font = get_font(int(POSTER_SIZE[0]*0.25))
    lines = textwrap.wrap(text, width=5)
    total_h = sum(draw.textbbox((0,0),l,font=font)[3] for l in lines)+80*(len(lines)-1)
    y=(POSTER_SIZE[1]-total_h)//2

    # 마스크 생성
    text_mask = Image.new("L", POSTER_SIZE, 0)
    mask_draw = ImageDraw.Draw(text_mask)
    for line in lines:
        w,h = mask_draw.textbbox((0,0),line,font=font)[2:]
        x=(POSTER_SIZE[0]-w)//2
        mask_draw.text((x,y),line,fill=255,font=font)
        y+=h+80

    blur_radius=float(style.get("blur",1.0))
    bleed_mask=text_mask.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    bleed_mask=ImageEnhance.Brightness(bleed_mask).enhance(1.1)

    brush_noise=Image.effect_noise(POSTER_SIZE,25).convert("L")
    brush_texture=ImageEnhance.Contrast(brush_noise).enhance(2.5)

    ink_intensity=max(0,60-int(style["ink_base"]*5))
    ink_color=(ink_intensity,ink_intensity,ink_intensity)
    text_layer=Image.new("RGBA",POSTER_SIZE,(0,0,0,0))
    ImageDraw.Draw(text_layer).bitmap((0,0),text_mask,fill=ink_color)

    merged=ImageChops.add(text_layer,Image.merge("RGBA",(bleed_mask,)*4))
    merged=ImageChops.multiply(merged,Image.merge("RGBA",(brush_texture,)*4))
    canvas.alpha_composite(merged)

    if os.path.exists(SEAL_PATH):
        seal=Image.open(SEAL_PATH).convert("RGBA").resize((90,90))
        canvas.alpha_composite(seal,(POSTER_SIZE[0]-120,POSTER_SIZE[1]-130))

    canvas=ImageEnhance.Contrast(canvas).enhance(1.5)
    canvas=ImageEnhance.Sharpness(canvas).enhance(2.0)
    return canvas.convert("RGB")

# ------------------------------------------------------------
# DB 저장 및 피드백
# ------------------------------------------------------------
def save_generation(text, emo, filename):
    con=db_conn();cur=con.cursor()
    cur.execute("INSERT INTO generations(text, emo, filename, created_at) VALUES (?,?,?,?)",
        (text, emo, filename, datetime.datetime.utcnow().isoformat()))
    gid=cur.lastrowid;con.commit();con.close();return gid

def save_feedback(gid, satisfied):
    con=db_conn();cur=con.cursor()
    cur.execute("INSERT INTO feedback(generation_id,satisfied,created_at) VALUES (?,?,?)",
        (gid,int(satisfied),datetime.datetime.utcnow().isoformat()))
    con.commit();con.close()

# ------------------------------------------------------------
# 튜닝 주기적 업데이트
# ------------------------------------------------------------
def training_job():
    con=db_conn();cur=con.cursor()
    cur.execute("SELECT emo, satisfied FROM feedback f JOIN generations g ON f.generation_id=g.id")
    rows=cur.fetchall();con.close()
    if not rows:return
    tuning=load_tuning();counts={emo:[0,0] for emo in EMO_LABELS}
    for emo,sat in rows:
        if emo in counts:
            counts[emo][0]+=1;counts[emo][1]+=int(sat)
    for emo,(n,s) in counts.items():
        if n>0:
            rate=s/n
            target=0.7+0.6*rate
            tuning[emo]=0.7*tuning.get(emo,1.0)+0.3*target
    save_tuning(tuning)
    print("[trainer] updated:",tuning)

scheduler=BackgroundScheduler()
scheduler.add_job(training_job,"interval",minutes=10)
scheduler.start()

# ------------------------------------------------------------
# 공통 Navbar
# ------------------------------------------------------------
def navbar():
    return """
    <nav class="navbar navbar-expand-lg navbar-dark bg-dark sticky-top">
      <div class="container-fluid">
        <a class="navbar-brand fw-bold" href="/">🖋️ AI Calligraphy</a>
        <button class="navbar-toggler" type="button" data-bs-toggle="collapse" data-bs-target="#nav"
                aria-controls="nav" aria-expanded="false" aria-label="Toggle navigation">
          <span class="navbar-toggler-icon"></span>
        </button>
        <div class="collapse navbar-collapse" id="nav">
          <ul class="navbar-nav ms-auto">
            <li class="nav-item"><a class="nav-link" href="/">Home</a></li>
            <li class="nav-item"><a class="nav-link" href="/admin">Admin</a></li>
          </ul>
        </div>
      </div>
    </nav>
    """

# ------------------------------------------------------------
# Routes
# ------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
def index():
    return f"""
    <html><head><meta charset='utf-8'>
    <meta name='viewport' content='width=device-width,initial-scale=1'>
    <link href='https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css' rel='stylesheet'>
    <script src='https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js'></script>
    </head><body class='bg-light'>{navbar()}
      <div class='container py-5'>
        <h2 class='text-center mb-4'>감성 캘리그래피 포스터 생성</h2>
        <form method='post' action='/render' class='card p-4 shadow-sm'>
          <label class='form-label'>문장/단어</label>
          <textarea name='text' class='form-control mb-3' rows='3'
           placeholder='예: 타오르는 열정, 고요한 마음'></textarea>
          <button type='submit' class='btn btn-dark w-100'>이미지 생성</button>
        </form>
      </div></body></html>
    """

@app.post("/render", response_class=HTMLResponse)
def render_endpoint(text: str = Form(...)):
    emo_score = analyze_emotion(text)
    dominant = max(emo_score, key=emo_score.get)
    tuning = load_tuning()
    style = emotion_to_style(emo_score, tuning)
    img = render_poster(text, dominant, style)

    fname = f"{time.strftime('%Y%m%d_%H%M%S')}.png"
    path = os.path.join("outputs", fname)
    img.save(path)
    gid = save_generation(text, dominant, fname)
    buf = io.BytesIO(); img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()

    rows = "".join([f"<tr><td>{k}</td><td>{emo_score[k]:.2f}</td><td>{tuning[k]:.2f}</td></tr>" for k in EMO_LABELS])
    return f"""
    <html><head><meta charset='utf-8'>
    <link href='https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css' rel='stylesheet'>
    </head><body class='bg-light'>{navbar()}
    <div class='container py-4'>
      <img class='img-fluid border rounded mb-3' src='data:image/png;base64,{b64}'/>
      <div class='d-flex gap-2'>
        <form method='post' action='/feedback'><input type='hidden' name='generation_id' value='{gid}'/>
        <input type='hidden' name='satisfied' value='1'/><button class='btn btn-success w-100'>😊 만족</button></form>
        <form method='post' action='/feedback'><input type='hidden' name='generation_id' value='{gid}'/>
        <input type='hidden' name='satisfied' value='0'/><button class='btn btn-danger w-100'>😞 불만족</button></form>
      </div>
      <div class='card mt-4'><div class='card-header'>감정 분석 & 튜닝</div>
        <table class='table table-sm mb-0'><thead><tr><th>감정</th><th>확률</th><th>튜닝</th></tr></thead><tbody>{rows}</tbody></table>
      </div></div></body></html>
    """

@app.post("/feedback")
def feedback_endpoint(generation_id: int = Form(...), satisfied: int = Form(...)):
    save_feedback(generation_id, satisfied)
    return RedirectResponse(url="/admin", status_code=303)

@app.get("/admin", response_class=HTMLResponse)
def admin():
    con=db_conn();cur=con.cursor()
    cur.execute("SELECT COUNT(*),SUM(satisfied) FROM feedback");n,s=cur.fetchone()
    rate=(s or 0)/(n or 1)
    cur.execute("SELECT g.id,g.text,g.emo,g.filename,g.created_at,(SELECT satisfied FROM feedback f WHERE f.generation_id=g.id ORDER BY id DESC LIMIT 1) FROM generations g ORDER BY g.id DESC LIMIT 30")
    rows=cur.fetchall();con.close()
    trs="".join([f"<tr><td>{i}</td><td>{emo}</td><td>{t}</td><td>{f}</td><td>{ts}</td><td>{'✅' if fb==1 else ('❌' if fb==0 else '—')}</td></tr>" for i,t,emo,f,ts,fb in rows])
    return f"""
    <html><head><meta charset='utf-8'>
    <link href='https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css' rel='stylesheet'>
    </head><body class='bg-dark text-light'>{navbar()}
    <div class='container py-5'>
      <h4>피드백 통계</h4>
      <p>총 {n or 0}건, 만족률 {rate:.1%}</p>
      <table class='table table-dark table-striped table-sm'>
        <thead><tr><th>ID</th><th>감정</th><th>텍스트</th><th>파일</th><th>시각</th><th>최근피드백</th></tr></thead>
        <tbody>{trs}</tbody>
      </table>
      <a href='/' class='btn btn-secondary'>← 생성 페이지</a>
    </div></body></html>
    """
