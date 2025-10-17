# app_calligraphy_master.py
# ----------------------------------------------------
# 🖌️ 강렬한 붓글씨 + 한지 질감 + 붉은 낙관(도장) 버전
# ----------------------------------------------------
# 실행: uvicorn app_calligraphy_master:app --reload
# ----------------------------------------------------

from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import numpy as np
import io, os, random, textwrap

app = FastAPI(title="K-Calligraphy Master (Brush + Hanji + Seal)")

# ====== 기본 설정 ======
FONT_PATH = "fonts/NanumBrush.ttf"  # ✅ 한글 붓글씨 폰트 권장
SEAL_PATH = "static/seal_red.png"   # ✅ 도장 PNG (투명배경)
HANJI_PATH = "static/hanji_texture.jpg"  # ✅ 한지 질감 이미지
CANVAS_W, CANVAS_H = 1600, 900
MARGIN = 120
BASE_FONT_SIZE = 150
LINE_SPACING = 1.45
# ========================

def get_font(size):
    return ImageFont.truetype(FONT_PATH, size=size)

def make_hanji_background() -> Image.Image:
    """한지 질감 배경 생성"""
    if os.path.exists(HANJI_PATH):
        hanji = Image.open(HANJI_PATH).convert("RGB").resize((CANVAS_W, CANVAS_H))
    else:
        hanji = Image.new("RGB", (CANVAS_W, CANVAS_H), (250, 248, 240))
        # 가짜 질감 노이즈
        noise = np.random.normal(0, 12, (CANVAS_H, CANVAS_W, 3))
        arr = np.clip(np.array(hanji).astype("float32") + noise, 0, 255).astype("uint8")
        hanji = Image.fromarray(arr)
    return hanji

def add_ink_effect(img: Image.Image) -> Image.Image:
    """먹 번짐 + 강한 질감 효과"""
    arr = np.array(img).astype("float32")
    arr = np.clip((arr - 40) * 1.4, 0, 255)
    img = Image.fromarray(arr.astype("uint8"))
    img = img.filter(ImageFilter.GaussianBlur(radius=1.6))
    noise = np.random.normal(0, 25, arr.shape)
    arr = np.clip(arr + noise, 0, 255).astype("uint8")
    return Image.fromarray(arr)

def apply_seal(canvas: Image.Image):
    """붉은 낙관 도장 합성"""
    if not os.path.exists(SEAL_PATH):
        return canvas
    seal = Image.open(SEAL_PATH).convert("RGBA")
    seal = seal.resize((180, 180))
    x = CANVAS_W - seal.width - 100
    y = CANVAS_H - seal.height - 80
    canvas.paste(seal, (x, y), seal)
    return canvas

def render_calligraphy(text: str) -> Image.Image:
    """캘리그래피 스타일 텍스트 렌더링"""
    canvas = make_hanji_background().convert("RGBA")
    draw = ImageDraw.Draw(canvas)

    avg_font = get_font(BASE_FONT_SIZE)
    max_text_width = CANVAS_W - 2 * MARGIN
    avg_char = draw.textlength("한", font=avg_font)
    max_chars_per_line = max(1, int(max_text_width / avg_char))
    wrapped_lines = []
    for line in text.splitlines():
        wrapped_lines.extend(textwrap.wrap(line, width=max_chars_per_line) or [" "])

    total_h = sum(draw.textbbox((0, 0), line, font=avg_font)[3] * LINE_SPACING for line in wrapped_lines)
    y = int((CANVAS_H - total_h) / 2)

    for line in wrapped_lines:
        x = MARGIN
        for ch in line:
            scale = random.uniform(0.85, 1.35)
            angle = random.uniform(-8, 8)
            offset_x = random.randint(-6, 6)
            offset_y = random.randint(-6, 6)
            font = get_font(int(BASE_FONT_SIZE * scale))
            temp = Image.new("RGBA", (400, 400), (255, 255, 255, 0))
            dtemp = ImageDraw.Draw(temp)
            ink_color = random.randint(0, 30)
            dtemp.text((50, 100), ch, font=font, fill=(ink_color, ink_color, ink_color, 255))
            rotated = temp.rotate(angle, resample=Image.BICUBIC, expand=True)
            bbox = rotated.getbbox()
            if bbox:
                char_img = rotated.crop(bbox)
                canvas.alpha_composite(char_img, (x + offset_x, y + offset_y))
            w = draw.textbbox((0, 0), ch, font=font)[2]
            x += int(w * random.uniform(0.9, 1.1))
        _, _, _, h = draw.textbbox((0, 0), line, font=avg_font)
        y += int(h * LINE_SPACING)

    canvas = add_ink_effect(canvas)
    canvas = apply_seal(canvas)
    return canvas.convert("RGB")

# ====== FastAPI 라우트 ======
@app.get("/", response_class=HTMLResponse)
def index():
    return """
    <html>
    <head>
      <meta charset="utf-8"/>
      <title>🖋️ 한글 붓글씨 생성기 (Master)</title>
      <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    </head>
    <body class="bg-light">
      <div class="container py-5 text-center">
        <h1 class="mb-4">🖌️ 한글 캘리그래피 생성기 (한지 + 낙관)</h1>
        <form method="post" action="/render" class="card p-4 shadow-sm mx-auto" style="max-width:900px;">
          <textarea name="text" class="form-control mb-3" rows="4" placeholder="예: 봄바람에 흩날리는 벚꽃처럼..."></textarea>
          <button type="submit" class="btn btn-dark">이미지 생성</button>
        </form>
      </div>
    </body>
    </html>
    """

@app.post("/render")
def render_endpoint(text: str = Form(...)):
    try:
        img = render_calligraphy(text)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)
        return StreamingResponse(buf, media_type="image/png")
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
