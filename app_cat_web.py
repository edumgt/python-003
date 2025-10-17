# app_cat_web.py
# --------------------------------------------------------
# FastAPI + Bootstrap + Torch 기반 웹 앱
# 고양이 vs 고양이 아님 예측 + 이미지 미리보기 표시
# 실행: uvicorn app_cat_web:app --host 0.0.0.0 --port 8000 --reload
# --------------------------------------------------------

from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from torchvision import models, transforms
from PIL import Image
import torch, json, io, os, base64

# --------------------------------------------------------
# 초기 설정
# --------------------------------------------------------
app = FastAPI(title="🐱 Cat Classifier Web App")
templates = Jinja2Templates(directory="templates")

# static 폴더 (Bootstrap, CSS, JS용)
app.mount("/static", StaticFiles(directory="static"), name="static")

MODEL_PATH = "models/cats_resnet18.pt"
CLASS_MAP_PATH = "models/class_to_idx.json"

# --------------------------------------------------------
# 모델 및 매핑 로드
# --------------------------------------------------------
with open(CLASS_MAP_PATH, "r", encoding="utf-8") as f:
    class_to_idx = json.load(f)
idx_to_class = {v: k for k, v in class_to_idx.items()}

model = models.resnet18(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, len(idx_to_class))
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()

# --------------------------------------------------------
# 이미지 전처리 함수
# --------------------------------------------------------
preprocess = transforms.Compose([
    transforms.Resize(int(224 * 1.15)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

def predict_image(image_bytes: bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img_t = preprocess(img).unsqueeze(0)
    with torch.no_grad():
        logits = model(img_t)
        probs = torch.nn.functional.softmax(logits, dim=1)
        conf, pred_idx = torch.max(probs, dim=1)
        label = idx_to_class[pred_idx.item()]
        return label, float(conf.item())

# --------------------------------------------------------
# 메인 페이지
# --------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# --------------------------------------------------------
# 이미지 업로드 및 예측
# --------------------------------------------------------
@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request, file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        label, confidence = predict_image(image_bytes)
        img_b64 = base64.b64encode(image_bytes).decode("utf-8")
        img_src = f"data:image/jpeg;base64,{img_b64}"
        return templates.TemplateResponse("index.html", {
            "request": request,
            "result": {"label": label, "confidence": round(confidence, 3)},
            "image_data": img_src
        })
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
