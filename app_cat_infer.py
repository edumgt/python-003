# app_cat_infer.py
# --------------------------------------------------------
# FastAPI 기반 이미지 업로드 + 고양이/비고양이 판별 API
# 실행: uvicorn app_cat_infer:app --host 0.0.0.0 --port 8000 --reload
# --------------------------------------------------------

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse, HTMLResponse
from torchvision import models, transforms
from PIL import Image
import torch
import io, json, os

app = FastAPI(title="🐱 Cat Classifier API", description="Upload an image to check if it's a cat or not.")

# --------------------------------------------------------
# 1️⃣ 모델 및 클래스 로드
# --------------------------------------------------------
MODEL_PATH = "models/cats_resnet18.pt"
CLASS_MAP_PATH = "models/class_to_idx.json"

# 모델 및 매핑 로드
with open(CLASS_MAP_PATH, "r", encoding="utf-8") as f:
    class_to_idx = json.load(f)
idx_to_class = {v: k for k, v in class_to_idx.items()}

model = models.resnet18(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, len(idx_to_class))
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()

# --------------------------------------------------------
# 2️⃣ 이미지 전처리 함수
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
# 3️⃣ HTML 업로드 폼
# --------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
async def home():
    return """
    <html>
        <head><title>Cat Classifier 🐱</title></head>
        <body style="font-family:sans-serif; text-align:center;">
            <h1>🐱 Cat vs Not Cat Classifier</h1>
            <form action="/predict" enctype="multipart/form-data" method="post">
                <input name="file" type="file" accept="image/*">
                <input type="submit" value="Upload and Predict">
            </form>
        </body>
    </html>
    """

# --------------------------------------------------------
# 4️⃣ 이미지 업로드 및 예측 API
# --------------------------------------------------------
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        label, confidence = predict_image(image_bytes)
        return JSONResponse({
            "filename": file.filename,
            "label": label,
            "confidence": round(confidence, 3)
        })
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
