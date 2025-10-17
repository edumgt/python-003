# infer_cats.py
# --------------------------------------------------
# 단일 이미지(cat vs not-cat) 예측용 스크립트
# 사용법:
#   python infer_cats.py --image path/to/image.jpg
# --------------------------------------------------

import torch
from torchvision import models, transforms
from PIL import Image
import json, argparse, os

def load_model(model_path="models/cats_resnet18.pt", class_map_path="models/class_to_idx.json"):
    # 클래스 인덱스 매핑 로드
    with open(class_map_path, "r", encoding="utf-8") as f:
        class_to_idx = json.load(f)
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    # 모델 구조 동일하게 로드
    model = models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, len(idx_to_class))
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    return model, idx_to_class

def predict_image(model, idx_to_class, image_path, img_size=224):
    # 이미지 전처리 (train과 동일한 정규화)
    preprocess = transforms.Compose([
        transforms.Resize(int(img_size * 1.15)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # 이미지 불러오기
    img = Image.open(image_path).convert("RGB")
    img_t = preprocess(img).unsqueeze(0)  # 배치 차원 추가

    # 예측
    with torch.no_grad():
        logits = model(img_t)
        probs = torch.nn.functional.softmax(logits, dim=1)
        conf, pred_idx = torch.max(probs, dim=1)
        label = idx_to_class[pred_idx.item()]
        return label, conf.item()

def main():
    parser = argparse.ArgumentParser(description="단일 이미지 예측 (Cat vs Not-Cat)")
    parser.add_argument("--image", type=str, required=True, help="이미지 경로")
    args = parser.parse_args()

    if not os.path.exists(args.image):
        print(f"❌ 이미지 파일을 찾을 수 없습니다: {args.image}")
        return

    model, idx_to_class = load_model()
    label, confidence = predict_image(model, idx_to_class, args.image)

    print(f"\n🖼️  입력 이미지: {args.image}")
    print(f"📊  예측 결과: {label.upper()} (confidence={confidence:.3f})")

if __name__ == "__main__":
    main()
