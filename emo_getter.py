# emotion_image_crawler.py
# ------------------------------------------------------------
# API Key 없이 무료로 감정별 배경 이미지 크롤링
# - Picsum Photos (https://picsum.photos)
# ------------------------------------------------------------
import os, random, requests
from io import BytesIO
from PIL import Image

POSTER_DIR = "static/poster_bg"
os.makedirs(POSTER_DIR, exist_ok=True)

# 감정별로 적합한 테마를 지정
EMO_BG_KEYWORDS = {
    "기쁨": ["nature", "sunrise", "flower", "colorful", "bright"],
    "슬픔": ["rain", "city", "dark", "blue", "fog"],
    "분노": ["fire", "storm", "red", "lava"],
    "평온": ["sea", "sky", "mountain", "forest"],
    "열정": ["concert", "sunset", "red-light", "energy"],
    "냉정": ["snow", "ice", "night", "minimal"]
}

def download_emotion_backgrounds(per_emotion=5):
    """
    각 감정별 폴더에 랜덤 이미지 저장 (1080x1920, 세로형)
    """
    for emo, keywords in EMO_BG_KEYWORDS.items():
        emo_dir = os.path.join(POSTER_DIR, emo)
        os.makedirs(emo_dir, exist_ok=True)
        print(f"🎨 [{emo}] 이미지 다운로드 중...")

        for i in range(per_emotion):
            # 랜덤 키워드 + 랜덤 시드
            kw = random.choice(keywords)
            seed = random.randint(1, 99999)
            # Picsum 랜덤 이미지 (1080x1920)
            url = f"https://picsum.photos/seed/{kw}-{seed}/1080/1920"
            try:
                res = requests.get(url, timeout=10)
                if res.status_code == 200:
                    img = Image.open(BytesIO(res.content))
                    save_path = os.path.join(emo_dir, f"{emo}_{i+1}.jpg")
                    img.save(save_path)
                    print(f"  ✅ {save_path}")
            except Exception as e:
                print(f"  ❌ {emo} {kw}: {e}")

if __name__ == "__main__":
    download_emotion_backgrounds()
