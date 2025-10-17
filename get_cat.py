# ✅ get_cat_v3.py
# DuckDuckGo(ddgs) 최신버전 호환 + rate limit 방지

from ddgs import DDGS
import requests, os, time, random

# 📁 폴더 생성
os.makedirs("dataset/cats", exist_ok=True)
os.makedirs("dataset/not_cats", exist_ok=True)

# ✅ 공통 다운로드 함수
def download_images(query, count, folder):
    print(f"\n🔍 '{query}' 이미지 {count}장 다운로드 시작...")
    ddg = DDGS()

    # ✅ 최신 버전에서는 'query'만 인자로 전달
    results = ddg.images(query, max_results=count)

    for i, r in enumerate(results):
        try:
            url = r.get("image")
            if not url:
                continue

            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                filename = f"{folder}/{query.replace(' ', '_')}_{i}.jpg"
                with open(filename, "wb") as f:
                    f.write(response.content)
                print(f"✅ 저장됨: {filename}")
            else:
                print(f"⚠️ 상태 코드 {response.status_code} : {url}")
        except Exception as e:
            print(f"❌ 오류: {e}")
        # ✅ 서버 과부하 방지
        time.sleep(random.uniform(1.0, 2.5))

# 🐱 고양이 10장 다운로드
download_images("cat photo", 10, "dataset/cats")

# 🚗 고양이 아님 2장 다운로드
download_images("car photo", 2, "dataset/not_cats")

print("\n🎉 다운로드 완료! dataset/cats, dataset/not_cats 폴더를 확인하세요.")
