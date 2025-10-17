## 수집한 12장을 기반으로
## → Augmentation으로 수천 장처럼 다양하게 변형
## → CNN 모델로 “고양이냐 / 아니냐”를 학습
## → On-the-fly augmentation을 사용하여 자동 변형

## 폴더, 파일 구조
PYTHON-003/
│
├─ app_cat_web.py        ← FastAPI 웹 앱 (Bootstrap + 미리보기 + 예측)
├─ app_cat_infer.py      ← FastAPI API 버전 (HTML 없이 JSON 응답)
├─ infer_cats.py         ← 콘솔 버전 단일 이미지 예측 (CLI)
├─ train_cats.py         ← CNN 모델 학습 (ResNet + on-the-fly augmentation)
├─ get_cat.py            ← 웹에서 고양이/비고양이 이미지 크롤링
├─ log_anal.py           ← 학습 로그 그래프 시각화
│
├─ models/               ← 학습된 모델 및 클래스 정보 저장
│   ├─ cats_resnet18.pt
│   └─ class_to_idx.json
│
├─ dataset/              ← 학습용 이미지 (고양이/비고양이)
│   ├─ cats/
│   └─ not_cats/
│
├─ logs/                 ← 학습 로그 (jsonl)
│
├─ templates/            ← 웹 템플릿 (HTML)
│   └─ index.html
│
├─ static/               ← 정적 파일(css/js/img)
│
├─ venv/                 ← 가상환경 (Python 의존성)
└─ Readme.md

## 상세 목록

| 구분                              | 파일/폴더                                     | 역할 요약                    | 주요 기능                                                                                                                     |
| ------------------------------- | ----------------------------------------- | ------------------------ | ------------------------------------------------------------------------------------------------------------------------- |
| 🧠 **모델 학습**                    | `train_cats.py`                           | CNN(ResNet18) 모델 학습 스크립트 | - `dataset/cats` / `dataset/not_cats` 기반 학습<br>- 실시간(on-the-fly) augmentation 적용<br>- 최적 모델 저장(`models/cats_resnet18.pt`) |
| 🐱 **단일 예측 (CLI)**              | `infer_cats.py`                           | 콘솔 기반 예측 스크립트            | - `python infer_cats.py --image sample1.jpg` 실행<br>- 라벨(`cats` or `not_cats`)과 confidence 출력                              |
| 🌐 **FastAPI API 서버 (JSON)**    | `app_cat_infer.py`                        | JSON 응답 전용 버전            | - `/predict` 엔드포인트<br>- 이미지 업로드 → JSON 결과 반환                                                                              |
| 🖥 **FastAPI 웹 버전 (Bootstrap)** | `app_cat_web.py`                          | HTML 기반 완성형 웹앱           | - `/` 업로드 폼<br>- 미리보기 표시 + 예측 결과 시각화<br>- Bootstrap 디자인                                                                   |
| 📦 **데이터 수집**                   | `get_cat.py`                              | DuckDuckGo 이미지 크롤러       | - “cat photo” / “car photo” 검색<br>- `dataset/cats/`, `dataset/not_cats/` 폴더에 이미지 저장                                       |
| 📊 **학습 로그 분석**                 | `log_anal.py` (또는 `plot_training_log.py`) | 학습 결과 시각화                | - `logs/train_log.jsonl` 읽어서<br>- Train/Val Loss & Accuracy 그래프 출력                                                        |
| 🧩 **모델 폴더**                    | `models/`                                 | 학습된 가중치 및 클래스 매핑         | - `.pt`: PyTorch 모델 가중치<br>- `.json`: 클래스 이름 ↔ 인덱스 매핑                                                                     |
| 🐾 **데이터셋 폴더**                  | `dataset/`                                | 학습용 이미지                  | - cats/, not_cats/ 서브폴더                                                                                                   |
| 🧾 **로그 폴더**                    | `logs/`                                   | 학습 과정 로그 저장              | - 각 epoch의 손실, 정확도 기록(jsonl)                                                                                              |
| 🎨 **템플릿 폴더**                   | `templates/`                              | HTML 템플릿 저장              | - FastAPI가 렌더링하는 `index.html` (Jinja2 문법)                                                                                 |
| 🧱 **정적 파일 폴더**                 | `static/`                                 | CSS/JS 이미지 등 정적 리소스      | - 현재는 비어있지만 향후 Bootstrap 커스터마이징용                                                                                          |
| 🧮 **가상환경 폴더**                  | `venv/`                                   | Python 패키지 격리 환경         | - pip install 패키지들이 여기에 설치됨                                                                                               |
| 🧾 **문서 파일**                    | `Readme.md`                               | 프로젝트 설명 문서               | - 실행 방법, 환경 세팅 요약                                                                                                         |
