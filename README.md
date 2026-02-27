# 26_winter_proj

### BITAMIN 16, 17기 Winter Project  
KOSPI 기반 공포·탐욕 지표 및 오실레이터 분석
NAVER 댓글 수집 후 감성분석 통해 K-Fear&Greed Index 구축하고, 시계열 예측 실험 진행

## 📌 Project Overview
- KOSPI 지수와 시장 심리 지표(K-Fear & Greed)를 활용한 분석
- NAVER 기사 중 핵심 키워드 선정 후 댓글 감성분석을 모델에 추가
- 시계열 데이터 기반 점수화 및 예측 실험
- 지수 예측 및 투자 전략 검증을 수행한 프로젝트

## 🎯 Objective
- 기존 CNN Fear & Greed Index의 구조를 참고하여 한국 시장(KOSPI)에 특화된 공포·탐욕 지표 개발
- 뉴스 댓글 감성 정보를 정량화하여 기존 거시/기술적 지표와 통합
- K-FGI 기반 수익률 예측 모델 및 전략 백테스트 수행

## 💻 Pipeline
![Pipeline](https://github.com/user-attachments/assets/0d14ddb9-a0ba-40cf-8dc5-1510b4e89f33)

## 📂 Directory Structure
```text
26_winter_proj/
│
├── 1_KFGI_sub_index/        # 기존 FGI 7개 sub_index 생성
├── 2_Naver_crawling/        # 네이버 기사 및 댓글 크롤링
├── 3_filtering_final/       # 정치/비주식 댓글 제거
├── 4_sentiment_analysis/    # 금융 특화 감성 분석
├── 5_merge_to_final_csv/    # 감성 + 거시지표 병합
├── 6_KFGI_weight/           # Ridge 기반 KFGI 가중치 산출
├── 7_modeling_final/        # 시계열 예측 모델
├── 8_dashboard/             # Streamlit 대시보드
│
├── data/
│   ├── KFG/                 # sub_index 원본 및 중간 산출물
│   └── NAVER/
│       ├── article/         # 기사 목록
│       ├── comments/        # 댓글 원본
│       ├── final_filtered/  # 필터링 완료 댓글
│       └── sentiment_final/ # 일별 감성지표
│
├── documents/               # 참고 논문 및 자료
├── README.md
└── .gitignore
```

## 🧪 Environment
### 🐍 언어
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
  
### 📚 주요 패키지
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)
![Requests](https://img.shields.io/badge/Requests-2CA5E0?style=for-the-badge&logo=python&logoColor=white)
![BeautifulSoup](https://img.shields.io/badge/BeautifulSoup-4B8BBE?style=for-the-badge&logo=python&logoColor=white)


## 🧠 Methodology

**1️⃣ Sub-Index Construction**
* KOSPI 기반 7개 시장 심리 관련 지표 수집
* 0–100 정규화
* 시계열 정렬 및 결측 처리

**2️⃣ Sentiment Extraction**
* NAVER 뉴스 기사 크롤링
* 댓글 필터링 (정치/비주식 제거 + 독성 제거)
* 금융 특화 BERT 기반 감성 확률 산출
* 일별 감성지표 생성

**3️⃣ Index Aggregation**
* Ridge 회귀 기반 가중치 추정
* PCA / Factor Analysis 비교
* K-FGI 지수 산출

**4️⃣ Forecasting & Strategy**
* Multi-horizon 예측 실험
* Directional Accuracy 분석
* 전략 수익률 및 샤프지수 비교


**5️⃣ Dashboard Visualization**
* 실행 : 8_dashboard/app.py
* 입력 : KFGI_final.csv, 예측 결과 파일
* 출력 : Streamlit 기반 시각화 대시보드, KFGI 지수 추이 및 투자 전략 결과 확인


## ⚠️ Notes
- `data/` 폴더의 csv 파일은 GitHub에 업로드되지 않습니다.
- 데이터는 별도 경로에서 관리됩니다.