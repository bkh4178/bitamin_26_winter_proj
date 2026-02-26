# 26_winter_proj

비타민 16, 17기 겨울 프로젝트  
KOSPI 기반 공포·탐욕 지표 및 오실레이터 분석
NAVER 댓글 수집 후 감성분석 통해 K-Fear&Greed Index 구축하고, 시계열 예측 실험 진행

## 📌 Project Overview
- KOSPI 지수와 시장 심리 지표(Fear & Greed)를 활용한 분석
- NAVER 기사 중 핵심 키워드 선정 후 댓글 감성분석을 모델에 추가
- 시계열 데이터 기반 점수화 및 예측 실험

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

🚀 How to Run (Local)

data/ 폴더는 GitHub에 올라가지 않으므로, 실행하면 로컬에 결과 CSV가 생성됩니다.

**1.	7 sub index 수집**

**2.	기사 & 댓글 수집**

* 실행: Naver_crawling/crawling_naver_{year}.py
* 출력: data/NAVER/comments/comments_{year}_top5.csv & data/NAVER/article/news_{year}_top5.csv

**3.  댓글 필터링**
* 실행 : filtering_final/1_politic_filter.py, 2_apply_toxicity_naver.py, final_filter.py
* 입력 : raw_comments_data(연도별), comments_political_removed_{year}.csv, comments_toxicity_kept_{year}.csv
* 출력 : comments_political_removed_{year}.csv, comments_toxicity_kept_{year}.csv, comments_final_stock_only_{year}.csv

**4.  댓글 감성분석**
* 실행 : sentiment_analysis/compute_sentiment.py
* 입력 : comments_final_stock_only_{year}.csv
* 출력 : comments_sentiment_{year}.csv


**5. 감성지표 + 거시지표 병합**
* 실행 : 5_merge_to_final_csv/sub_index_merge.py
* 입력 : KFG sub_index 7개
	* comments_sentiment_{year}.csv
	* 출력 : KFG_final_2.csv (일별 통합 데이터)


**6. KFGI 지수 산출 (가중치 계산)**
* 실행 : 6_KFGI_weight/index.py (또는 PCA/FA 버전 스크립트)
* 입력 :
	* KFG_with_sentiment.csv
	* 출력 :
	* KFGI_final.csv
	* 가중치 계수 파일 (모델별)

⸻

**7. 수익률 예측 모델링**
* 실행 : 7_modeling_final/ 내 예측 스크립트
(예: multi-horizon, validation 등)
* 입력 :
	* KFGI_final.csv
	* KOSPI 로그수익률 데이터
	* 출력 : prediction_multihorizon.csv
	* 모델 성능 지표 (RMSE, Directional Accuracy 등)

⸻

**8. 대시보드 시각화**
* 실행 : 8_dashboard/app.py
* 입력 : KFGI_final.csv
* 예측 결과 파일
* 출력 : Streamlit 기반 시각화 대시보드, KFGI 지수 추이 및 투자 전략 결과 확인


## ⚠️ Notes
- `data/` 폴더의 csv 파일은 GitHub에 업로드되지 않습니다.
- 데이터는 별도 경로에서 관리됩니다.


