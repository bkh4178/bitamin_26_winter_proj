# 26_winter_proj

비타민 16기 겨울 프로젝트  
KOSPI 기반 공포·탐욕 지표 및 오실레이터 분석
NAVER 댓글 수집 후 감성분석 통해 K-Fear&Greed Index 구축하고, 시계열 예측 실험 진행

## 📌 Project Overview
- KOSPI 지수와 시장 심리 지표(Fear & Greed)를 활용한 분석
- NAVER 기사 중 핵심 키워드 선정 후 댓글 감성분석을 모델에 추가
- 시계열 데이터 기반 점수화 및 예측 실험

## 📂 Directory Structure
```text
26_winter_proj/
├── data/
│   ├── KFG/          # Fear & Greed 관련 지표 데이터 (gitignore, 로컬 전용)
│   └── NAVER/
│     ├─ article/                  # 기사 리스트 CSV
│     └─ comments/                 # 댓글 CSV
├─ Naver_comments/                 # 네이버 기사/댓글 수집 및 전처리 코드
│  ├─ test/                        # 실험/테스트 코드 및 출력물
├─ Oscillator/     # 오실레이터 분석
├─ documents/      # 참고 논문 및 자료
├─ README.md
└─ .gitignore
```

## 🧪 Environment
- Python 3.x
- 주요 패키지: pandas, numpy, matplotlib, scikit-learn, pykrx, requests, beautifulsoup4

🚀 How to Run (Local)

data/ 폴더는 GitHub에 올라가지 않으므로, 실행하면 로컬에 결과 CSV가 생성됩니다.

1.	기사 수집
* 실행: Naver_comments/article_crawling.py
* 출력: data/NAVER/article/articles_2025_financial.csv

2.	댓글 수집

* 실행: Naver_comments/comments_crawling.py
* 입력: data/NAVER/article/articles_2025_financial.csv
* 출력: data/NAVER/comments/comments_2025.csv

## ⚠️ Notes
- `data/` 폴더의 csv 파일은 GitHub에 업로드되지 않습니다.
- 데이터는 별도 경로에서 관리됩니다.
