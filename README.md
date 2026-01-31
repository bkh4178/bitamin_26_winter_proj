# 26_winter_proj

비타민 26기 겨울 프로젝트  
KOSPI 기반 공포·탐욕 지표 및 오실레이터 분석

## 📌 Project Overview
- KOSPI 지수와 시장 심리 지표(Fear & Greed, Oscillator)를 활용한 분석
- 시계열 데이터 기반 점수화 및 예측 실험

## 📂 Directory Structure
```text
26_winter_proj/
├── data/
│   ├── KFG/          # Fear & Greed 관련 지표 데이터 (gitignore, 로컬 전용)
│   └── NAVER/        # 네이버 댓글/텍스트 데이터 (gitignore, 로컬 전용)
├─ Naver/          # 네이버 댓글 분석
├─ Oscillator/     # 오실레이터 분석
├─ documents/      # 참고 논문 및 자료
├─ README.md
└─ .gitignore
```

## 🧪 Environment
- Python 3.x
- pandas, numpy, matplotlib, scikit-learn, pykrx

## ⚠️ Notes
- `data/` 폴더의 csv 파일은 GitHub에 업로드되지 않습니다.
- 데이터는 별도 경로에서 관리됩니다.