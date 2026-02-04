#%%
import requests
import pandas as pd
from bs4 import BeautifulSoup
from urllib.parse import quote
import time
import os
from datetime import date, timedelta

from utils import extract_oid_aid_key, is_financial_title, day_ranges, collect_links_day

# -----------------------------
# 설정
# -----------------------------
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (iPhone; CPU iPhone OS 16_0 like Mac OS X) "
        "AppleWebKit/605.1.15 (KHTML, like Gecko) "
        "Version/16.0 Mobile/15E148 Safari/604.1"
    )
}

KEYWORDS = ["폭락", "급락", "급등", "반등", "조정", "과열", "버블", "패닉", "랠리"]
YEAR = 2025
SLEEP_SEC = 0.3

OUTPUT_DIR = "../data/NAVER/article"
OUTPUT_PATH = f"{OUTPUT_DIR}/articles_2025_financial.csv"

# 금융 맥락 키워드
FIN_KEYWORDS = [
    "증시","주식","코스피","코스닥","시장","지수",
    "투자","매도","매수","외국인","기관","개인"
]

# -----------------------------
# 메인
# -----------------------------
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # key(oid+aid) 기준으로만 중복 제거
    uniq = {}

    days = day_ranges(YEAR)
    print(f"수집 대상 날짜 수: {len(days)}")

    for d in days:
        print(f"\n📅 {d}")
        for kw in KEYWORDS:
            rows = collect_links_day(kw, d, HEADERS, SLEEP_SEC, fin_keywords=FIN_KEYWORDS)
            for r in rows:
                uniq.setdefault(r["key"], r)

    df = pd.DataFrame(uniq.values()).drop_duplicates(subset=["url"])
    df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

    print("\n✅ 완료")
    print("총 기사 수:", len(df))
    print("저장 위치:", OUTPUT_PATH)

if __name__ == "__main__":
    main()


#%%
# 확인용
import pandas as pd

df = pd.read_csv("../data/NAVER/article/articles_2025_financial.csv")

print("총 기사 수:", len(df))
print("is_financial=1 비율:", df["is_financial"].mean())

daily = df.groupby("date").size()
print("0건인 날짜 수:", (daily==0).sum())  # 원래 거의 0이 뜰 거라 의미 없음
print("1개 미만(=0) 날짜 수:", (daily<1).sum())
print("하루 평균:", daily.mean())
print("하루 중앙값:", daily.median())
print("nan : ", df['is_financial'].isnull().sum())