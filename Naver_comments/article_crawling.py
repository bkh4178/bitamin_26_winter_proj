#%%
# article_crawling.py
import requests
import pandas as pd
from bs4 import BeautifulSoup
from urllib.parse import quote
import time
import os
from datetime import date, timedelta

from utils import extract_oid_aid_key, is_financial_title

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

# -----------------------------
# 날짜 리스트 (미래 날짜 제외)
# -----------------------------
def day_ranges(year: int):
    today = date.today()
    end = min(date(year, 12, 31), today)
    d = date(year, 1, 1)
    days = []
    while d <= end:
        days.append(d)
        d += timedelta(days=1)
    return days

# -----------------------------
# 일 × 키워드 기사 수집
# -----------------------------
def collect_links_day(keyword: str, day: date):
    q = quote(keyword)
    ds = day.strftime("%Y.%m.%d")

    url = (
        f"https://m.search.naver.com/search.naver"
        f"?where=m_news&query={q}&pd=3&ds={ds}&de={ds}"
    )

    res = requests.get(url, headers=HEADERS, timeout=10)
    soup = BeautifulSoup(res.text, "html.parser")

    rows = []
    for a in soup.select("a[href*='n.news.naver.com/article']"):
        title = a.get("title") or a.text.strip()
        href = a.get("href", "")
        flag = is_financial_title(title)

        key = extract_oid_aid_key(href)
        if not key:
            continue

        rows.append({
        "key": key,
        "keyword": keyword,
        "title": title,
        "url": href,
        "date": ds,
        "is_financial": int(flag)
        })

        key = extract_oid_aid_key(href)
        if not key:
            continue
        rows.append({
            "key": key,
            "keyword": keyword,
            "title": title,
            "url": href,
            "date": ds
        })

    time.sleep(SLEEP_SEC)
    return rows

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
            rows = collect_links_day(kw, d)
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
import pandas as pd

df = pd.read_csv("../data/NAVER/article/articles_2025_financial.csv")

print("총 기사 수:", len(df))
print("is_financial=1 비율:", df["is_financial"].mean())

daily = df.groupby("date").size()
print("0건인 날짜 수:", (daily==0).sum())  # 원래 거의 0이 뜰 거라 의미 없음
print("1개 미만(=0) 날짜 수:", (daily<1).sum())
print("하루 평균:", daily.mean())
print("하루 중앙값:", daily.median())