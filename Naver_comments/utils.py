#%%
import re
from datetime import date, timedelta

# 금융 맥락 키워드
FIN_KEYWORDS = [
    "증시","주식","코스피","코스닥","시장","지수",
    "투자","매도","매수","외국인","기관","개인"
]

def extract_oid_aid_key(url: str):
    m = re.search(r"/article/(\d+)/(\d+)", url)
    if not m:
        return None
    return f"{m.group(1)}_{m.group(2)}"

def is_financial_title(title: str) -> bool:
    return any(k in title for k in FIN_KEYWORDS)

def week_ranges(year: int):
    """
    1년을 주(7일) 단위로 분할
    """
    ranges = []
    cur = date(year, 1, 1)
    end = date(year, 12, 31)

    while cur <= end:
        r_end = min(cur + timedelta(days=6), end)
        ranges.append((cur, r_end))
        cur = r_end + timedelta(days=1)

    return ranges