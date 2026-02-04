#%%
import re
from datetime import date, timedelta
import requests
import pandas as pd
from bs4 import BeautifulSoup
from urllib.parse import quote
import time
import os
from datetime import timedelta
import json

HEADERS_BASE = {
    "User-Agent": "Mozilla/5.0"
}

def extract_oid_aid_key(url: str):
    """네이버 뉴스 URL에서 oid와 aid를 추출해 고유 기사 key 생성"""
    m = re.search(r"/article/(\d+)/(\d+)", url)
    if not m:
        return None
    return f"{m.group(1)}_{m.group(2)}"


def is_financial_title(title: str, fin_keywords) -> bool:
    """기사 제목에 금융 관련 키워드가 포함되어 있는지 여부 판단"""
    return any(k in title for k in fin_keywords)


def day_ranges(year: int):
    """해당 연도의 모든 날짜를 하루 단위 리스트로 생성 (미래 날짜 제외)"""
    today = date.today()
    end = min(date(year, 12, 31), today)
    d = date(year, 1, 1)
    days = []
    while d <= end:
        days.append(d)
        d += timedelta(days=1)
    return days


def collect_links_day(keyword: str, day: date, headers, sleep_sec:float, fin_keywords=None):
    """특정 날짜와 키워드에 대해 네이버 뉴스 기사 링크 목록 수집"""
    q = quote(keyword)
    ds = day.strftime("%Y.%m.%d")

    url = (
        f"https://m.search.naver.com/search.naver"
        f"?where=m_news&query={q}&pd=3&ds={ds}&de={ds}"
    )

    res = requests.get(url, headers=headers, timeout=10)
    soup = BeautifulSoup(res.text, "html.parser")

    rows = []
    for a in soup.select("a[href*='n.news.naver.com/article']"):
        title = a.get("title") or a.text.strip()
        href = a.get("href", "")
        flag = is_financial_title(title, fin_keywords)

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

    time.sleep(sleep_sec)
    return rows

def parse_oid_aid(article_url):
    """기사 URL에서 oid, aid를 분리 추출"""
    m = re.search(r"/article/(\d+)/(\d+)", article_url)
    if not m:
        return None, None
    return m.group(1), m.group(2)


def to_legacy_url(article_url):
    """댓글 API 호출을 위한 네이버 뉴스 레거시 URL 생성"""
    oid, aid = parse_oid_aid(article_url)
    if oid is None:
        return None
    return f"https://news.naver.com/main/read.nhn?oid={oid}&aid={aid}"

def safe_jsonp_load(text):
    """JSONP 형태의 문자열을 안전하게 JSON으로 파싱"""
    if "(" not in text or ")" not in text:
        return None
    try:
        return json.loads(text[text.find("(")+1 : text.rfind(")")])
    except:
        return None
    

def collect_comments(article_url, page_size, page_sleep):
    """커서 기반 페이지네이션을 이용해 기사 댓글 전체 수집"""
    legacy_url = to_legacy_url(article_url)
    if legacy_url is None:
        return []

    oid, aid = parse_oid_aid(article_url)
    object_id = f"news{oid},{aid}"

    headers = {
        **HEADERS_BASE,
        "Referer": legacy_url
    }

    base = (
        "https://apis.naver.com/commentBox/cbox/web_naver_list_jsonp.json"
        "?ticket=news"
        "&templateId=view_politics"
        "&pool=cbox5"
        "&lang=ko"
        "&country=KR"
        f"&objectId={object_id.replace(',', '%2C')}"
        "&sort=favorite"
        "&initialize=true"
        f"&pageSize={page_size}"
    )

    all_comments = []
    seen_ids = set()

    next_cursor = None
    seen_cursors = set()

    while True:
        if next_cursor is None:
            url = base  # 첫 페이지(초기 로딩)
        else:
            # 🔥 다음 페이지는 page 번호가 아니라 cursor로 넘김
            url = (
                base
                + "&pageType=more"
                + f"&moreParam.next={next_cursor}"
                + "&initialize=false"
            )

        r = requests.get(url, headers=headers, timeout=10)
        data = safe_jsonp_load(r.text)
        if not data:
            break

        result = data.get("result", {})
        comment_list = result.get("commentList", [])
        if not comment_list:
            break

        new_count = 0
        for c in comment_list:
            cid = c.get("commentNo")
            if cid is None or cid in seen_ids:
                continue
            seen_ids.add(cid)
            new_count += 1

            all_comments.append({
                "comment_id": cid,
                "article_url": article_url,
                "contents": c.get("contents", "").replace("\n", " ").strip(),
                "sympathy": c.get("sympathyCount", 0),
                "antipathy": c.get("antipathyCount", 0),
                "reg_time": c.get("regTime")
            })

        # 새 댓글이 더 이상 안 나오면 종료
        if new_count == 0:
            break

        mp = result.get("morePage", {})
        next_cursor_new = mp.get("next")

        # next 커서가 없거나, 반복되면 종료(무한루프 방지)
        if not next_cursor_new or next_cursor_new in seen_cursors:
            break

        seen_cursors.add(next_cursor_new)
        next_cursor = next_cursor_new

        time.sleep(page_sleep)

    return all_comments