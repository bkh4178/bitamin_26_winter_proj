# collect_naver_2025_top5.py
# Python 3.9 compatible

import argparse
import csv
import json
import random
import re
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, Iterable, List, Optional, Tuple
from urllib.parse import urlencode, urlparse, parse_qs, urlunparse

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm


KEYWORDS = ["위기", "침체", "불황", "부도", "파산", "금융위기", "쇼크"]

SECTION_LIST_ENDPOINT = "https://news.naver.com/section/template/SECTION_ARTICLE_LIST_FOR_LATEST"
COMMENT_COUNT_ENDPOINT = "https://news.naver.com/section/template/NEWS_COMMENT_COUNT_LIST"
BREAKING_BASE = "https://news.naver.com/breakingnews/section/101"  # /{sid2}?date=YYYYMMDD


@dataclass
class Article:
    date: str
    sid2: int
    url: str
    oid: str
    aid: str
    title: str
    keyword: str
    object_id: str  # news{oid},{aid}


def make_session() -> requests.Session:
    s = requests.Session()
    s.headers.update({
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept-Language": "ko-KR,ko;q=0.9,en;q=0.7",
    })
    return s


def safe_sleep(base: float):
    time.sleep(base + random.uniform(0.0, base * 0.5))


def daterange_yyyymmdd(start_yyyymmdd: str, end_yyyymmdd: str) -> Iterable[str]:
    start = datetime.strptime(start_yyyymmdd, "%Y%m%d")
    end = datetime.strptime(end_yyyymmdd, "%Y%m%d")
    cur = start
    while cur <= end:
        yield cur.strftime("%Y%m%d")
        cur += timedelta(days=1)


def first_matched_keyword(title: str) -> str:
    for k in KEYWORDS:
        if k in title:
            return k
    return ""


def extract_oid_aid(url: str) -> Optional[Tuple[str, str]]:
    m = re.search(r"/article/(\d{3})/(\d+)", url)
    if m:
        return m.group(1), m.group(2)
    m1 = re.search(r"[?&]oid=(\d{3})", url)
    m2 = re.search(r"[?&]aid=(\d+)", url)
    if m1 and m2:
        return m1.group(1), m2.group(1)
    return None


def strip_jsonp(text: str) -> str:
    text = text.strip()
    if text.startswith("{") and text.endswith("}"):
        return text
    # callback({...}); 형태
    m = re.search(r"\((\{.*\})\)\s*;?\s*$", text, flags=re.DOTALL)
    return m.group(1) if m else text


def flatten_strings(obj):
    if isinstance(obj, dict):
        for v in obj.values():
            yield from flatten_strings(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from flatten_strings(v)
    elif isinstance(obj, str):
        yield obj


def parse_articles_from_html(html: str) -> List[Tuple[str, str]]:
    """Return list of (url, title) extracted from HTML snippet."""
    soup = BeautifulSoup(html, "html.parser")
    out = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if "/article/" not in href:
            continue
        title = a.get_text(strip=True)
        if not title:
            continue
        if href.startswith("/"):
            url = "https://news.naver.com" + href
        else:
            url = href
        out.append((url, title))
    # dedupe
    seen = set()
    uniq = []
    for u, t in out:
        if u in seen:
            continue
        seen.add(u)
        uniq.append((u, t))
    return uniq


def get_initial_next_from_html(session: requests.Session, date: str, sid2: int, sleep_sec: float) -> Optional[str]:
    """Extract initial next token from breakingnews page HTML for that date/section."""
    url = f"{BREAKING_BASE}/{sid2}"
    r = session.get(url, params={"date": date}, timeout=20)
    if r.status_code != 200:
        return None
    html = r.text
    m = re.search(r"SECTION_ARTICLE_LIST_FOR_LATEST[^\"']*next=(\d{12,})", html)
    safe_sleep(sleep_sec)
    return m.group(1) if m else None


def fetch_section_articles_for_day(session: requests.Session, date: str, sid2: int,
                                  sleep_sec: float, max_pages: int = 80) -> List[Tuple[str, str]]:
    """Collect (url, title) for a given day/section using SECTION_ARTICLE_LIST_FOR_LATEST paging."""
    all_items: List[Tuple[str, str]] = []
    seen = set()

    next_token = get_initial_next_from_html(session, date, sid2, sleep_sec=sleep_sec)

    for page_no in range(1, max_pages + 1):
        params = {
            "sid": "101",
            "sid2": str(sid2),
            "cluid": "",
            "pageNo": str(page_no),
            "date": date,
        }
        if next_token:
            params["next"] = next_token

        r = session.get(SECTION_LIST_ENDPOINT, params=params, timeout=20)
        if r.status_code != 200:
            break

        try:
            data = r.json()
        except Exception:
            try:
                data = json.loads(r.text)
            except Exception:
                break

        html_snips = [s for s in flatten_strings(data) if "/article/" in s]
        page_new = 0
        for snip in html_snips:
            for url, title in parse_articles_from_html(snip):
                if url in seen:
                    continue
                seen.add(url)
                all_items.append((url, title))
                page_new += 1

        # update next token (best-effort)
        found_next = None
        if isinstance(data, dict) and isinstance(data.get("next"), str):
            found_next = data["next"]
        if not found_next:
            m = re.search(rf"{date}\d{{6,}}", r.text)
            if m:
                found_next = m.group(0)
        if found_next:
            next_token = found_next

        if page_new == 0:
            break

        safe_sleep(sleep_sec)

    return all_items


def chunked(lst: List[str], n: int) -> Iterable[List[str]]:
    for i in range(0, len(lst), n):
        yield lst[i:i+n]


def fetch_comment_counts(session: requests.Session, object_ids: List[str], sleep_sec: float,
                        chunk_size: int = 80) -> Dict[str, int]:
    """objectId -> total comment count"""
    counts: Dict[str, int] = {}
    for part in chunked(object_ids, chunk_size):
        params = {"ticket": "news", "objectIds": ";".join(part)}
        r = session.get(COMMENT_COUNT_ENDPOINT, params=params, timeout=20)
        if r.status_code != 200:
            safe_sleep(sleep_sec)
            continue
        try:
            data = r.json()
        except Exception:
            try:
                data = json.loads(r.text)
            except Exception:
                safe_sleep(sleep_sec)
                continue

        def walk(obj):
            if isinstance(obj, dict):
                if "objectId" in obj:
                    oid = obj.get("objectId")
                    for k in ("commentCount", "count", "totalCount"):
                        if k in obj:
                            try:
                                counts[str(oid)] = int(obj[k])
                            except Exception:
                                pass
                            break
                for v in obj.values():
                    walk(v)
            elif isinstance(obj, list):
                for v in obj:
                    walk(v)

        walk(data)
        safe_sleep(sleep_sec)

    return counts


def build_comment_list_url(template_url: str, object_id: str, sort: str, page_size: int) -> str:
    """
    Use captured web_naver_list_jsonp URL as template.
    Overwrite objectId/sort/pageSize/page/initialize only.
    """
    u = urlparse(template_url)
    q = parse_qs(u.query)

    q["objectId"] = [object_id]
    q["sort"] = [sort]
    q["pageSize"] = [str(page_size)]
    q["page"] = ["1"]
    q["initialize"] = ["true"]

    # keep existing callback/_cv/pool/templateId/etc as-is
    new_query = urlencode({k: v[0] for k, v in q.items()}, doseq=False)
    return urlunparse((u.scheme, u.netloc, u.path, u.params, new_query, u.fragment))


def fetch_comments(session: requests.Session, article_url: str, template_url: str,
                   object_id: str, sort: str, page_size: int, sleep_sec: float) -> List[Dict]:
    url = build_comment_list_url(template_url, object_id, sort, page_size)
    headers = {"Referer": article_url}
    r = session.get(url, headers=headers, timeout=20)
    if r.status_code != 200:
        return []

    body = strip_jsonp(r.text)
    try:
        data = json.loads(body)
    except Exception:
        return []

    comments = []

    def walk(obj):
        if isinstance(obj, dict):
            if ("commentNo" in obj or "commentId" in obj) and ("contents" in obj or "content" in obj or "text" in obj):
                cid = obj.get("commentNo", obj.get("commentId"))
                text = obj.get("contents", obj.get("content", obj.get("text", "")))
                like_cnt = obj.get("sympathyCount", obj.get("likeCount", obj.get("goodCount", 0)))
                dislike_cnt = obj.get("antipathyCount", obj.get("dislikeCount", obj.get("badCount", 0)))
                created = obj.get("regTime", obj.get("createdAt", obj.get("time", "")))

                comments.append({
                    "comment_id": str(cid),
                    "comment_at": str(created),
                    "text_raw": str(text),
                    "like_count": int(like_cnt) if str(like_cnt).isdigit() else 0,
                    "dislike_count": int(dislike_cnt) if str(dislike_cnt).isdigit() else 0,
                    "sort": sort,
                })
            for v in obj.values():
                walk(v)
        elif isinstance(obj, list):
            for v in obj:
                walk(v)

    walk(data)
    safe_sleep(sleep_sec)
    return comments


def ensure_csv(path: str, header: List[str]):
    try:
        with open(path, "r", encoding="utf-8-sig", newline="") as _:
            return
    except FileNotFoundError:
        with open(path, "w", encoding="utf-8-sig", newline="") as f:
            csv.writer(f).writerow(header)


def append_rows(path: str, rows: List[List]):
    if not rows:
        return
    with open(path, "a", encoding="utf-8-sig", newline="") as f:
        csv.writer(f).writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="20250101")
    ap.add_argument("--end", default="20251231")
    ap.add_argument("--topk", type=int, default=5)      # Top5
    ap.add_argument("--per_sort", type=int, default=30) # 공감 30, 최신 30
    ap.add_argument("--sleep", type=float, default=0.45)
    ap.add_argument("--comment_template_url", required=True)
    ap.add_argument("--out_news", default="news_2025_top5.csv")
    ap.add_argument("--out_comments", default="comments_2025_top5.csv")
    ap.add_argument("--test_days", type=int, default=0, help="예: 3이면 3일만 테스트")
    args = ap.parse_args()

    session = make_session()

    ensure_csv(args.out_news, [
        "news_date", "news_id", "section", "keyword", "title",
        "comment_total", "rank", "url"
    ])
    ensure_csv(args.out_comments, [
        "news_id", "comment_id", "comment_at", "sort",
        "text_raw", "like_count", "dislike_count"
    ])

    dates = list(daterange_yyyymmdd(args.start, args.end))
    if args.test_days > 0:
        dates = dates[:args.test_days]

    for date in tqdm(dates, desc="Dates"):
        # 1) 날짜별 기사 수집 (259 금융 + 258 증권)
        candidates: List[Article] = []
        for sid2 in (259, 258):
            items = fetch_section_articles_for_day(session, date, sid2, sleep_sec=args.sleep)
            for url, title in items:
                kw = first_matched_keyword(title)
                if not kw:
                    continue
                oa = extract_oid_aid(url)
                if not oa:
                    continue
                oid, aid = oa
                obj_id = f"news{oid},{aid}"
                candidates.append(Article(
                    date=date, sid2=sid2, url=url, oid=oid, aid=aid,
                    title=title, keyword=kw, object_id=obj_id
                ))

        if not candidates:
            continue

        # 2) 댓글 총개수 → Top5
        obj_ids = list({a.object_id for a in candidates})
        counts = fetch_comment_counts(session, obj_ids, sleep_sec=args.sleep)

        scored = []
        for a in candidates:
            scored.append((counts.get(a.object_id, 0), a))
        scored.sort(key=lambda x: x[0], reverse=True)

        # unique by news_id
        top = []
        seen_news = set()
        for c, a in scored:
            news_id = f"{a.oid}_{a.aid}"
            if news_id in seen_news:
                continue
            seen_news.add(news_id)
            top.append((c, a))
            if len(top) >= args.topk:
                break

        if not top:
            continue

        # 3) news 저장
        news_rows = []
        for rank, (c, a) in enumerate(top, start=1):
            news_id = f"{a.oid}_{a.aid}"
            news_rows.append([date, news_id, a.sid2, a.keyword, a.title, c, rank, a.url])
        append_rows(args.out_news, news_rows)

        # 4) 댓글 저장 (공감30 + 최신30)
        comment_rows = []
        for _, a in top:
            news_id = f"{a.oid}_{a.aid}"

            fav = fetch_comments(session, a.url, args.comment_template_url, a.object_id,
                                 sort="favorite", page_size=args.per_sort, sleep_sec=args.sleep)
            new = fetch_comments(session, a.url, args.comment_template_url, a.object_id,
                                 sort="new", page_size=args.per_sort, sleep_sec=args.sleep)

            merged = {}
            for item in fav + new:
                merged[item["comment_id"]] = item  # comment_id로 중복 제거

            for item in merged.values():
                comment_rows.append([
                    news_id,
                    item["comment_id"],
                    item["comment_at"],
                    item["sort"],
                    item["text_raw"],
                    item["like_count"],
                    item["dislike_count"],
                ])

        append_rows(args.out_comments, comment_rows)
        safe_sleep(args.sleep)


if __name__ == "__main__":
    main()
