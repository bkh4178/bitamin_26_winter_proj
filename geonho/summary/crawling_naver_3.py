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
from pathlib import Path

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

from utils_crawling_3 import *

KEYWORDS = ['주식', '한국증시', '삼성전자', 'SK하이닉스'] 

SECTION_LIST_ENDPOINT = "https://news.naver.com/section/template/SECTION_ARTICLE_LIST_FOR_LATEST"
COMMENT_COUNT_ENDPOINT = "https://news.naver.com/section/template/NEWS_COMMENT_COUNT_LIST"
BREAKING_BASE = "https://news.naver.com/breakingnews/section/101"  # /{sid2}?date=YYYYMMDD

def run_crawling(
    year: int,
    comment_template_url: str,
    topk: int = 5,
    per_article: int = 30,
    sleep: float = 0.9,
    test_days: int = 0,
    comment_page_size: int = 100,
    max_comment_pages: int = 50,
    strict_pubdate: bool = False,
):

    start = f"{year}0603"
    end = f"{year}1231"

    Path("data/NAVER/article").mkdir(parents=True, exist_ok=True)
    Path("data/NAVER/comments").mkdir(parents=True, exist_ok=True)

    out_news = f"data/NAVER/article/news_{year}_3_top{topk}.csv"
    out_comments = f"data/NAVER/comments/comments_{year}_3_top{topk}.csv"

    session = make_session()

    ensure_csv(out_news, [
        "loop_date", "pub_date", "news_id", "section", "keyword", "title",
        "comment_total_all", "rank_in_section", "url"
    ])
    ensure_csv(out_comments, [
        "news_id", "pub_date", "comment_id", "comment_at",
        "text_raw", "like_count", "dislike_count"
    ])

    dates = list(daterange_yyyymmdd(start, end))
    if test_days > 0:
        dates = dates[:test_days]

    processed_news = set()

    for loop_date in tqdm(dates, desc="Dates"):
        for sid2 in (259, 258):

            raw_items = fetch_section_articles_for_day(
                session, loop_date, sid2, sleep_sec=sleep
            )

            candidates = []
            for url, title in raw_items:
                kw = first_matched_keyword(title)
                if not kw:
                    continue

                oa = extract_oid_aid(url)
                if not oa:
                    continue

                oid, aid = oa
                obj_id = f"news{oid},{aid}"

                candidates.append(Article(
                    list_date=loop_date,
                    sid2=sid2,
                    url=url,
                    oid=oid,
                    aid=aid,
                    title=title,
                    keyword=kw,
                    object_id=obj_id
                ))

            if not candidates:
                continue

            obj_ids = list({a.object_id for a in candidates})
            counts = fetch_comment_counts(session, obj_ids, sleep_sec=sleep)

            scored = [(counts.get(a.object_id, 0), a) for a in candidates]
            scored.sort(key=lambda x: x[0], reverse=True)

            top = []
            seen_in_section = set()

            for c, a in scored:
                news_id = f"{a.oid}_{a.aid}"
                if news_id in seen_in_section:
                    continue
                if news_id in processed_news:
                    continue

                pub_date = get_article_published_yyyymmdd(
                    session, a.url, sleep_sec=sleep
                ) or loop_date
                a.pub_date = pub_date

                if strict_pubdate and pub_date != loop_date:
                    continue

                seen_in_section.add(news_id)
                top.append((c, a))

                if len(top) >= topk:
                    break

            if not top:
                continue

            news_rows = []
            comment_rows = []

            for rank, (c_total, a) in enumerate(top, start=1):

                news_id = f"{a.oid}_{a.aid}"
                pub_date = a.pub_date or loop_date

                if strict_pubdate and pub_date != loop_date:
                    continue

                processed_news.add(news_id)

                news_rows.append([
                    loop_date,
                    pub_date,
                    news_id,
                    a.sid2,
                    a.keyword,
                    a.title,
                    c_total,
                    rank,
                    a.url
                ])

                day_comments = collect_same_day_comments_topliked(
                    session=session,
                    article_url=a.url,
                    template_url=comment_template_url,
                    object_id=a.object_id,
                    target_day=pub_date,
                    want_n=per_article,
                    page_size=comment_page_size,
                    max_pages=max_comment_pages,
                    sleep_sec=sleep,
                )

                for it in day_comments:
                    comment_rows.append([
                        news_id,
                        pub_date,
                        it["comment_id"],
                        it["comment_at"],
                        it["text_raw"],
                        it["like_count"],
                        it["dislike_count"],
                    ])

            append_rows(out_news, news_rows)
            append_rows(out_comments, comment_rows)

            safe_sleep(sleep)