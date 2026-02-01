#%%
import requests
import json
import pandas as pd
import time
import re
import os

from utils import to_legacy_url, parse_oid_aid, safe_jsonp_load, collect_comments

# --------------------------------------------------
# 설정
# --------------------------------------------------
HEADERS_BASE = {
    "User-Agent": "Mozilla/5.0"
}

BASE_LIST_URL = (
    "https://apis.naver.com/commentBox/cbox/web_naver_list_jsonp.json"
    "?ticket=news"
    "&templateId=view_politics"
    "&pool=cbox5"
    "&lang=ko"
    "&country=KR"
    "&objectId={object_id}"
    "&pageSize={page_size}"
    "&page={page}"
    "&sort=favorite"
    "&initialize=true"
)

ARTICLE_CSV = "../data/NAVER/article/articles_2025_financial.csv"
OUTPUT_CSV = "../data/NAVER/comments/comments_2025_adj.csv"

PAGE_SIZE = 100      # 네이버 서버가 사실상 허용하는 최대
MAX_PAGES = 100      # 안전 장치
ARTICLE_SLEEP = 0.5
PAGE_SLEEP = 0.2

# --------------------------------------------------
# 메인 실행 (기사 단위 append 저장)
# --------------------------------------------------
def main():
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

    df = pd.read_csv(ARTICLE_CSV)
    first_write = not os.path.exists(OUTPUT_CSV)

    print("총 기사 수:", len(df))

    for i, row in df.iterrows():
        article_url = row["url"]
        print(f"[{i+1}/{len(df)}] 댓글 수집:", article_url)

        comments = collect_comments(article_url, PAGE_SIZE, PAGE_SLEEP)
        print("  수집 댓글 수:", len(comments))

        if comments:
            out_df = pd.DataFrame(comments)
            out_df.to_csv(
                OUTPUT_CSV,
                mode="a",
                header=first_write,
                index=False,
                encoding="utf-8-sig"
            )
            first_write = False

        time.sleep(ARTICLE_SLEEP)

    print("\n✅ 댓글 수집 완료")
    print("저장 파일:", OUTPUT_CSV)

if __name__ == "__main__":
    main()
#%%
comments = collect_comments("https://n.news.naver.com/article/011/0004445148")
print("수집 댓글 수:", len(comments))