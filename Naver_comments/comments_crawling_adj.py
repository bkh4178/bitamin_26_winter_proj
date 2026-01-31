#%%
import requests
import json
import pandas as pd
import time
import re
import os

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
# oid / aid 추출
# --------------------------------------------------
def parse_oid_aid(article_url):
    m = re.search(r"/article/(\d+)/(\d+)", article_url)
    if not m:
        return None, None
    return m.group(1), m.group(2)

def to_legacy_url(article_url):
    oid, aid = parse_oid_aid(article_url)
    if oid is None:
        return None
    return f"https://news.naver.com/main/read.nhn?oid={oid}&aid={aid}"

# --------------------------------------------------
# JSONP 안전 파서
# --------------------------------------------------
def safe_jsonp_load(text):
    if "(" not in text or ")" not in text:
        return None
    try:
        return json.loads(text[text.find("(")+1 : text.rfind(")")])
    except:
        return None

# --------------------------------------------------
# 댓글 전체 수집 (중복 ID 감지 종료)
# --------------------------------------------------
def collect_comments(article_url):
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
        f"&pageSize={PAGE_SIZE}"
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

        time.sleep(PAGE_SLEEP)

    return all_comments

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

        comments = collect_comments(article_url)
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