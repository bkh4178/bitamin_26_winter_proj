#%%
import requests
import json
import pandas as pd
import time
import re

# --------------------------------------------------
# 0. 공통 설정
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

# --------------------------------------------------
# 1. oid / aid 추출
# --------------------------------------------------
def parse_oid_aid(article_url):
    """
    https://n.news.naver.com/article/011/0004445148
    """
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
# 2. JSONP 파서
# --------------------------------------------------
def safe_jsonp_load(text):
    if "(" not in text or ")" not in text:
        return None
    try:
        return json.loads(text[text.find("(")+1 : text.rfind(")")])
    except:
        return None

# --------------------------------------------------
# 3. 댓글 수집 (중복 감지 종료)
# --------------------------------------------------
def collect_comments(article_url, page_size=100, max_pages=50):
    legacy_url = to_legacy_url(article_url)
    if legacy_url is None:
        return []

    oid, aid = parse_oid_aid(article_url)
    object_id = f"news{oid},{aid}"

    headers = {
        **HEADERS_BASE,
        "Referer": legacy_url
    }

    all_comments = []
    seen_ids = set()
    page = 1

    while page <= max_pages:
        url = BASE_LIST_URL.format(
            object_id=object_id.replace(",", "%2C"),
            page_size=page_size,
            page=page
        )

        r = requests.get(url, headers=headers, timeout=10)
        data = safe_jsonp_load(r.text)
        if not data:
            break

        comment_list = data.get("result", {}).get("commentList", [])
        if not comment_list:
            break

        new_count = 0
        for c in comment_list:
            cid = c.get("commentNo")
            if cid in seen_ids:
                continue
            seen_ids.add(cid)
            new_count += 1

            all_comments.append({
                "comment_id": cid,
                "comment": c.get("contents", "").replace("\n", " ").strip(),
                "sympathy": c.get("sympathyCount", 0),
                "antipathy": c.get("antipathyCount", 0),
                "reg_time": c.get("regTime")
            })

        # 🔴 종료 조건
        if new_count == 0:
            break

        page += 1
        time.sleep(0.2)

    return all_comments

# --------------------------------------------------
# 4. 메인 실행부 (CSV에 있는 기사 전부)
# --------------------------------------------------
df = pd.read_csv("output/articles_test.csv")
results = []

for i, row in df.iterrows():
    article_url = row["url"]
    keyword = row.get("keyword", "폭락")

    print(f"[{i+1}/{len(df)}] 처리 중:", article_url)

    comments = collect_comments(article_url)
    print("  수집 댓글 수:", len(comments))

    for c in comments:
        c["article_url"] = article_url
        c["keyword"] = keyword
        results.append(c)

    # 기사 단위 휴식 (차단 방지)
    time.sleep(0.5)

# --------------------------------------------------
# 5. 저장
# --------------------------------------------------
out_df = pd.DataFrame(results)
out_df.to_csv(
    "output/comments_final.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\n=== 전체 완료 ===")
print("총 댓글 수:", len(out_df))