#%%
import requests
import json
import re

ARTICLE_URL = "https://n.news.naver.com/article/011/0004445148"

HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Referer": ARTICLE_URL
}

BASE_URL = (
    "https://apis.naver.com/commentBox/cbox/web_naver_list_jsonp.json"
    "?ticket=news"
    "&templateId=view_politics"
    "&pool=cbox5"
    "&lang=ko"
    "&country=KR"
    "&objectId={object_id}"
    "&pageSize={page_size}"
    "&page=1"
    "&sort=favorite"
    "&initialize=true"
)

def parse_oid_aid(url):
    m = re.search(r"/article/(\d+)/(\d+)", url)
    return m.group(1), m.group(2)

def safe_jsonp(text):
    return json.loads(text[text.find("(")+1 : text.rfind(")")])

def test_page_size(page_size):
    oid, aid = parse_oid_aid(ARTICLE_URL)
    object_id = f"news{oid}%2C{aid}"

    url = BASE_URL.format(
        object_id=object_id,
        page_size=page_size
    )

    r = requests.get(url, headers=HEADERS, timeout=10)
    data = safe_jsonp(r.text)

    comments = data.get("result", {}).get("commentList", [])
    print(f"pageSize={page_size:>3} → 반환 댓글 수: {len(comments)}")

# 테스트
for size in [10, 20, 50, 100, 150, 200, 300]:
    test_page_size(size)

'''
pageSize= 10 → 반환 댓글 수: 10
pageSize= 20 → 반환 댓글 수: 20
pageSize= 50 → 반환 댓글 수: 50
pageSize=100 → 반환 댓글 수: 100
pageSize=150 → 반환 댓글 수: 100
pageSize=200 → 반환 댓글 수: 100
pageSize=300 → 반환 댓글 수: 100
'''
# 결론: pageSize 최대 100까지만 정상 동작