#%%
import requests
import pandas as pd
from bs4 import BeautifulSoup
from urllib.parse import quote
import time

HEADERS = {
    "User-Agent": "Mozilla/5.0"
}

def collect_article_links(keyword):
    q = quote(keyword)
    url = (
        f"https://m.search.naver.com/search.naver"
        f"?where=m_news&query={q}&pd=3"
        f"&ds=2025.01.01&de=2025.01.31"
    )

    res = requests.get(url, headers=HEADERS)
    soup = BeautifulSoup(res.text, "html.parser")

    links = set()
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if "n.news.naver.com/article" in href:
            links.add(href)

    return list(links)


if __name__ == "__main__":
    keyword = "폭락"
    print("기사 수집 시작")

    links = collect_article_links(keyword)
    print("기사 수:", len(links))

    df = pd.DataFrame({
        "keyword": keyword,
        "url": links
    })

    df.to_csv("output/articles_test.csv", index=False, encoding="utf-8-sig")
    print("기사 URL 저장 완료")