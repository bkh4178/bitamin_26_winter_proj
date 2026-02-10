#%%
import pandas as pd
import re
from collections import Counter
from konlpy.tag import Okt
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False  # 마이너스 깨짐 방지
#%%

# 데이터 로드
df = pd.read_csv("/Users/user/Desktop/bitamin/26_winter_proj/data/NAVER/comments/comments_2024_증시금리.csv")

texts = df["text_raw"].dropna().astype(str)

# 전처리 함수
def clean_text(text):
    text = re.sub(r"http\S+", "", text)          # URL 제거
    text = re.sub(r"[^가-힣\s]", " ", text)       # 한글만
    return text

texts = texts.apply(clean_text)
#%%
okt = Okt()

# 불용어 (최소 예시, 반드시 커스터마이즈)
stopwords = set([
    # 지시/시간
    "이제","지금","오늘","내일","요즘","이번",

    # 존재/동작 동사
    "있다","있는","있음","한다","하는","된다","되는","됨",

    # 조사/연결
    "그리고","그래서","하지만","그런데","때문","때문에",

    # 지시어
    "이거","그거","저거","이게","그게","저게",

    # 일반 추임새
    "진짜","그냥","솔직히","사실","정말",

    # 분석 의미 없는 일반어
    "나라","한국","대한민국",

    # 뉴스 댓글 상투어
    "기자","기사","뉴스","댓글",
    "때문"
])

stop_nouns = set([
    "것","수","때","사람","생각","정도",
    "지금","이제","오늘","이번",
    "기자","기사","뉴스", "때문", '하고','계속', '없다',

])

nouns = []
for t in texts:
    t = clean_text(t)
    for n in okt.nouns(t):
        if len(n) >= 2 and n not in stop_nouns:
            nouns.append(n)

counter = Counter(nouns)
counter.most_common(20)
#%%
# 워드클라우드
wc = WordCloud(
    font_path="/System/Library/Fonts/AppleSDGothicNeo.ttc",
    background_color="white",
    width=800,
    height=600
)

wc.generate_from_frequencies(counter)

plt.figure(figsize=(10, 8))
plt.imshow(wc)
plt.axis("off")
plt.show()

#%%
# 상위 단어 시각화

top_n = 20
top_words = counter.most_common(top_n)

labels = [w for w, _ in top_words]
counts = [c for _, c in top_words]

plt.figure(figsize=(10, 6))
plt.barh(labels[::-1], counts[::-1])  # 가로 막대 (가독성 좋음)
plt.xlabel("빈도수")
plt.title("증시·금리 댓글 상위 키워드 (Top 20)")
plt.tight_layout()
plt.show()

#%%
# 좋아요 수 반영한 가중치 단어 빈도 분석
weighted_words = []

for _, row in df.iterrows():
    text = str(row["text_raw"])
    like = row.get("like_count", 0)
    weight = like + 1  # 0 방지

    for w in clean_text(text).split():
        if len(w) >= 2 and w not in stopwords:
            weighted_words.extend([w] * weight)

counter_like = Counter(weighted_words)
top_words_like = counter_like.most_common(20)

labels = [w for w, _ in top_words_like]
counts = [c for _, c in top_words_like]

plt.figure(figsize=(10, 6))
plt.barh(labels[::-1], counts[::-1])
plt.xlabel("가중 빈도 (좋아요 반영)")
plt.title("공감 기반 상위 키워드 (Top 20)")
plt.tight_layout()
plt.show()

#%%
# 결측치 확인
print('전체 댓글 수:', len(df))
print("결측치 개수 : ", df['text_raw'].isnull().sum())\

#%%
# 결측치 제거 후 일별 댓글 수 시각화
df_clean = df.dropna(subset=['text_raw'])
df_clean['comment_date'] = pd.to_datetime(df_clean['comment_at']).dt.date
daily_counts = df_clean.groupby('comment_date').size() 
daily_counts.plot(kind='bar', figsize=(12, 6), title='일별 댓글 수')
plt.xlabel('날짜')
plt.ylabel('댓글 수')
plt.tight_layout()
plt.show()  

#%%
daily_counts.describe()