#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from collections import Counter

# seaborn 먼저
sns.set_theme(style="whitegrid")

# inline 환경에서는 이름으로 지정
plt.rcParams['font.family'] = 'Apple SD Gothic Neo'
plt.rcParams['axes.unicode_minus'] = False

print("=" * 80)
print("정치 키워드 기반 댓글 분류 - 시각화")
print("=" * 80)


# ============================
# 1. 데이터 로딩 (여기만 네 파일로 수정)
# ============================
df = pd.read_csv("/Users/user/Desktop/bitamin/26_winter_proj/data/NAVER/comments/comments_2023_증시금리.csv")

df = df[df['text_raw'].notna() & (df['text_raw'].str.strip() != '')].copy()

# ============================
# 2. 정치 키워드 정의
# ============================
politics_patterns = [
    r'윤석열|석열|이재명|한동훈',
    r'민주당|국민의힘|민주|국힘',
    r'탄핵|계엄|내란|구속|체포',
    r'의원|국회|여당|야당',
]

def contains_political_keywords(text):
    text_lower = str(text).lower()
    for pattern in politics_patterns:
        if re.search(pattern, text_lower):
            return True
    return False

def get_matched_keywords(text):
    text_lower = str(text).lower()
    matched = []
    for pattern in politics_patterns:
        keywords = pattern.split('|')
        for keyword in keywords:
            if keyword in text_lower:
                matched.append(keyword)
    return list(set(matched))

df['is_political'] = df['text_raw'].apply(contains_political_keywords)
df['matched_keywords'] = df['text_raw'].apply(get_matched_keywords)
df['keyword_count'] = df['matched_keywords'].apply(len)

political_comments = df[df['is_political']].copy()
non_political_comments = df[~df['is_political']].copy()

# ============================
# 3. 기본 분류 결과 (파이 + 막대)
# ============================
sizes = [len(political_comments), len(non_political_comments)]
labels = ['정치 관련', '비정치']
colors = ['#e74c3c', '#3498db']
explode = (0.05, 0)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

axes[0].pie(
    sizes, labels=labels, autopct='%1.1f%%',
    colors=colors, explode=explode,
    startangle=90, textprops={'fontsize': 14, 'weight': 'bold'}
)
axes[0].set_title('댓글 분류 결과')

axes[1].bar(labels, sizes, color=colors, alpha=0.8)
axes[1].set_ylabel('댓글 수')
axes[1].set_title('댓글 수 비교')

plt.tight_layout()
plt.show()

# ============================
# 4. 키워드별 매칭 빈도
# ============================
all_keywords = []
for keywords in political_comments['matched_keywords']:
    all_keywords.extend(keywords)

keyword_freq = Counter(all_keywords)
top_20 = keyword_freq.most_common(20)

if len(top_20) > 0:
    keywords, counts = zip(*top_20)

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.barh(range(len(keywords)), counts, color=plt.cm.Reds(np.linspace(0.4, 0.9, len(keywords))))
    ax.set_yticks(range(len(keywords)))
    ax.set_yticklabels(keywords)
    ax.set_xlabel("매칭 횟수")
    ax.set_title("정치 키워드 빈도 TOP 20")
    ax.invert_yaxis()
    plt.tight_layout()
    plt.show()

# ============================
# 5. 키워드 개수 분포
# ============================
keyword_count_dist = political_comments['keyword_count'].value_counts().sort_index()

fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(keyword_count_dist.index, keyword_count_dist.values, color='#e74c3c')
ax.set_xlabel("댓글당 매칭된 키워드 개수")
ax.set_ylabel("댓글 수")
ax.set_title("정치 댓글의 키워드 개수 분포")
plt.tight_layout()
plt.show()

# ============================
# 6. 일별 정치 댓글 비율
# ============================
df['comment_date'] = pd.to_datetime(df['comment_at'])
df['date'] = df['comment_date'].dt.date

daily_stats = df.groupby(['date', 'is_political']).size().unstack(fill_value=0)

daily_stats['total'] = daily_stats.sum(axis=1)
daily_stats['political_ratio'] = (daily_stats.get(True, 0) / daily_stats['total'] * 100)

fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(daily_stats.index, daily_stats['political_ratio'], marker='o')
ax.set_ylabel("정치 댓글 비율 (%)")
ax.set_title("일별 정치 댓글 비율 추이")
ax.tick_params(axis='x', rotation=45)
plt.tight_layout()
plt.show()

# ============================
# 7. 좋아요 분포 비교
# ============================
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

axes[0].boxplot(
    [political_comments['like_count'], non_political_comments['like_count']],
    labels=['정치', '비정치'],
    patch_artist=True
)
axes[0].set_title("좋아요 분포 비교")

axes[1].hist(political_comments['like_count'], bins=30, alpha=0.6, label='정치')
axes[1].hist(non_political_comments['like_count'], bins=30, alpha=0.6, label='비정치')
axes[1].legend()
axes[1].set_title("좋아요 히스토그램")

plt.tight_layout()
plt.show()