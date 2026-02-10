#%%
import pandas as pd
import numpy as np

df = pd.read_csv("/Users/user/Desktop/bitamin/26_winter_proj/data/NAVER/comments/comments_2024_증시금리.csv")
df.head()

#%%
# 1,2월 데이터만 필터링
df['comment_at'] = pd.to_datetime(df['comment_at'])
df_filtered = df[(df['comment_at'] >= '2024-01-01') & (df['comment_at'] < '2024-02-01')]
df_filtered.reset_index(drop=True, inplace=True)
df_filtered.head()

# /n 제거
df_filtered['text_raw'] = df_filtered['text_raw'].str.replace('\n', ' ')
df_filtered.head()

#%%
# is_politic 컬럼 추가
POLITICAL_KEYWORDS = [
    # 인물
    "윤석열", "이재명", "한동훈", "조국",
    # 정당
    "국민의힘", "민주당", "정의당",
    # 정책/제도
    "총선", "대통령", "국회", "선거", "탄핵",
    # 일반 정치 용어
    "정권", "야당", "여당"
]

import re

def is_political(text):
    if not isinstance(text, str):
        return 0
    for kw in POLITICAL_KEYWORDS:
        if kw in text:
            return 1
    return 0

df_filtered["is_politic"] = df_filtered["text_raw"].apply(is_political)

#%%
# 정치 혐오 사전
POL_HATE_WORDS = [
    "병신", "개새끼", "쓰레기", "무능",
    "망했다", "적폐", "벌레", "정신병"
]

# pol_tox score func. (간단 버전)
def calc_pol_tox(text):
    if not isinstance(text, str):
        return 0.0
    
    score = 0
    for w in POL_HATE_WORDS:
        if w in text:
            score += 1

    # K-HATERS 0/1/2 구조를 흉내
    if score == 0:
        return 0.0      # 비혐오
    elif score == 1:
        return 0.5      # 약한 정치 혐오 (Level-1)
    else:
        return 1.0      # 강한 정치 혐오 (Level-2)

#%%
df_filtered["pol_tox"] = 0.0

mask = df_filtered["is_politic"] == 1
df_filtered.loc[mask, "pol_tox"] = df_filtered.loc[mask, "text_raw"].apply(calc_pol_tox)
df_filtered.head()

#%%
# pol_tox 분포 확인
df_f = df_filtered[df_filtered['is_politic'] == 1]
print("정치 관련 댓글 수:", len(df_f))
print("정치 관련 댓글 중 정치 혐오 댓글 수:", len(df_f[df_f['pol_tox'] > 0]))
print("정치 혐오 비율: {:.2f}%".format(100 * len(df_f[df_f['pol_tox'] > 0]) / len(df_f)))


#%%
import matplotlib.pyplot as plt

plt.hist(df_f['pol_tox'], bins=[-0.1, 0.1, 0.6, 1.1], edgecolor='black')
plt.xticks([0, 0.5, 1], ['0 (Non-toxic)', '0.5 (Mildly toxic)', '1.0 (Highly toxic)']) 
plt.xlabel('Political Toxicity Score')
plt.ylabel('Number of Comments')
plt.title('Distribution of Political Toxicity Scores in Comments')
plt.show()

#%%
df_filtered.loc[
    (df_filtered["is_politic"] == 1) & (df_filtered["pol_tox"] == 0.5),
    "text_raw"
].sample(10, random_state=42)