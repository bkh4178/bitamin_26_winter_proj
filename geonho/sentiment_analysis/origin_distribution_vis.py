
#%%
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================
# 1. 기본 설정
# ==========================

data_dir = '/Users/user/Desktop/bitamin/26_winter_proj/data/NAVER/sentiment_scores_raw'

years = [2023, 2024, 2025]

dfs = {}

# ==========================
# 2. 데이터 로드
# ==========================

for y in years:
    path = os.path.join(data_dir, f"comments_sentiment_{y}.csv")
    
    if not os.path.exists(path):
        print(f"{y} 파일 없음:", path)
        continue
    
    df = pd.read_csv(path)
    dfs[y] = df
    
    print(f"\n===== {y} 댓글 단위 통계 =====")
    print(df["sentiment_score"].describe())
    print("부정 비율:", (df["sentiment_score"] < 0).mean())
    print("긍정 비율:", (df["sentiment_score"] > 0).mean())

# ==========================
# 3. 댓글 단위 분포 시각화
# ==========================

for y in years:
    if y not in dfs:
        continue
    
    plt.figure(figsize=(7,4))
    plt.hist(dfs[y]["sentiment_score"], bins=60)
    plt.axvline(0, linestyle="--")
    plt.title(f"Comment-level Sentiment Distribution - {y}")
    plt.xlabel("sentiment_score")
    plt.ylabel("count")
    plt.show()

# ==========================
# 4. 연도별 KDE 비교
# ==========================

plt.figure(figsize=(8,6))

for y in years:
    if y not in dfs:
        continue
    
    sns.kdeplot(dfs[y]["sentiment_score"], label=str(y), fill=False)

plt.axvline(0, linestyle="--")
plt.title("Comment-level Sentiment Distribution Comparison (Raw: pos-neg)")
plt.legend()
plt.show()

# ==========================
# 5. 일별 평균 감성 계산 및 분포
# ==========================

plt.figure(figsize=(8,6))

for y in years:
    if y not in dfs:
        continue
    
    df = dfs[y].copy()
    df["date"] = pd.to_datetime(df["comment_at"]).dt.date
    
    daily_mean = df.groupby("date")["sentiment_score"].mean()
    
    print(f"\n===== {y} 일별 평균 감성 통계 =====")
    print(daily_mean.describe())
    
    sns.kdeplot(daily_mean, label=str(y), fill=False)

plt.axvline(0, linestyle="--")
plt.title("Daily Mean Sentiment Distribution (Raw: pos-neg)")
plt.legend()
plt.show()

# ==========================
# 6. 일별 평균 시계열 시각화
# ==========================

for y in years:
    if y not in dfs:
        continue
    
    df = dfs[y].copy()
    df["date"] = pd.to_datetime(df["comment_at"]).dt.date
    daily_mean = df.groupby("date")["sentiment_score"].mean()
    
    plt.figure(figsize=(10,4))
    plt.plot(daily_mean.index, daily_mean.values)
    plt.axhline(0, linestyle="--")
    plt.title(f"Daily Mean Sentiment - {y} (Raw: pos-neg)")
    plt.xticks(rotation=45)
    plt.show()

print("\n분포 확인 완료 (원래 방식: pos - neg)")