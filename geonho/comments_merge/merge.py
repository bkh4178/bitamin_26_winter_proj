import pandas as pd
from glob import glob


'''
연도별 감성 점수 파일을 로드하여 하나의 데이터프레임으로 병합
'''
def load_year(year):
    pattern = f"data/NAVER/sentiment_scores/sentiment_with_prob_{year}_*.csv"
    files = glob(pattern)

    if len(files) == 0:
        raise ValueError(f"{year} 파일이 없습니다. 패턴: {pattern}")

    print(f"{year} → {len(files)}개 파일 로드")

    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

    df["comment_at"] = pd.to_datetime(df["comment_at"])
    df["date"] = df["comment_at"].dt.date

    return df


df_2023 = load_year(2023)
df_2024 = load_year(2024)
df_2025 = load_year(2025)

print("2023:", len(df_2023))
print("2024:", len(df_2024))
print("2025:", len(df_2025))

df_all = pd.concat([df_2023, df_2024, df_2025])
df_all = df_all.sort_values("date").reset_index(drop=True)


def make_daily(df):
    df = df.copy()

    # 댓글 단위 감성 계산
    df["sent_raw"] = df["p_pos"] - df["p_neg"]

    # 가중 감성
    df["sent_raw_weighted"] = df["sent_raw"] * df["weight"]
    df["p_pos_weighted"] = df["p_pos"] * df["weight"]
    df["p_neg_weighted"] = df["p_neg"] * df["weight"]
    df["p_neu_weighted"] = df["p_neu"] * df["weight"]

    grouped = df.groupby("date")

    daily = grouped.agg(
        weight_sum=("weight", "sum"),
        daily_volume=("p_pos", "size"),

        # 단순 평균 (비교용으로 남겨도 됨)
        daily_pos_mean=("p_pos", "mean"),
        daily_neg_mean=("p_neg", "mean"),
    )

    # -----------------------------
    # 1️⃣ Weighted mean 확률
    # -----------------------------
    daily["pos_mean_w"] = (
        grouped["p_pos_weighted"].sum() / daily["weight_sum"]
    )

    daily["neg_mean_w"] = (
        grouped["p_neg_weighted"].sum() / daily["weight_sum"]
    )

    daily["neu_mean_w"] = (
        grouped["p_neu_weighted"].sum() / daily["weight_sum"]
    )

    # -----------------------------
    # 2️⃣ Weighted sentiment
    # -----------------------------
    daily["sent_raw_w"] = (
        grouped["sent_raw_weighted"].sum() / daily["weight_sum"]
    )

    daily["sent_norm_w"] = (
        (daily["pos_mean_w"] - daily["neg_mean_w"]) /
        (daily["pos_mean_w"] + daily["neg_mean_w"] + 1e-8)
    )

    daily["pos_ratio_soft_w"] = (
        daily["pos_mean_w"] /
        (daily["pos_mean_w"] + daily["neg_mean_w"] + 1e-8)
    )

    daily["sent_strength_w"] = (
        daily["pos_mean_w"] + daily["neg_mean_w"]
    )

    daily["sent_std"] = grouped["sent_raw"].std()

    daily["weight_sq_sum"] = grouped["weight"].apply(lambda x: (x**2).sum())
    daily["effective_n"] = (daily["weight_sum"]**2) / (daily["weight_sq_sum"] + 1e-8)

    # -----------------------------
    # 3️⃣ Negativity + Relative Change
    # -----------------------------
    daily = daily.sort_index()
    
    # 절대 부정 강도
    daily["neg_score"] = daily["neg_mean_w"]

    # 30일 이동평균 대비 변화
    daily["neg_roll30"] = daily["neg_score"].rolling(30).mean()
    daily["neg_delta"] = daily["neg_score"] - daily["neg_roll30"]

    # 60일 Z-score (권장)
    daily["neg_z"] = (
        (daily["neg_score"] - daily["neg_score"].rolling(60).mean()) /
        (daily["neg_score"].rolling(60).std() + 1e-8)
    )

    return daily.reset_index()


daily_all = make_daily(df_all)
daily_all = daily_all.dropna().reset_index(drop=True)



