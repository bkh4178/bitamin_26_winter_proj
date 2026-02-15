#%%
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm

# ==========================
# 0. 설정
# ==========================

year = 2025
file_path = f'/Users/user/Desktop/bitamin/26_winter_proj/data/NAVER/final_filtered/comments_stock_clean_{year}.csv'

# ==========================
# 1. 데이터 로드
# ==========================

df = pd.read_csv(file_path)

df["comment_at"] = pd.to_datetime(df["comment_at"])

# 예시: 2025년 1월만 테스트
df = df[
    (df["comment_at"].dt.year == 2025) &
    (df["comment_at"].dt.month == 1)
].copy()

# 유효 댓글만
df = df[(df["is_empty"] == 0) & (df["keep"] == 1)]
df = df.dropna(subset=["text_raw"])
df = df[df["text_raw"].str.strip() != ""]

df.reset_index(drop=True, inplace=True)

print("댓글 수:", len(df))

# ==========================
# 2. 모델 로드
# ==========================

model_name = "snunlp/KR-FinBert-SC"

device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
model.to(device)
model.eval()

# ==========================
# 3. 감성 점수 계산 (댓글 단위)
# ==========================

@torch.no_grad()
def get_sentiment_score(texts, batch_size=64, max_length=64):

    neg_id = model.config.label2id["negative"]
    pos_id = model.config.label2id["positive"]

    scores = []

    for i in tqdm(range(0, len(texts), batch_size)):
        batch = texts[i:i+batch_size]

        encoded = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        ).to(device)

        outputs = model(**encoded)
        probs = torch.softmax(outputs.logits, dim=1)

        sentiment = (probs[:, pos_id] - probs[:, neg_id]) / \
            (probs[:, pos_id] + probs[:, neg_id] + 1e-8)
        scores.extend(sentiment.cpu().numpy())

    return scores


print("감성 계산 시작")
df["sentiment_score"] = get_sentiment_score(df["text_raw"].tolist())
print("감성 계산 완료")

# ==========================
# 4. 저장 (댓글 단위 유지)
# ==========================

df.to_csv("comments_with_sentiment_2025_01_test.csv", index=False)

print("저장 완료")

# ==========================
# 5. 분포 확인
# ==========================

print(df["sentiment_score"].describe())




#%%
'''
시각화
'''
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(8,5))
sns.histplot(df["sentiment_score"], bins=50, kde=True)
plt.axvline(0, linestyle='--')
plt.title("Sentiment Score Distribution")
plt.show()

#%%
print("부정 비율:", (df["sentiment_score"] < 0).mean())
print("긍정 비율:", (df["sentiment_score"] > 0).mean())

#%%
df.groupby(df["comment_at"].dt.date)["sentiment_score"].mean().plot()