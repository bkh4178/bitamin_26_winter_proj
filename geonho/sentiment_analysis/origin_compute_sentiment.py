
import argparse
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
import os

# ==========================
# 1. Argument 받기
# ==========================

parser = argparse.ArgumentParser()
parser.add_argument("--year", type=int, required=True)
args = parser.parse_args()

year = args.year

# ==========================
# 2. 경로 설정
# ==========================

base_dir = os.path.expanduser("~/Desktop/bitamin/26_winter_proj")
file_path = os.path.join(base_dir, f"data/NAVER/final_filtered/comments_stock_clean_{year}.csv")
save_dir = os.path.join(base_dir, "data/NAVER/sentiment_scores_raw")

os.makedirs(save_dir, exist_ok=True)

print(f"===== {year} 시작 (재정규화 이전 버전) =====")

# ==========================
# 3. 데이터 로드
# ==========================

df = pd.read_csv(file_path)

df["comment_at"] = pd.to_datetime(df["comment_at"])
df = df[df["comment_at"].dt.year == year]

df = df[(df["is_empty"] == 0) & (df["keep"] == 1)]
df = df.dropna(subset=["text_raw"])
df = df[df["text_raw"].str.strip() != ""]
df.reset_index(drop=True, inplace=True)

if len(df) == 0:
    print("데이터 없음. 종료.")
    exit()

print("댓글 수:", len(df))

# ==========================
# 4. 모델 로드
# ==========================

model_name = "snunlp/KR-FinBert-SC"

device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
model.to(device)
model.eval()

# ==========================
# 5. 감성 계산 함수 (재정규화 이전: pos - neg)
# ==========================

@torch.no_grad()
def get_sentiment_score(texts, batch_size=64, max_length=64):

    neg_id = model.config.label2id["negative"]
    pos_id = model.config.label2id["positive"]

    scores = []

    total_batches = (len(texts) + batch_size - 1) // batch_size

    for i in tqdm(range(0, len(texts), batch_size), total=total_batches):
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

        # 기본 방식: pos - neg
        sentiment = probs[:, pos_id] - probs[:, neg_id]

        scores.extend(sentiment.cpu().numpy())

    return scores


print("감성 계산 시작")
df["sentiment_score"] = get_sentiment_score(df["text_raw"].tolist())
print("감성 계산 완료")

# ==========================
# 6. 저장
# ==========================

save_path = os.path.join(save_dir, f"comments_sentiment_{year}.csv")
df.to_csv(save_path, index=False)

print(f"{year} 저장 완료:", save_path)

# ==========================
# 7. 분포 확인
# ==========================

print(df["sentiment_score"].describe())