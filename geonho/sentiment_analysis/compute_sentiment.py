'''
python geonho/sentiment_analysis/compute_sentiment.py --year 2023
python geonho/sentiment_analysis/compute_sentiment.py --year 2024
python geonho/sentiment_analysis/compute_sentiment.py --year 2025

각각 터미널에서 실행, 코드, 파일 경로는 바꿔야  함
'''
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

# 파일 경로 설정 (각자 환경에 맞게 수정 필요)
file_path = f'/Users/user/Desktop/bitamin/26_winter_proj/data/NAVER/final_filtered/comments_final_stock_only_{year}.csv'
save_dir = f'/Users/user/Desktop/bitamin/26_winter_proj/data/NAVER/sentiment_scores'

os.makedirs(save_dir, exist_ok=True)

print(f"===== {year} 시작 =====")

# ==========================
# 2. 데이터 로드
# ==========================

df = pd.read_csv(file_path)

df["comment_at"] = pd.to_datetime(df["comment_at"])
df = df[df["comment_at"].dt.year == year]

df = df[(df["is_empty"] == 0) & (df["keep"] == 1)]
df = df.dropna(subset=["text_raw"])
df = df[df["text_raw"].str.strip() != ""]
df.reset_index(drop=True, inplace=True)

print("댓글 수:", len(df))

# ==========================
# 3. 모델 로드
# ==========================

model_name = "snunlp/KR-FinBert-SC"

device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
model.to(device)
model.eval()

# ==========================
# 4. 감성 계산 함수
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
# 5. 저장
# ==========================

df.to_csv(f'{save_dir}/comments_sentiment_{year}.csv', index=False)

print(f"{year} 저장 완료")