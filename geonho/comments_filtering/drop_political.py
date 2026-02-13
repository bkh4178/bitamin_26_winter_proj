import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from collections import Counter
import os
import json
from pathlib import Path

# ==============================
# argparse 추가
# ==============================
parser = argparse.ArgumentParser(description="정치 키워드 기반 댓글 분류 분석")
parser.add_argument("--inp", required=True, help="입력 CSV 경로")
parser.add_argument("--year", required=True, help="연도 (예: 2023)")
args = parser.parse_args()

INPUT_PATH = Path(args.inp)
YEAR = args.year

# ==============================
# 저장 경로 변경 (네 프로젝트 기준)
# ==============================
output_dir = Path("data/NAVER/comments")
output_dir.mkdir(parents=True, exist_ok=True)

DROP_OUTPUT = output_dir / f"comments_{YEAR}_drop_poli.csv"

# ==============================
# 한글 폰트 설정
# ==============================
plt.rcParams['font.family'] = 'AppleGothic'  # Mac
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")
sns.set_palette("husl")

print("=" * 80)
print("정치 키워드 기반 댓글 분류 분석")
print("=" * 80)

# ========== 1. 데이터 로딩 ==========
print("\n1. 데이터 로딩 중...")

if not INPUT_PATH.exists():
    raise FileNotFoundError(f"파일 없음: {INPUT_PATH}")

df = pd.read_csv(INPUT_PATH)

df_original_len = len(df)
df = df[df['text_raw'].notna() & (df['text_raw'].str.strip() != '')].copy()

print(f"   전체 댓글 수: {df_original_len:,}개")
print(f"   빈 댓글 제거 후: {len(df):,}개")
print(f"   제거된 댓글: {df_original_len - len(df):,}개")

# ========== 2. 정치 키워드 정의 ==========
print("\n2. 정치 키워드 패턴 정의")
print("-" * 80)

politics_patterns = [
    r'윤석열|석열|이재명|한동훈',
    r'민주당|국민의힘|민주|국힘',
    r'탄핵|계엄|내란|구속|체포',
    r'의원|국회|여당|야당',
]

print("정의된 정치 키워드 패턴:")
for i, pattern in enumerate(politics_patterns, 1):
    keywords = pattern.split('|')
    print(f"  패턴 {i}: {', '.join(keywords)}")

# ========== 3. 분류 함수 정의 ==========
def contains_political_keywords(text):
    if pd.isna(text):
        return False
    text_lower = str(text).lower()
    for pattern in politics_patterns:
        if re.search(pattern, text_lower):
            return True
    return False

def get_matched_keywords(text):
    if pd.isna(text):
        return []
    text_lower = str(text).lower()
    matched = []
    for pattern in politics_patterns:
        keywords = pattern.split('|')
        for keyword in keywords:
            if keyword in text_lower:
                matched.append(keyword)
    return list(set(matched))

# ========== 4. 분류 실행 ==========
print("\n3. 정치 키워드 기반 댓글 분류 중...")

df['is_political'] = df['text_raw'].apply(contains_political_keywords)
df['matched_keywords'] = df['text_raw'].apply(get_matched_keywords)
df['keyword_count'] = df['matched_keywords'].apply(len)

political_comments = df[df['is_political']].copy()
non_political_comments = df[~df['is_political']].copy()

print("\n분류 결과:")
print("-" * 80)
print(f"정치 관련 댓글: {len(political_comments):,}개 ({len(political_comments)/len(df)*100:.1f}%)")
print(f"비정치 댓글: {len(non_political_comments):,}개 ({len(non_political_comments)/len(df)*100:.1f}%)")

# ========== 5. 정치 댓글 제거 (🔥 반영된 부분) ==========
df_clean = df[~df['is_political']].copy()
print(f"\n정치 댓글 제거 후 남은 댓글 수: {len(df_clean):,}개")

# ========== 6. 최종 저장 (네 요구사항 반영) ==========
df_clean.drop(columns=['is_political','matched_keywords','keyword_count']).to_csv(
    DROP_OUTPUT,
    index=False,
    encoding='utf-8-sig'
)

print(f"\n✓ 정치 댓글 제거 파일 저장 완료: {DROP_OUTPUT}")

# ========== 7. 요약 JSON 저장 ==========
summary = {
    'year': YEAR,
    'total_comments': len(df),
    'political_comments': len(political_comments),
    'non_political_comments': len(non_political_comments),
    'political_ratio': len(political_comments) / len(df) * 100,
    'avg_likes_political': political_comments['like_count'].mean(),
    'avg_likes_non_political': non_political_comments['like_count'].mean(),
    'top_10_keywords': dict(Counter(
        [k for sublist in political_comments['matched_keywords'] for k in sublist]
    ).most_common(10))
}

with open(output_dir / f"comments_{YEAR}_drop_poli_summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

print("=" * 80)
print("분석 완료")
print("=" * 80)

#%%
import pandas as pd
# 2023 정치 제거 csv
df_2023 = pd.read_csv("/Users/user/Desktop/bitamin/26_winter_proj/data/NAVER/comments/comments_2023_drop_poli.csv")
# 2024 정치 제거 csv
df_2024 = pd.read_csv("/Users/user/Desktop/bitamin/26_winter_proj/data/NAVER/comments/comments_2024_drop_poli.csv")    
# 2025 정치 제거 csv
df_2025 = pd.read_csv("/Users/user/Desktop/bitamin/26_winter_proj/data/NAVER/comments/comments_2025_drop_poli.csv")    

# 2023 일별 댓글수 확인
df_2023['comment_at'] = pd.to_datetime(df_2023['comment_at'])
df_2023['comment_date'] = df_2023['comment_at'].dt.date
daily_counts_2023 = df_2023.groupby('comment_date').size()
print("2023년 일별 댓글 수:")
print(daily_counts_2023)
# 2023 일별 댓글수 통계량
print("\n2023년 댓글 수 통계량:")
print(daily_counts_2023.describe())