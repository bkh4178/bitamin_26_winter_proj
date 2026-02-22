import os
import json
import re
from collections import Counter

import pandas as pd


politics_patterns = [
    r'윤석열|석열|윤통|용산|이재명|재명|개딸|한동훈|동훈|뚜껑|조국|문재인|재앙|박근혜|근혜|이명박|MB',
    r'민주당|더불어민주당|국민의힘|국힘|국짐|정의당|개혁신당|조국혁신당|좌파|우파|좌빨|수구|빨갱이|종북|토착왜구',
    r'탄핵|계엄|특검|비상계엄|내란|반역|구속|체포|영장|기소|검찰|독재|관권선거|부정선거|공천',
    r'의원|국회의원|국회|여당|야당|거대야당|당대표|원내대표|장관|차관|국무총리|대통령실|방통위|권익위',
    r'친문|친명|친윤|비윤|반윤|개헌|정권교체|심판|지지자|촛불집회|태극기부대|정치인|정치질'
]


def contains_political_keywords(text: str) -> bool:
    if pd.isna(text):
        return False

    text_lower = text.lower()
    for pattern in politics_patterns:
        if re.search(pattern, text_lower):
            return True
    return False


def get_matched_keywords(text: str):
    if pd.isna(text):
        return []

    text_lower = text.lower()
    matched = []

    for pattern in politics_patterns:
        keywords = pattern.split('|')
        for keyword in keywords:
            if keyword in text_lower:
                matched.append(keyword)

    return list(set(matched))


def run_politic_filter(year: int):

    print("=" * 80)
    print(f"{year} 정치 키워드 기반 댓글 필터 시작")
    print("=" * 80)

    input_file = f"data/NAVER/comments/comments_{year}_3_top5.csv"
    output_dir = "data/NAVER/political_filter"
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(input_file)

    df = df[df['text_raw'].notna() & (df['text_raw'].str.strip() != '')].copy()

    df['is_political'] = df['text_raw'].apply(contains_political_keywords)
    df['matched_keywords'] = df['text_raw'].apply(get_matched_keywords)
    df['keyword_count'] = df['matched_keywords'].apply(len)

    political_comments = df[df['is_political']].copy()
    non_political_comments = df[~df['is_political']].copy()

    # 키워드 통계
    all_matched_keywords = []
    for keywords in political_comments['matched_keywords']:
        all_matched_keywords.extend(keywords)

    keyword_freq = Counter(all_matched_keywords)

    # 전체 분류 저장
    df.to_csv(
        f"{output_dir}/comments_{year}_1_classified.csv",
        index=False,
        encoding="utf-8-sig"
    )

    # 정치만
    political_comments.to_csv(
        f"{output_dir}/comments_{year}_3_political_only.csv",
        index=False,
        encoding="utf-8-sig"
    )

    # 비정치만
    non_political_comments.to_csv(
        f"{output_dir}/comments_{year}_3_non_political_only.csv",
        index=False,
        encoding="utf-8-sig"
    )

    # 정치 제거 클린 데이터 (다음 단계 입력용)
    cleaned = non_political_comments.drop(
        columns=['is_political', 'matched_keywords', 'keyword_count'],
        errors='ignore'
    )

    cleaned.to_csv(
        f"{output_dir}/comments_political_removed_{year}_3.csv",
        index=False,
        encoding="utf-8-sig"
    )

    summary = {
        'total_comments': len(df),
        'political_comments': len(political_comments),
        'non_political_comments': len(non_political_comments),
        'political_ratio': len(political_comments) / len(df) * 100,
        'top_10_keywords': dict(keyword_freq.most_common(10)),
    }

    with open(f"{output_dir}/classification_summary_{year}.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n총 댓글: {len(df):,}")
    print(f"정치 댓글 제거: {len(political_comments):,}")
    print(f"남은 댓글: {len(non_political_comments):,}")
    print("=" * 80)