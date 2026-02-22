import pandas as pd
import re
from collections import Counter
import json
from pathlib import Path


def run_stock_filter(year: int, start_date: str):

    print("=" * 80)
    print(f"{year} 주식/경제 키워드 분석 + 자동 필터링 시작")
    print("=" * 80)

    # --------------------------
    # 1. 데이터 로딩
    # --------------------------
    input_file = Path(f"data/NAVER/toxicity/comments_toxicity_kept_{year}_4_{start_date}.csv")
    output_dir = Path("data/NAVER/final_filtered")
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_file.exists():
        raise FileNotFoundError(f"Input not found: {input_file}")

    df = pd.read_csv(input_file)

    df = df[df['text_raw'].notna() & (df['text_raw'].str.strip() != '')].copy()

    print(f"전체 댓글 수: {len(df):,}")

    # --------------------------
    # 2. 키워드 추출
    # --------------------------
    def extract_words(text):
        if pd.isna(text):
            return []
        return re.findall(r'[가-힣]{2,6}|[a-zA-Z]{2,10}', text.lower())

    all_words = []
    for text in df['text_raw']:
        all_words.extend(extract_words(text))

    word_freq = Counter(all_words)

    stock_patterns = [
        r'코스피|코스닥|kospi|kosdaq',
        r'주식|주가|증시|시장|장',
        r'매수|매도|투자|손절|익절',
        r'개미|외인|기관|외국인',
        r'삼전|삼성전자|하닉|하이닉스',
        r'지수|시총|배당|상장',
        r'급등|급락|폭등|폭락|상승|하락',
        r'수급|거래량|환율|금리',
        r'종목|실적|반도체|전지',
        r'펀드|연기금|국민연금',
    ]

    def is_stock_related_word(word):
        return any(re.search(pattern, word) for pattern in stock_patterns)

    word_document_count = {}
    for text in df['text_raw']:
        words = set(extract_words(text))
        for word in words:
            word_document_count[word] = word_document_count.get(word, 0) + 1

    word_df = sorted(word_document_count.items(), key=lambda x: x[1], reverse=True)

    threshold = len(df) * 0.01
    core_keywords = []
    support_keywords = []

    for word, doc_count in word_df:
        if is_stock_related_word(word):
            pct = doc_count / len(df) * 100
            if doc_count >= threshold:
                core_keywords.append((word, doc_count, pct))
            elif doc_count >= threshold * 0.3:
                support_keywords.append((word, doc_count, pct))

    CORE_STOCK_KEYWORDS = [w for w, _, _ in core_keywords]
    SUPPORT_STOCK_KEYWORDS = [w for w, _, _ in support_keywords]

    print(f"핵심 키워드 {len(CORE_STOCK_KEYWORDS)}개")
    print(f"보조 키워드 {len(SUPPORT_STOCK_KEYWORDS)}개")

    # --------------------------
    # 3. 점수 계산
    # --------------------------
    def score_comment(text):
        if pd.isna(text):
            return 0
        text_lower = text.lower()
        core_count = sum(1 for kw in CORE_STOCK_KEYWORDS if kw in text_lower)
        support_count = sum(1 for kw in SUPPORT_STOCK_KEYWORDS if kw in text_lower)
        return core_count * 10 + support_count * 3

    def classify_comment(text, score):
        if score < 10:
            return 'other'
        text_lower = text.lower() if pd.notna(text) else ''
        has_core = any(kw in text_lower for kw in CORE_STOCK_KEYWORDS)
        return 'stock' if has_core else 'other'

    df['stock_score'] = df['text_raw'].apply(score_comment)
    df['is_stock'] = df.apply(
        lambda row: classify_comment(row['text_raw'], row['stock_score']),
        axis=1
    )

    stock_df = df[df['is_stock'] == 'stock'].copy()
    other_df = df[df['is_stock'] == 'other'].copy()

    print(f"주식/경제 관련: {len(stock_df):,}개")
    print(f"기타: {len(other_df):,}개")

    # --------------------------
    # 4. 저장
    # --------------------------
    # 키워드 결과
    keyword_result = {
        'core_keywords': [{'word': w, 'count': c, 'percentage': p} for w, c, p in core_keywords],
        'support_keywords': [{'word': w, 'count': c, 'percentage': p} for w, c, p in support_keywords],
    }

    with open(output_dir / f"keyword_analysis_result_{year}_4_{start_date}.json", "w", encoding="utf-8") as f:
        json.dump(keyword_result, f, ensure_ascii=False, indent=2)

    df.to_csv(output_dir / f"classified_stock_comments_{year}_4_{start_date}.csv", index=False, encoding="utf-8-sig")
    stock_df.to_csv(output_dir / f"comments_final_stock_only_{year}_4_{start_date}.csv", index=False, encoding="utf-8-sig")

    stock_df.drop(columns=['stock_score', 'is_stock'], errors='ignore').to_csv(
        output_dir / f"comments_stock_clean_{year}_4_{start_date}.csv",
        index=False,
        encoding="utf-8-sig"
    )

    other_df.to_csv(output_dir / f"comments_other_{year}_4_{start_date}.csv", index=False, encoding="utf-8-sig")

    summary = {
        'total_comments': len(df),
        'stock_comments': len(stock_df),
        'other_comments': len(other_df),
        'stock_ratio': len(stock_df) / len(df) * 100,
    }

    with open(output_dir / f"stock_classification_summary_{year}_4_{start_date}.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("=" * 80)
    print("Stock Filter 완료")
    print("=" * 80)