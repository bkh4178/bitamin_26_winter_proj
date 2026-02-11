import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from collections import Counter

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'  # Windows
# plt.rcParams['font.family'] = 'AppleGothic'  # Mac
# plt.rcParams['font.family'] = 'NanumGothic'  # Linux
plt.rcParams['axes.unicode_minus'] = False

# 스타일 설정
sns.set_style("whitegrid")
sns.set_palette("husl")

print("=" * 80)
print("정치 키워드 기반 댓글 분류 분석")
print("=" * 80)

# ========== 1. 데이터 로딩 ==========
print("\n1. 데이터 로딩 중...")
df = pd.read_csv('C:\\Users\\philh\\OneDrive\\바탕 화면\\댓글수집_여원코드\\0205_1차수집_comments.csv')

# 빈 댓글 제거
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

# 패턴별로 어떤 키워드가 포함되는지 출력
print("정의된 정치 키워드 패턴:")
for i, pattern in enumerate(politics_patterns, 1):
    keywords = pattern.split('|')
    print(f"  패턴 {i}: {', '.join(keywords)}")

# ========== 3. 분류 함수 정의 ==========
def contains_political_keywords(text):
    """
    텍스트에 정치 키워드가 포함되어 있는지 확인
    
    Parameters:
    -----------
    text : str
        검사할 댓글 텍스트
    
    Returns:
    --------
    bool : 정치 키워드 포함 여부
    """
    if pd.isna(text):
        return False
    
    text_lower = text.lower()
    
    for pattern in politics_patterns:
        if re.search(pattern, text_lower):
            return True
    
    return False

def get_matched_keywords(text):
    """
    텍스트에서 매칭된 정치 키워드 목록 반환
    
    Parameters:
    -----------
    text : str
        검사할 댓글 텍스트
    
    Returns:
    --------
    list : 매칭된 키워드 리스트
    """
    if pd.isna(text):
        return []
    
    text_lower = text.lower()
    matched = []
    
    for pattern in politics_patterns:
        # 패턴에 매칭되는 키워드 찾기
        keywords = pattern.split('|')
        for keyword in keywords:
            if keyword in text_lower:
                matched.append(keyword)
    
    return list(set(matched))  # 중복 제거

# ========== 4. 분류 실행 ==========
print("\n3. 정치 키워드 기반 댓글 분류 중...")

df['is_political'] = df['text_raw'].apply(contains_political_keywords)
df['matched_keywords'] = df['text_raw'].apply(get_matched_keywords)
df['keyword_count'] = df['matched_keywords'].apply(len)

# 분류 결과
political_comments = df[df['is_political']].copy()
non_political_comments = df[~df['is_political']].copy()

print("\n분류 결과:")
print("-" * 80)
print(f"정치 관련 댓글: {len(political_comments):,}개 ({len(political_comments)/len(df)*100:.1f}%)")
print(f"비정치 댓글: {len(non_political_comments):,}개 ({len(non_political_comments)/len(df)*100:.1f}%)")

# ========== 5. 매칭된 키워드 통계 ==========
print("\n4. 매칭된 정치 키워드 통계")
print("-" * 80)

# 모든 매칭된 키워드 수집
all_matched_keywords = []
for keywords in political_comments['matched_keywords']:
    all_matched_keywords.extend(keywords)

keyword_freq = Counter(all_matched_keywords)
print(f"\n가장 많이 매칭된 정치 키워드 TOP 20:")
for keyword, count in keyword_freq.most_common(20):
    pct = count / len(political_comments) * 100
    print(f"  {keyword:15s}: {count:5,}회 (정치 댓글의 {pct:5.1f}%)")

# ========== 6. 샘플 댓글 확인 ==========
print("\n5. 샘플 댓글 확인")
print("-" * 80)

print("\n[정치 관련 댓글 샘플 10개]")
political_sample = political_comments.sample(min(10, len(political_comments)), random_state=42)
for idx, row in political_sample.iterrows():
    text = row['text_raw'][:80].replace('\n', ' ')
    keywords = ', '.join(row['matched_keywords'])
    print(f"  [키워드: {keywords}]")
    print(f"  {text}...\n")

print("\n[비정치 댓글 샘플 10개]")
non_political_sample = non_political_comments.sample(min(10, len(non_political_comments)), random_state=42)
for idx, row in non_political_sample.iterrows():
    text = row['text_raw'][:80].replace('\n', ' ')
    print(f"  {text}...\n")

# ========== 7. 시각화 ==========
print("\n6. 시각화 생성 중...")

# 출력 디렉토리 생성
import os
output_dir = '/home/claude/political_classification_viz'
os.makedirs(output_dir, exist_ok=True)

# 7-1. 기본 분류 결과 (파이 차트)
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# 파이 차트
sizes = [len(political_comments), len(non_political_comments)]
labels = ['정치 관련', '비정치']
colors = ['#e74c3c', '#3498db']
explode = (0.05, 0)

axes[0].pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors, explode=explode,
           startangle=90, textprops={'fontsize': 14, 'weight': 'bold'})
axes[0].set_title('댓글 분류 결과', fontsize=16, fontweight='bold', pad=20)

# 막대 차트
axes[1].bar(['정치 관련', '비정치'], sizes, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
axes[1].set_ylabel('댓글 수', fontsize=12)
axes[1].set_title('댓글 분류 결과 (막대 그래프)', fontsize=16, fontweight='bold', pad=20)
for i, (label, size) in enumerate(zip(['정치 관련', '비정치'], sizes)):
    axes[1].text(i, size, f'{size:,}개\n({size/len(df)*100:.1f}%)',
                ha='center', va='bottom', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig(f'{output_dir}/01_classification_overview.png', dpi=300, bbox_inches='tight')
print(f"✓ 저장: 01_classification_overview.png")
plt.close()

# 7-2. 키워드별 매칭 빈도
fig, ax = plt.subplots(figsize=(14, 8))

top_20_keywords = keyword_freq.most_common(20)
keywords, counts = zip(*top_20_keywords)

colors_gradient = plt.cm.Reds(np.linspace(0.4, 0.9, len(keywords)))
bars = ax.barh(range(len(keywords)), counts, color=colors_gradient)

ax.set_yticks(range(len(keywords)))
ax.set_yticklabels(keywords, fontsize=11)
ax.set_xlabel('매칭 횟수', fontsize=12)
ax.set_title('정치 키워드별 매칭 빈도 TOP 20', fontsize=16, fontweight='bold', pad=20)
ax.invert_yaxis()

for i, (bar, count) in enumerate(zip(bars, counts)):
    pct = count / len(political_comments) * 100
    ax.text(count, i, f' {count:,}회 ({pct:.1f}%)', va='center', fontsize=10)

plt.tight_layout()
plt.savefig(f'{output_dir}/02_keyword_frequency.png', dpi=300, bbox_inches='tight')
print(f"✓ 저장: 02_keyword_frequency.png")
plt.close()

# 7-3. 댓글당 매칭된 키워드 개수 분포
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# 히스토그램
keyword_count_dist = political_comments['keyword_count'].value_counts().sort_index()
axes[0].bar(keyword_count_dist.index, keyword_count_dist.values, color='#e74c3c', alpha=0.8, edgecolor='black')
axes[0].set_xlabel('댓글당 매칭된 키워드 개수', fontsize=12)
axes[0].set_ylabel('댓글 수', fontsize=12)
axes[0].set_title('정치 댓글의 키워드 개수 분포', fontsize=14, fontweight='bold', pad=15)
for x, y in zip(keyword_count_dist.index, keyword_count_dist.values):
    axes[0].text(x, y, f'{y:,}', ha='center', va='bottom', fontsize=10)

# 파이 차트 (키워드 개수별 비율)
axes[1].pie(keyword_count_dist.values, labels=[f'{x}개' for x in keyword_count_dist.index],
           autopct='%1.1f%%', startangle=90, colors=plt.cm.Reds(np.linspace(0.4, 0.9, len(keyword_count_dist))))
axes[1].set_title('키워드 개수별 댓글 비율', fontsize=14, fontweight='bold', pad=15)

plt.tight_layout()
plt.savefig(f'{output_dir}/03_keyword_count_distribution.png', dpi=300, bbox_inches='tight')
print(f"✓ 저장: 03_keyword_count_distribution.png")
plt.close()

# 7-4. 시간대별 정치 댓글 비율
df['comment_date'] = pd.to_datetime(df['comment_at'])
df['date'] = df['comment_date'].dt.date

daily_stats = df.groupby(['date', 'is_political']).size().unstack(fill_value=0)

# 컬럼명 확인 및 처리
if True in daily_stats.columns:
    daily_stats['political'] = daily_stats[True]
else:
    daily_stats['political'] = 0
    
if False in daily_stats.columns:
    daily_stats['non_political'] = daily_stats[False]
else:
    daily_stats['non_political'] = 0

daily_stats['total'] = daily_stats['political'] + daily_stats['non_political']
daily_stats['political_ratio'] = (daily_stats['political'] / daily_stats['total'] * 100).fillna(0)

fig, axes = plt.subplots(2, 1, figsize=(16, 10))

# 일별 댓글 수
daily_stats[['political', 'non_political']].plot(kind='bar', stacked=True, ax=axes[0], 
                                 color=['#e74c3c', '#3498db'], alpha=0.8)
axes[0].set_xlabel('Date', fontsize=12)
axes[0].set_ylabel('Number of Comments', fontsize=12)
axes[0].set_title('일별 댓글 수 (정치 vs 비정치)', fontsize=14, fontweight='bold', pad=15)
axes[0].legend(['Political', 'Non-Political'], loc='upper right')
axes[0].tick_params(axis='x', rotation=45)

# 일별 정치 댓글 비율
axes[1].plot(range(len(daily_stats)), daily_stats['political_ratio'], 
            marker='o', linewidth=2, markersize=6, color='#e74c3c')
axes[1].fill_between(range(len(daily_stats)), daily_stats['political_ratio'], 
                     alpha=0.3, color='#e74c3c')
axes[1].set_xlabel('Date', fontsize=12)
axes[1].set_ylabel('Political Comment Ratio (%)', fontsize=12)
axes[1].set_title('일별 정치 댓글 비율 추이', fontsize=14, fontweight='bold', pad=15)
axes[1].set_xticks(range(0, len(daily_stats), max(1, len(daily_stats)//10)))
axes[1].set_xticklabels([str(daily_stats.index[i]) for i in range(0, len(daily_stats), max(1, len(daily_stats)//10))], 
                       rotation=45, ha='right')
axes[1].axhline(y=daily_stats['political_ratio'].mean(), color='red', 
               linestyle='--', linewidth=2, label=f'Mean: {daily_stats["political_ratio"].mean():.1f}%')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{output_dir}/04_temporal_analysis.png', dpi=300, bbox_inches='tight')
print(f"✓ 저장: 04_temporal_analysis.png")
plt.close()

# 7-5. 좋아요 수 비교 (정치 vs 비정치)
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# 박스플롯
data_for_box = [
    political_comments['like_count'],
    non_political_comments['like_count']
]
bp = axes[0].boxplot(data_for_box, labels=['정치 관련', '비정치'], patch_artist=True)
for patch, color in zip(bp['boxes'], ['#e74c3c', '#3498db']):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
axes[0].set_ylabel('좋아요 수', fontsize=12)
axes[0].set_title('좋아요 수 분포 비교', fontsize=14, fontweight='bold', pad=15)
axes[0].grid(True, alpha=0.3)

# 히스토그램 (정치)
axes[1].hist(political_comments['like_count'], bins=30, color='#e74c3c', 
            alpha=0.7, edgecolor='black', range=(0, 100))
axes[1].axvline(political_comments['like_count'].mean(), color='red', 
               linestyle='--', linewidth=2, label=f'평균: {political_comments["like_count"].mean():.1f}')
axes[1].set_xlabel('좋아요 수', fontsize=12)
axes[1].set_ylabel('댓글 수', fontsize=12)
axes[1].set_title('정치 댓글의 좋아요 분포', fontsize=14, fontweight='bold', pad=15)
axes[1].legend()

# 히스토그램 (비정치)
axes[2].hist(non_political_comments['like_count'], bins=30, color='#3498db',
            alpha=0.7, edgecolor='black', range=(0, 100))
axes[2].axvline(non_political_comments['like_count'].mean(), color='blue',
               linestyle='--', linewidth=2, label=f'평균: {non_political_comments["like_count"].mean():.1f}')
axes[2].set_xlabel('좋아요 수', fontsize=12)
axes[2].set_ylabel('댓글 수', fontsize=12)
axes[2].set_title('비정치 댓글의 좋아요 분포', fontsize=14, fontweight='bold', pad=15)
axes[2].legend()

plt.tight_layout()
plt.savefig(f'{output_dir}/05_like_comparison.png', dpi=300, bbox_inches='tight')
print(f"✓ 저장: 05_like_comparison.png")
plt.close()

# 7-6. 종합 대시보드
fig = plt.figure(figsize=(20, 12))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# 6-1. 분류 결과 (파이)
ax1 = fig.add_subplot(gs[0, 0])
ax1.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors, explode=explode,
       startangle=90, textprops={'fontsize': 11, 'weight': 'bold'})
ax1.set_title('분류 결과', fontsize=13, fontweight='bold', pad=10)

# 6-2. 통계 테이블
ax2 = fig.add_subplot(gs[0, 1])
ax2.axis('off')
summary_data = [
    ['구분', '정치 관련', '비정치'],
    ['댓글 수', f'{len(political_comments):,}', f'{len(non_political_comments):,}'],
    ['비율', f'{len(political_comments)/len(df)*100:.1f}%', f'{len(non_political_comments)/len(df)*100:.1f}%'],
    ['평균 좋아요', f'{political_comments["like_count"].mean():.1f}', f'{non_political_comments["like_count"].mean():.1f}'],
    ['최다 좋아요', f'{political_comments["like_count"].max()}', f'{non_political_comments["like_count"].max()}'],
]
table = ax2.table(cellText=summary_data, cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2)
for i in range(len(summary_data)):
    if i == 0:
        for j in range(3):
            table[(i, j)].set_facecolor('#34495e')
            table[(i, j)].set_text_props(weight='bold', color='white')
    else:
        table[(i, 0)].set_facecolor('#ecf0f1')
        table[(i, 1)].set_facecolor('#fadbd8')
        table[(i, 2)].set_facecolor('#d6eaf8')
ax2.set_title('분류 통계', fontsize=13, fontweight='bold', pad=10)

# 6-3. TOP 10 키워드
ax3 = fig.add_subplot(gs[0, 2])
top_10 = keyword_freq.most_common(10)
words, counts = zip(*top_10)
ax3.barh(range(len(words)), counts, color=plt.cm.Reds(np.linspace(0.5, 0.9, len(words))))
ax3.set_yticks(range(len(words)))
ax3.set_yticklabels(words, fontsize=9)
ax3.set_xlabel('빈도', fontsize=10)
ax3.set_title('TOP 10 정치 키워드', fontsize=13, fontweight='bold', pad=10)
ax3.invert_yaxis()
for i, count in enumerate(counts):
    ax3.text(count, i, f' {count:,}', va='center', fontsize=8)

# 6-4. 키워드 개수 분포
ax4 = fig.add_subplot(gs[1, :2])
keyword_count_dist = political_comments['keyword_count'].value_counts().sort_index()
ax4.bar(keyword_count_dist.index, keyword_count_dist.values, color='#e74c3c', 
       alpha=0.8, edgecolor='black')
ax4.set_xlabel('댓글당 매칭된 키워드 개수', fontsize=11)
ax4.set_ylabel('댓글 수', fontsize=11)
ax4.set_title('정치 댓글의 키워드 개수 분포', fontsize=13, fontweight='bold', pad=10)
for x, y in zip(keyword_count_dist.index, keyword_count_dist.values):
    ax4.text(x, y, f'{y:,}', ha='center', va='bottom', fontsize=9)

# 6-5. 좋아요 비교
ax5 = fig.add_subplot(gs[1, 2])
data_for_box = [political_comments['like_count'], non_political_comments['like_count']]
bp = ax5.boxplot(data_for_box, labels=['정치', '비정치'], patch_artist=True)
for patch, color in zip(bp['boxes'], ['#e74c3c', '#3498db']):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax5.set_ylabel('좋아요 수', fontsize=11)
ax5.set_title('좋아요 분포 비교', fontsize=13, fontweight='bold', pad=10)

# 6-6. 일별 추이
ax6 = fig.add_subplot(gs[2, :])
daily_stats['political_ratio'].plot(ax=ax6, marker='o', linewidth=2, 
                                    markersize=5, color='#e74c3c')
ax6.fill_between(range(len(daily_stats)), daily_stats['political_ratio'], 
                alpha=0.3, color='#e74c3c')
ax6.set_xlabel('Date', fontsize=11)
ax6.set_ylabel('Political Comment Ratio (%)', fontsize=11)
ax6.set_title('일별 정치 댓글 비율 추이', fontsize=13, fontweight='bold', pad=10)
ax6.set_xticks(range(0, len(daily_stats), max(1, len(daily_stats)//10)))
ax6.set_xticklabels([str(daily_stats.index[i]) for i in range(0, len(daily_stats), max(1, len(daily_stats)//10))], 
                   rotation=45, ha='right', fontsize=8)
ax6.axhline(y=daily_stats['political_ratio'].mean(), color='red', 
           linestyle='--', linewidth=2, label=f'Mean: {daily_stats["political_ratio"].mean():.1f}%')
ax6.legend()
ax6.grid(True, alpha=0.3)

plt.suptitle('정치 키워드 기반 댓글 분류 종합 대시보드', fontsize=18, fontweight='bold', y=0.995)
plt.savefig(f'{output_dir}/06_comprehensive_dashboard.png', dpi=300, bbox_inches='tight')
print(f"✓ 저장: 06_comprehensive_dashboard.png")
plt.close()

# ========== 8. 결과 저장 ==========
print("\n7. 분류 결과 저장 중...")

# CSV 파일로 저장
output_csv = f'{output_dir}/classified_comments.csv'
df[['news_id', 'comment_id', 'text_raw', 'is_political', 'matched_keywords', 
    'keyword_count', 'like_count', 'dislike_count', 'comment_at']].to_csv(
    output_csv, index=False, encoding='utf-8-sig')
print(f"✓ 저장: classified_comments.csv")

# 정치 댓글만 별도 저장
political_output_csv = f'{output_dir}/political_comments_only.csv'
political_comments[['news_id', 'comment_id', 'text_raw', 'matched_keywords', 
                   'keyword_count', 'like_count', 'dislike_count', 'comment_at']].to_csv(
    political_output_csv, index=False, encoding='utf-8-sig')
print(f"✓ 저장: political_comments_only.csv")

# 통계 요약
summary = {
    'total_comments': len(df),
    'political_comments': len(political_comments),
    'non_political_comments': len(non_political_comments),
    'political_ratio': len(political_comments) / len(df) * 100,
    'avg_likes_political': political_comments['like_count'].mean(),
    'avg_likes_non_political': non_political_comments['like_count'].mean(),
    'top_10_keywords': dict(keyword_freq.most_common(10)),
}

import json
with open(f'{output_dir}/classification_summary.json', 'w', encoding='utf-8') as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)
print(f"✓ 저장: classification_summary.json")

# ========== 9. 결과 요약 ==========
print("\n" + "=" * 80)
print("분석 완료!")
print("=" * 80)
print(f"\n저장 위치: {output_dir}")
print("\n생성된 파일:")
print("  [시각화]")
print("  01_classification_overview.png - 분류 결과 개요")
print("  02_keyword_frequency.png - 키워드별 매칭 빈도")
print("  03_keyword_count_distribution.png - 키워드 개수 분포")
print("  04_temporal_analysis.png - 시간대별 분석")
print("  05_like_comparison.png - 좋아요 수 비교")
print("  06_comprehensive_dashboard.png - 종합 대시보드")
print("\n  [데이터]")
print("  classified_comments.csv - 분류 결과 전체")
print("  political_comments_only.csv - 정치 댓글만")
print("  classification_summary.json - 통계 요약")

print("\n" + "=" * 80)
print("요약 통계")
print("=" * 80)
print(f"전체 댓글: {len(df):,}개")
print(f"정치 관련: {len(political_comments):,}개 ({len(political_comments)/len(df)*100:.1f}%)")
print(f"비정치: {len(non_political_comments):,}개 ({len(non_political_comments)/len(df)*100:.1f}%)")
print(f"\n평균 좋아요:")
print(f"  정치 관련: {political_comments['like_count'].mean():.2f}개")
print(f"  비정치: {non_political_comments['like_count'].mean():.2f}개")
print(f"\n가장 많이 매칭된 키워드 TOP 5:")
for keyword, count in keyword_freq.most_common(5):
    print(f"  {keyword}: {count:,}회")
print("=" * 80)
