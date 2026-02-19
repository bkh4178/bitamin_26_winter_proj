import argparse

from crawling_naver_1 import run_crawling
from politic_filter_1 import run_politic_filter
from toxicity_module_1 import run_toxicity
from stock_filter_1 import run_stock_filter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, required=True)
    ap.add_argument("--comment_template_url", required=True)

    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--per_article", type=int, default=30)

    ap.add_argument("--tox_mode", default="weight", choices=["drop", "weight"])
    ap.add_argument("--tox_tau", type=float, default=0.90)

    args = ap.parse_args()

    year = args.year

    print("\n" + "=" * 80)
    print(f"{year} 전체 파이프라인 시작")
    print("=" * 80)

    # 1️⃣ 크롤링
    print("\n[STEP 1] Crawling")
    #run_crawling(
    #    year=year,
    #    comment_template_url=args.comment_template_url,
    #    topk=args.topk,
    #    per_article=args.per_article,
    #)

    # 2️⃣ 정치 필터
    print("\n[STEP 2] Political Filter")
    run_politic_filter(year)

    # 3️⃣ Toxicity
    print("\n[STEP 3] Toxicity")
    run_toxicity(
        year=year,
        mode=args.tox_mode,
        tau=args.tox_tau,
    )

    # 4️⃣ 주식 필터
    print("\n[STEP 4] Stock Filter")
    run_stock_filter(year)

    print("\n" + "=" * 80)
    print("🎉 전체 파이프라인 완료")
    print("=" * 80)


if __name__ == "__main__":
    main()