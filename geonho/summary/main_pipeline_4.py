import argparse

from crawling_naver_4 import run_crawling
from politic_filter_4 import run_politic_filter
from toxicity_module_4 import run_toxicity
from stock_filter_4 import run_stock_filter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, required=True)
    ap.add_argument("--comment_template_url", required=True)

    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--per_article", type=int, default=30)

    ap.add_argument("--tox_mode", default="weight", choices=["drop", "weight"])
    ap.add_argument("--tox_tau", type=float, default=0.90)
    ap.add_argument("--start_date", default="0101", help="MMDD 형식")
    ap.add_argument("--end_date", default="1231", help="MMDD 형식")

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
    #    start_date=args.start_date,
    #    end_date=args.end_date,
    #)

    # 2️⃣ 정치 필터
    print("\n[STEP 2] Political Filter")
    #run_politic_filter(year, start_date=args.start_date)

    # 3️⃣ Toxicity
    print("\n[STEP 3] Toxicity")
    #run_toxicity(
    #    year=year,
    #    mode=args.tox_mode,
    #    tau=args.tox_tau,
    #    start_date=args.start_date,
    #)

    # 4️⃣ 주식 필터
    print("\n[STEP 4] Stock Filter")
    run_stock_filter(year, start_date=args.start_date)

    print("\n" + "=" * 80)
    print("🎉 전체 파이프라인 완료")
    print("=" * 80)


if __name__ == "__main__":
    main()