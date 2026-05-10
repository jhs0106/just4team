# === test.py — 진입점 (검색 / 셋업 추천 인터랙티브 메뉴)

from cli import run_search, run_setup_recommendation
from db.db_manager import DBManager
from search.recommender import SetupRecommender
from search.searcher import ProductSearcher


def main():
    db = DBManager()
    try:
        searcher = ProductSearcher(db)
        recommender = SetupRecommender(searcher)

        print("검색 준비 완료. 종료는 'q'")
        while True:
            print("\n1: 단일 상품 검색")
            print("2: 셋업 추천")
            choice = input("모드 선택 (1/2, 종료=q): ").strip().lower()

            if choice == "q":
                break
            if choice == "1":
                query = input("검색어 입력: ").strip()
                if query:
                    run_search(searcher, query)
                continue
            if choice == "2":
                run_setup_recommendation(recommender)
                continue

            print("올바른 메뉴를 선택하세요.")
    finally:
        db.close()


if __name__ == "__main__":
    main()
