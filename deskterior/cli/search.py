"""Search CLI entrypoint.

Preferred command:
    python -m deskterior.cli.search
"""

from deskterior.cli.cli import run_search
from deskterior.database.manager import DBManager
from deskterior.retrieval.searcher import ProductSearcher


def main() -> None:
    db = DBManager()
    try:
        searcher = ProductSearcher(db)

        print("검색 준비 완료. 종료는 'q'")
        while True:
            query = input("\n검색어 입력 (종료=q): ").strip()

            if query.lower() == "q":
                break
            if query:
                run_search(searcher, query)
    finally:
        db.close()


if __name__ == "__main__":
    main()
