"""
main.py — 전체 진입점

사용법:
  python main.py                     # 인터랙티브 메뉴 (검색 / 셋업 추천)
  python main.py recommend           # 신규 추천 엔진 (테마 + 예산)
  python main.py search              # 단일 상품 검색
  python main.py collect             # 상품 수집
  python main.py clean-collect       # 테이블 초기화 후 재수집
  python main.py vectorize           # 임베딩 벡터화
  python main.py vectorize-local     # 로컬 이미지로 벡터화
  python main.py reset-embeddings    # 임베딩 전체 초기화
"""

import sys


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "menu"

    if mode in ("recommend",):
        from deskterior.cli.recommend import main as run
        run()

    elif mode in ("search",):
        from deskterior.cli.search import main as run
        run()

    elif mode in ("menu", "interactive"):
        from deskterior.cli.interactive import main as run
        run()

    elif mode in ("collect", "clean-collect", "vectorize", "vectorize-local", "reset-embeddings", "all"):
        from deskterior.cli.pipeline import main as run
        run()

    else:
        print(f"[오류] 알 수 없는 명령: '{mode}'")
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
