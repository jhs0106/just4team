"""
main.py — 진입점

사용법:
  python main.py           # 테마 + 예산 기반 셋업 추천
  python main.py recommend # 동일
"""

import sys


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "recommend"

    if mode == "recommend":
        from deskterior.cli.recommend import main as run
        run()
    else:
        print(f"[오류] 알 수 없는 명령: '{mode}'")
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
