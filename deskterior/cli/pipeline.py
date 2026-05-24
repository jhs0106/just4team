# =============================================================================
# pipeline.py — 전체 프로세스 엔트리 포인트
#
# 실행:
#   python -m deskterior.cli.pipeline                           # 수집 → 벡터화 (전체)
#   python -m deskterior.cli.pipeline collect                   # 수집만
#   python -m deskterior.cli.pipeline clean-collect             # 테이블 완전 초기화(id=1부터) 후 재수집
#   python -m deskterior.cli.pipeline vectorize                 # 배경 제거 후 벡터화
#   python -m deskterior.cli.pipeline vectorize-local images.zip  # 로컬 이미지(zip/폴더)로 벡터화
#   python -m deskterior.cli.pipeline reset-embeddings          # 전체 embedding 컬럼 NULL 초기화
# =============================================================================

import os
import sys
import logging

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def run_clean_collect(db):
    """테이블 데이터 전체 삭제 + id 시퀀스 1로 리셋 후 재수집."""
    with db.conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM products")
        before = cur.fetchone()[0]
        cur.execute("TRUNCATE TABLE products RESTART IDENTITY")
    db.conn.commit()
    print(f"[초기화] 전체 {before}개 삭제 완료, id 시퀀스 1로 리셋")
    run_collect(db)


def run_reset_embeddings(db):
    with db.conn.cursor() as cur:
        cur.execute(
            """
            SELECT COUNT(*)
            FROM products
            WHERE embedding_img IS NOT NULL
               OR embedding_txt IS NOT NULL
            """
        )
        before = cur.fetchone()[0]
        cur.execute(
            "UPDATE products "
            "SET embedding_img = NULL, embedding_txt = NULL"
        )
    db.conn.commit()
    print(f"[초기화] embedding_img/embedding_txt NULL 처리 완료: {before}개 → 0개")


def run_collect(db):
    from deskterior.pipeline.collector import NaverCollector
    from deskterior.core.config import CATEGORY_QUERIES

    default_total = 10
    category_totals = {}

    print("\n[수집 설정] 카테고리별 수집 개수를 입력하세요. (엔터=기본 10, 0=건너뜀)")
    for code, query in CATEGORY_QUERIES.items():
        while True:
            raw = input(f"  - {code} ({query}): ").strip()
            if raw == "":
                category_totals[code] = default_total
                break
            try:
                value = int(raw)
                if value < 0:
                    print("    0 이상의 정수를 입력하세요.")
                    continue
                category_totals[code] = value
                break
            except ValueError:
                print("    숫자만 입력하세요.")

    collector = NaverCollector(db)
    total = collector.collect_all(total=default_total, category_totals=category_totals)
    db.show_summary()
    return total


def run_vectorize(db):
    from deskterior.pipeline.vectorizer import ProductVectorizer
    vectorizer = ProductVectorizer(db)
    updated = vectorizer.vectorize_all()
    return updated


def run_vectorize_local(db):
    from deskterior.pipeline.vectorizer import ProductVectorizer

    default_zip = os.path.join("data", "raw", "processed_images.zip")
    image_source = sys.argv[2] if len(sys.argv) > 2 else default_zip
    vectorizer = ProductVectorizer(db)
    updated = vectorizer.vectorize_from_local_images(image_source)
    return updated


def main():
    from deskterior.database.manager import DBManager

    mode = sys.argv[1] if len(sys.argv) > 1 else "all"

    db = DBManager()
    try:
        if mode in ("collect", "all"):
            db.create_table()
            run_collect(db)

        if mode == "clean-collect":
            db.create_table()
            run_clean_collect(db)

        if mode in ("vectorize", "all"):
            run_vectorize(db)

        if mode == "vectorize-local":
            run_vectorize_local(db)

        if mode == "reset-embeddings":
            run_reset_embeddings(db)

        if mode not in ("collect", "clean-collect", "vectorize", "vectorize-local", "all", "reset-embeddings"):
            print(f"[오류] 알 수 없는 모드: '{mode}'")
            print("사용법: python -m deskterior.cli.pipeline [collect|clean-collect|vectorize|vectorize-local|all|reset-embeddings]")

    finally:
        db.close()


if __name__ == "__main__":
    main()
