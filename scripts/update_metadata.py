# =============================================================================
# scripts/update_metadata.py
# DB에 적재된 상품의 metadata(사이즈)를 최신 parse_size 로직으로 전체 재적재
# SIZE_RELEVANT 카테고리(MONITOR, KEYBOARD, MOUSE, HEADSET, MOUSEPAD)만 대상
# 사용: python scripts/update_metadata.py
# =============================================================================

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import psycopg2
from psycopg2.extras import Json
from tqdm import tqdm

from deskterior.core.config import DB_CONFIG, SIZE_RELEVANT
from deskterior.database.manager import parse_size


def fetch_targets(conn) -> list[tuple]:
    """SIZE_RELEVANT 카테고리 상품 전체 반환 (기존 값 포함 전부 덮어씀)."""
    placeholders = ",".join(["%s"] * len(SIZE_RELEVANT))
    with conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT id, title, category
            FROM products
            WHERE category IN ({placeholders})
            ORDER BY id
            """,
            list(SIZE_RELEVANT),
        )
        return cur.fetchall()


def main():
    print("[DB] 연결 중...")
    conn = psycopg2.connect(**DB_CONFIG)

    targets = fetch_targets(conn)
    print(f"[대상] metadata 미적재 상품: {len(targets)}개")

    if not targets:
        print("업데이트할 행이 없습니다.")
        conn.close()
        return

    updated = 0
    with conn.cursor() as cur:
        for row_id, title, category in tqdm(targets, desc="metadata 업데이트"):
            size_data = parse_size(title, category)
            cur.execute(
                "UPDATE products SET metadata = %s WHERE id = %s",
                (Json(size_data), row_id),
            )
            updated += 1

    conn.commit()
    conn.close()
    print(f"\n[완료] 업데이트: {updated}행")


if __name__ == "__main__":
    main()
