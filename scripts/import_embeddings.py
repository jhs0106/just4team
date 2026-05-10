# =============================================================================
# scripts/import_embeddings.py
# Colab에서 생성한 npy 파일을  DB에 업데이트
# 사용: python scripts/import_embeddings.py
# =============================================================================

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import psycopg2
from tqdm import tqdm

from core.config import DB_CONFIG, EMBEDDING_DIM

NPY_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "embeddings")
IDS_PATH = os.path.join(NPY_DIR, "product_ids.npy")
IMG_PATH  = os.path.join(NPY_DIR, "image_embeds.npy")
TXT_PATH  = os.path.join(NPY_DIR, "text_embeds.npy")


def vec_to_pg_str(vec) -> str:
    """numpy 1-D 배열을 pgvector 문자열로 변환."""
    return "[" + ",".join(f"{v:.8f}" for v in vec.tolist()) + "]"


def alter_columns(conn, dim: int):
    """embedding_img, embedding_txt 컬럼을 vector(dim) 타입으로 변경."""
    with conn.cursor() as cur:
        for col in ("embedding_img", "embedding_txt"):
            cur.execute(f"ALTER TABLE products ADD COLUMN IF NOT EXISTS {col} vector({dim});")
            # 기존 컬럼이 다른 차원이면 DROP 후 재생성
            cur.execute(f"""
                SELECT atttypmod
                FROM pg_attribute
                JOIN pg_class ON pg_class.oid = pg_attribute.attrelid
                WHERE pg_class.relname = 'products'
                  AND pg_attribute.attname = '{col}'
                  AND pg_attribute.attnum > 0;
            """)
            row = cur.fetchone()
            if row and row[0] != dim + 1:
                print(f"  [{col}] 차원 불일치 (현재 {row[0]-1} → {dim}) → 컬럼 재생성")
                cur.execute(f"ALTER TABLE products DROP COLUMN IF EXISTS {col};")
                cur.execute(f"ALTER TABLE products ADD COLUMN {col} vector({dim});")
    conn.commit()
    print(f"[DB] 컬럼 준비 완료 (vector({dim}))")


def main():
    # ── npy 로드 ──────────────────────────────────────────────────────────────
    print("[로드] npy 파일 읽는 중...")
    ids = np.load(IDS_PATH, allow_pickle=True)   # (N,) string
    img = np.load(IMG_PATH).astype("float32")    # (N, D)
    txt = np.load(TXT_PATH).astype("float32")    # (N, D)

    assert img.ndim == 2 and txt.ndim == 2, "image/text 임베딩은 2D여야 합니다"
    assert img.shape == txt.shape, f"shape 불일치: img={img.shape}, txt={txt.shape}"
    N, D = img.shape
    print(f"  product_ids: {ids.shape}, image: {img.shape}, text: {txt.shape}")
    print(f"  image non-zero: {(np.linalg.norm(img, axis=1) > 0).sum()}/{N}")
    print(f"  text  non-zero: {(np.linalg.norm(txt, axis=1) > 0).sum()}/{N}")

    if D != EMBEDDING_DIM:
        print(f"[경고] npy 차원({D}) ≠ config EMBEDDING_DIM({EMBEDDING_DIM})")


    # ── DB 연결 ───────────────────────────────────────────────────────────────
    print("[DB] 연결 중...")
    conn = psycopg2.connect(**DB_CONFIG)

    alter_columns(conn, D)

    # ── 업데이트 ──────────────────────────────────────────────────────────────
    sql = """
        UPDATE products
        SET embedding_img = %s::vector,
            embedding_txt = %s::vector
        WHERE id = %s
    """
    updated = 0
    skipped = 0

    with conn.cursor() as cur:
        for i, pid in enumerate(tqdm(ids, desc="DB 업데이트")):
            db_id = int(pid)
            img_str = vec_to_pg_str(img[i])
            txt_str = vec_to_pg_str(txt[i])
            cur.execute(sql, (img_str, txt_str, db_id))
            if cur.rowcount == 0:
                skipped += 1
            else:
                updated += 1

        conn.commit()

    conn.close()
    print(f"\n[완료] 업데이트: {updated}행  / 스킵(ID 없음): {skipped}행")


if __name__ == "__main__":
    main()
