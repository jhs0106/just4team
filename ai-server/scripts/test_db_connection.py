# DB 접속 + pgvector + 데이터 검증 테스트.
# 실행: python scripts\test_db_connection.py

import os
import sys
from pathlib import Path
from dotenv import load_dotenv
import psycopg2

REPO_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(REPO_ROOT / ".env")

try:
    conn = psycopg2.connect(
        host=os.getenv("DB_HOST"),
        port=int(os.getenv("DB_PORT", "5432")),
        dbname=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        connect_timeout=5,
    )
    print("[OK] DB 접속 성공")
except Exception as e:
    print(f"[FAIL] DB 접속 실패: {e}")
    sys.exit(1)

with conn.cursor() as cur:
    cur.execute("SELECT version();")
    pg_ver = cur.fetchone()[0]
    print(f"[INFO] PostgreSQL: {pg_ver.split(',')[0]}")

    cur.execute("SELECT extname, extversion FROM pg_extension WHERE extname='vector';")
    row = cur.fetchone()
    if row:
        print(f"[OK] pgvector 활성: {row[1]}")
    else:
        print("[FAIL] pgvector extension 없음")
        sys.exit(1)

    cur.execute("SELECT COUNT(*) FROM products;")
    n_products = cur.fetchone()[0]
    print(f"[OK] products 행 수: {n_products:,}")

    cur.execute("SELECT COUNT(*) FROM products WHERE embedding_img IS NOT NULL;")
    n_img = cur.fetchone()[0]
    print(f"[OK] embedding_img 있는 행: {n_img:,}")

    cur.execute("SELECT COUNT(*) FROM products WHERE embedding_txt IS NOT NULL;")
    n_txt = cur.fetchone()[0]
    print(f"[OK] embedding_txt 있는 행: {n_txt:,}")

    cur.execute("""
        SELECT id, title, category, lprice, metadata->'width_mm' AS w_mm
        FROM products WHERE id = 222;
    """)
    row = cur.fetchone()
    if row:
        print(f"[OK] id=222: {row}")
    else:
        print("[WARN] id=222 없음")

    cur.execute("""
        SELECT category, COUNT(*) AS n
        FROM products
        GROUP BY category
        ORDER BY n DESC
        LIMIT 15;
    """)
    print(f"[INFO] 카테고리별 제품 수 (top 15):")
    for cat, n in cur.fetchall():
        print(f"  {cat or '(NULL)':20} {n:5,}")

conn.close()
print("\n전체 OK. DB 사용 준비 완료.")
