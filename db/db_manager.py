# =============================================================================
# db_manager.py — DB 연결, 테이블 관리, UPSERT, 유틸리티 함수
# =============================================================================

import re
import logging
from html import unescape
import psycopg2
from psycopg2.extras import Json

from core.config import DB_CONFIG, DEFAULT_SIZES, SIZE_RELEVANT, EMBEDDING_DIM

logger = logging.getLogger(__name__)


# ── 유틸리티 ──────────────────────────────────────────────────────────────────

def strip_html(text: str) -> str:
    """HTML 태그 제거."""
    return re.sub(r"<[^>]+>", "", text or "")


def parse_size(title: str, category: str):
    """DESK/MONITOR/DESK_SHELF만 사이즈 파싱. 나머지는 None 반환."""
    if category not in SIZE_RELEVANT:
        return None
    match = re.search(r"(\d{2,4})\s*[xX×*]\s*(\d{2,4})", title)
    if match:
        w, d = int(match.group(1)), int(match.group(2))
        if w < 100:
            w *= 10
        if d < 100:
            d *= 10
        return {"width_mm": w, "depth_mm": d, "source": "parsed"}
    defaults = DEFAULT_SIZES.get(category, {})
    return {
        "width_mm": defaults.get("width_mm"),
        "depth_mm": defaults.get("depth_mm"),
        "source": "default",
    }


# ── DB 매니저 클래스 ──────────────────────────────────────────────────────────

class DBManager:
    """PostgreSQL 연결 및 products 테이블 관리."""

    def __init__(self):
        self.conn = psycopg2.connect(**DB_CONFIG)
        logger.info("[DB] 연결 완료 (%s:%s/%s)", DB_CONFIG["host"], DB_CONFIG["port"], DB_CONFIG["dbname"])

    def close(self):
        if self.conn and not self.conn.closed:
            self.conn.close()
            logger.info("[DB] 연결 종료")

    # ── 테이블 초기화 ─────────────────────────────────────────────────────────

    def create_table(self):
        """테이블 생성. 스키마 변경 시 기존 테이블 유지."""
        with self.conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS products (
                    id           SERIAL PRIMARY KEY,
                    title        TEXT,
                    link         TEXT,
                    image        TEXT,
                    lprice       INTEGER,
                    hprice       TEXT,
                    mall_name    TEXT,
                    product_id   TEXT UNIQUE,
                    product_type TEXT,
                    brand        TEXT,
                    maker        TEXT,
                    category1    TEXT,
                    category2    TEXT,
                    category3    TEXT,
                    category4    TEXT,
                    category     TEXT,
                    embedding_img vector({EMBEDDING_DIM}),
                    embedding_txt vector({EMBEDDING_DIM}),
                    metadata     JSONB
                );
            """)
            # 기존 테이블에도 분리 임베딩 컬럼을 안전하게 추가
            cur.execute(f"ALTER TABLE products ADD COLUMN IF NOT EXISTS embedding_img vector({EMBEDDING_DIM});")
            cur.execute(f"ALTER TABLE products ADD COLUMN IF NOT EXISTS embedding_txt vector({EMBEDDING_DIM});")
        self.conn.commit()
        logger.info("[DB] 테이블 생성 완료")
        print("[초기화] 테이블 준비 완료")

    def ensure_split_embedding_columns(self):
        """기존 DB에 분리 임베딩 컬럼이 없으면 추가."""
        with self.conn.cursor() as cur:
            cur.execute(f"ALTER TABLE products ADD COLUMN IF NOT EXISTS embedding_img vector({EMBEDDING_DIM});")
            cur.execute(f"ALTER TABLE products ADD COLUMN IF NOT EXISTS embedding_txt vector({EMBEDDING_DIM});")
        self.conn.commit()

    # ── UPSERT ────────────────────────────────────────────────────────────────

    def upsert_products(self, products: list, category_code: str) -> int:
        """상품 리스트를 UPSERT (product_id 기준 중복 시 업데이트)."""
        sql = """
            INSERT INTO products (
                title, link, image, lprice, hprice, mall_name,
                product_id, product_type, brand, maker,
                category1, category2, category3, category4,
                category, metadata
            ) VALUES (
                %(title)s, %(link)s, %(image)s, %(lprice)s, %(hprice)s, %(mall_name)s,
                %(product_id)s, %(product_type)s, %(brand)s, %(maker)s,
                %(category1)s, %(category2)s, %(category3)s, %(category4)s,
                %(category)s, %(metadata)s
            )
            ON CONFLICT (product_id) DO UPDATE SET
                title        = EXCLUDED.title,
                link         = EXCLUDED.link,
                image        = EXCLUDED.image,
                lprice       = EXCLUDED.lprice,
                hprice       = EXCLUDED.hprice,
                mall_name    = EXCLUDED.mall_name,
                product_type = EXCLUDED.product_type,
                brand        = EXCLUDED.brand,
                maker        = EXCLUDED.maker,
                category1    = EXCLUDED.category1,
                category2    = EXCLUDED.category2,
                category3    = EXCLUDED.category3,
                category4    = EXCLUDED.category4,
                category     = EXCLUDED.category,
                metadata     = EXCLUDED.metadata;
        """
        with self.conn.cursor() as cur:
            for p in products:
                title = strip_html(p.get("title", ""))
                try:
                    lprice = int(p.get("lprice") or 0)
                except (ValueError, TypeError):
                    lprice = 0

                cur.execute(sql, {
                    "title":        title,
                    "link":         unescape(p.get("link") or ""),
                    "image":        unescape(p.get("image") or ""),
                    "lprice":       lprice,
                    "hprice":       p.get("hprice"),
                    "mall_name":    p.get("mallName"),
                    "product_id":   p.get("productId"),
                    "product_type": p.get("productType"),
                    "brand":        p.get("brand"),
                    "maker":        p.get("maker"),
                    "category1":    p.get("category1"),
                    "category2":    p.get("category2"),
                    "category3":    p.get("category3") or "",
                    "category4":    p.get("category4") or "",
                    "category":     category_code,
                    "metadata":     Json(parse_size(title, category_code)),
                })
        self.conn.commit()
        return len(products)

    # ── 조회 ──────────────────────────────────────────────────────────────────

    def get_products_missing_split_embeddings(self) -> list:
        """분리 임베딩(이미지/텍스트)이 비어 있고 image URL이 있는 상품 목록 반환."""
        with self.conn.cursor() as cur:
            cur.execute(
                "SELECT id, image, category, title FROM products "
                "WHERE (embedding_img IS NULL OR embedding_txt IS NULL) "
                "AND image IS NOT NULL"
            )
            return cur.fetchall()

    def update_split_embeddings(
        self,
        product_id: int,
        img_vec_str: str,
        txt_vec_str: str,
    ):
        """단일 상품의 이미지/텍스트 임베딩을 업데이트."""
        try:
            with self.conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE products
                    SET embedding_img = %s::vector,
                        embedding_txt = %s::vector
                    WHERE id = %s
                    """,
                    (img_vec_str, txt_vec_str, product_id),
                )
            self.conn.commit()
        except Exception:
            self.conn.rollback()
            raise

    def count_products_by_category(self, category_code: str) -> int:
        """지정 카테고리의 현재 상품 수를 반환."""
        with self.conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM products WHERE category = %s", (category_code,))
            return int(cur.fetchone()[0])

    # ── 요약 출력 ─────────────────────────────────────────────────────────────

    def show_summary(self):
        """카테고리별 저장 현황과 샘플 5개를 출력."""
        with self.conn.cursor() as cur:
            cur.execute("""
                SELECT category, COUNT(*) AS cnt
                FROM products
                GROUP BY category
                ORDER BY cnt DESC;
            """)
            rows = cur.fetchall()

        print("\n[결과] 카테고리별 저장 현황")
        print("=" * 40)
        total = 0
        for row in rows:
            print(f"  {row[0]:<12}: {row[1]}개")
            total += row[1]
        print(f"  {'합계':<12}: {total}개")
        print("=" * 40)

        with self.conn.cursor() as cur:
            cur.execute("""
                SELECT title, lprice, category, brand, metadata
                FROM products
                ORDER BY id
                LIMIT 5;
            """)
            samples = cur.fetchall()

        print("\n[샘플] 저장된 데이터 예시 (5개)")
        print("=" * 60)
        for i, row in enumerate(samples, 1):
            print(f"\n  #{i}")
            print(f"  title   : {row[0]}")
            print(f"  lprice  : {row[1]}원")
            print(f"  category: {row[2]}")
            print(f"  brand   : {row[3]}")
            print(f"  metadata: {row[4]}")
        print("=" * 60)


if __name__ == "__main__":
    db_manager = DBManager()
    db_manager.create_table()
    db_manager.close()