import base64
import json
import os
from io import BytesIO
from pathlib import Path

import psycopg2
from PIL import Image

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from .config import (
    _CATEGORY_ALIASES,
    TABLETOP_MIN_HEIGHT_RATIO, TABLETOP_MIN_HEIGHT_PX,
    TABLETOP_MAX_HEIGHT_RATIO, TABLETOP_MIN_BOTTOM_MARGIN_RATIO,
    TABLETOP_MAX_WIDTH_HEIGHT_RATIO,
)

PRODUCT_IMAGE_DIR = Path("data/test/processed_images")

_product_catalog: dict = {}


def _get_db_connection():
    # .env의 DB_* 변수로 PostgreSQL 연결. recommendation과 동일 DB (products 테이블 공유).
    return psycopg2.connect(
        host=os.getenv("DB_HOST"),
        port=int(os.getenv("DB_PORT", "5432")),
        dbname=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        connect_timeout=5,
    )


def b64_to_image(b64: str) -> Image.Image:
    return Image.open(BytesIO(base64.b64decode(b64))).convert("RGB")


def image_to_b64(image: Image.Image) -> str:
    buf = BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def validate_tabletop_bbox(
    bbox: tuple, image_w: int, image_h: int,
) -> tuple[bool, str, dict]:
    # front-view tabletop bbox sanity check.
    # top-view anchor를 front-view에 투영하는 기준 영역이므로 검증 실패 시 placement 강행 금지.
    # returns (is_valid, reason, meta_dict).
    x1, y1, x2, y2 = bbox
    w = max(0, x2 - x1)
    h = max(0, y2 - y1)
    h_ratio = h / max(image_h, 1)
    w_h     = w / max(h, 1)
    bottom_margin_ratio = (image_h - y2) / max(image_h, 1)

    meta = {
        "fv_dh":               h,
        "fv_dw":               w,
        "fv_dh_ratio":         round(h_ratio, 4),
        "fv_dw_dh_ratio":      round(w_h, 3),
        "bottom_margin_ratio": round(bottom_margin_ratio, 4),
    }

    if h < TABLETOP_MIN_HEIGHT_PX:
        return False, f"too_thin_abs:{h}px<{TABLETOP_MIN_HEIGHT_PX}", meta
    if h_ratio < TABLETOP_MIN_HEIGHT_RATIO:
        return False, f"too_thin_ratio:{h_ratio:.3f}<{TABLETOP_MIN_HEIGHT_RATIO}", meta
    if h_ratio > TABLETOP_MAX_HEIGHT_RATIO:
        return False, f"too_thick_ratio:{h_ratio:.3f}>{TABLETOP_MAX_HEIGHT_RATIO}", meta
    if bottom_margin_ratio < TABLETOP_MIN_BOTTOM_MARGIN_RATIO:
        return False, f"too_close_to_bottom:{bottom_margin_ratio:.3f}<{TABLETOP_MIN_BOTTOM_MARGIN_RATIO}", meta
    if w_h > TABLETOP_MAX_WIDTH_HEIGHT_RATIO:
        return False, f"too_wide_aspect:{w_h:.2f}>{TABLETOP_MAX_WIDTH_HEIGHT_RATIO}", meta

    return True, "valid", meta


def normalize_category(category: str) -> str:
    key = category.strip().upper().replace("-", "_")
    key_space = key.replace("_", " ")
    return _CATEGORY_ALIASES.get(key, _CATEGORY_ALIASES.get(key_space, key))


def find_product_image(image_id: int) -> Path | None:
    for ext in [".png", ".jpg", ".jpeg", ".webp"]:
        path = PRODUCT_IMAGE_DIR / f"{image_id}{ext}"
        if path.exists():
            return path
    return None


def _to_int_or_none(v):
    try:
        if v is None or v == "":
            return None
        return int(float(v))
    except Exception:
        return None


def load_product_catalog() -> dict:
    # DB의 products 테이블 lazy load → image_id → {category, title, width_mm, depth_mm} dict
    global _product_catalog
    if _product_catalog:
        return _product_catalog
    try:
        conn = _get_db_connection()
        with conn.cursor() as cur:
            cur.execute("SELECT id, title, category, metadata FROM products WHERE id IS NOT NULL")
            for pid, title, category, metadata in cur.fetchall():
                meta = metadata or {}
                if isinstance(meta, str):
                    try:
                        meta = json.loads(meta)
                    except Exception:
                        meta = {}
                width_mm = _to_int_or_none(meta.get("width_mm"))
                depth_mm = _to_int_or_none(meta.get("depth_mm")) or _to_int_or_none(meta.get("height_mm"))
                _product_catalog[int(pid)] = {
                    "category": normalize_category(category or ""),
                    "title":    title or "",
                    "width_mm": width_mm,
                    "depth_mm": depth_mm,
                }
        conn.close()
        print(f"[Catalog] {len(_product_catalog)}개 제품 로드 (DB)")
    except Exception as e:
        print(f"[Catalog] DB 로드 실패: {e} — fallback 치수 사용")
    return _product_catalog


def enrich_products_from_db(products: list) -> list:
    catalog = load_product_catalog()
    enriched = []
    for p in products:
        updates: dict = {}
        norm_cat = normalize_category(p.category)
        if norm_cat != p.category:
            updates["category"] = norm_cat
        if p.image_id is not None and p.image_id in catalog:
            meta = catalog[p.image_id]
            if not updates.get("category") and meta["category"]:
                updates["category"] = meta["category"]
            if p.width_mm is None and meta["width_mm"]:
                updates["width_mm"] = meta["width_mm"]
            if p.depth_mm is None and meta["depth_mm"]:
                updates["depth_mm"] = meta["depth_mm"]
        if updates:
            p = p.model_copy(update=updates)
        enriched.append(p)
    return enriched
