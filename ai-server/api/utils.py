import ast
import csv
import base64
from io import BytesIO
from pathlib import Path

from PIL import Image

from .config import (
    _CATEGORY_ALIASES,
    TABLETOP_MIN_HEIGHT_RATIO, TABLETOP_MIN_HEIGHT_PX,
    TABLETOP_MAX_HEIGHT_RATIO, TABLETOP_MIN_BOTTOM_MARGIN_RATIO,
    TABLETOP_MAX_WIDTH_HEIGHT_RATIO,
)

PRODUCT_IMAGE_DIR = Path("data/test/processed_images")
_CATALOG_CSV      = Path("data/test/products.csv")

_product_catalog: dict = {}


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
    global _product_catalog
    if _product_catalog:
        return _product_catalog
    try:
        with open(_CATALOG_CSV, encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                try:
                    image_id = _to_int_or_none(row.get("id") or row.get("image_id"))
                    if image_id is None:
                        continue
                    meta = {}
                    if row.get("metadata"):
                        try:
                            meta = ast.literal_eval(row["metadata"])
                        except Exception:
                            meta = {}
                    width_mm = (
                        _to_int_or_none(row.get("width_mm"))
                        or _to_int_or_none(meta.get("width_mm"))
                    )
                    depth_mm = (
                        _to_int_or_none(row.get("depth_mm"))
                        or _to_int_or_none(meta.get("depth_mm"))
                    )
                    _product_catalog[image_id] = {
                        "category": normalize_category(row.get("category", "")),
                        "title":    row.get("title") or row.get("name") or "",
                        "width_mm": width_mm,
                        "depth_mm": depth_mm,
                    }
                except Exception:
                    continue
        print(f"[Catalog] {len(_product_catalog)}개 제품 로드")
    except FileNotFoundError:
        print(f"[Catalog] {_CATALOG_CSV} 없음 — fallback 치수 사용")
    return _product_catalog


def enrich_products_from_csv(products: list) -> list:
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
