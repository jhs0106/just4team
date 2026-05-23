import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .config import (
    _CV_CAT_MAX_SCALE, _CATEGORY_DIMS_MM,
    _FRONT_HEIGHT_RATIO, _DESK_W_RATIO,
    _CONTACT_Y_OFFSET,
)
from .utils import normalize_category


def has_meaningful_alpha(img: Image.Image) -> bool:
    if img.mode != "RGBA":
        return False
    alpha = np.array(img.getchannel("A"))
    return alpha.min() < 250


def make_white_bg_transparent(
    img: Image.Image,
    threshold: int = 245,
    feather: int = 4,
) -> Image.Image:
    rgba = img.convert("RGBA")
    arr  = np.array(rgba)
    rgb  = arr[:, :, :3]
    white = (
        (rgb[:, :, 0] >= threshold) &
        (rgb[:, :, 1] >= threshold) &
        (rgb[:, :, 2] >= threshold)
    )
    mask = white.astype(np.uint8) * 255
    h, w = mask.shape
    flood   = mask.copy()
    ff_mask = np.zeros((h + 2, w + 2), np.uint8)
    for sx, sy in [(0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)]:
        if flood[sy, sx] > 0:
            cv2.floodFill(flood, ff_mask, (sx, sy), 128)
    bg    = flood == 128
    alpha = arr[:, :, 3]
    alpha[bg] = 0
    if feather > 0:
        alpha = cv2.GaussianBlur(alpha, (feather * 2 + 1, feather * 2 + 1), 0)
    arr[:, :, 3] = alpha
    return Image.fromarray(arr)


def _tight_crop_rgba(img: Image.Image, padding: int = 4) -> Image.Image:
    alpha = np.array(img.getchannel("A"))
    rows  = np.any(alpha > 10, axis=1)
    cols  = np.any(alpha > 10, axis=0)
    if not rows.any():
        return img
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    h, w = alpha.shape
    rmin = max(0, rmin - padding)
    rmax = min(h - 1, rmax + padding)
    cmin = max(0, cmin - padding)
    cmax = min(w - 1, cmax + padding)
    return img.crop((cmin, rmin, cmax + 1, rmax + 1))


def prepare_product_image_for_composite(img: Image.Image) -> Image.Image:
    rgba   = img.convert("RGBA")
    result = rgba if has_meaningful_alpha(rgba) else make_white_bg_transparent(rgba)
    return _tight_crop_rgba(result)


def composite_product_simple(
    base: Image.Image,
    product_img: Image.Image,
    region: tuple,
    edge_feather: int = 4,
    category: str = "",
) -> Image.Image:
    x1, y1, x2, y2 = region
    target_w = max(1, x2 - x1)
    target_h = max(1, y2 - y1)

    prod    = prepare_product_image_for_composite(product_img)
    _max_sc = _CV_CAT_MAX_SCALE.get(category, 1.0)
    scale   = min(target_w / max(prod.width, 1), target_h / max(prod.height, 1), _max_sc)
    new_w   = max(30, int(prod.width * scale))
    new_h   = max(30, int(prod.height * scale))
    if (new_w, new_h) != (prod.width, prod.height):
        prod = prod.resize((new_w, new_h), Image.Resampling.LANCZOS)

    if edge_feather > 0:
        alpha_arr  = np.array(prod.getchannel("A"))
        ksize      = edge_feather * 2 + 1
        alpha_blur = cv2.GaussianBlur(alpha_arr, (ksize, ksize), 0)
        prod_arr   = np.array(prod)
        prod_arr[:, :, 3] = alpha_blur
        prod = Image.fromarray(prod_arr)

    px  = x1 + (target_w - prod.width) // 2
    py  = y2 - prod.height
    out = base.convert("RGBA")
    out.alpha_composite(prod, (max(0, px), max(0, py)))
    return out.convert("RGB")


def composite_one_with_silhouette(
    base: Image.Image,
    product_img: Image.Image,
    region: tuple,
    edge_feather: int = 4,
    category: str = "",
) -> tuple[Image.Image, Image.Image]:
    # Visual-RAG의 [A] Augmentation (retrieval injection) 핵심 함수.
    # Retrieved 제품 PNG 픽셀을 cleaned_desk에 alpha composite으로 그대로 inject.
    # silhouette은 feather 전 sharp alpha 사용 → Stage 3 strength_map의 seam ring 계산에 쓰여
    # faithfulness guarantee(제품 영역 strength=0)를 정확히 만들어냄.
    x1, y1, x2, y2 = region
    target_w = max(1, x2 - x1)
    target_h = max(1, y2 - y1)

    prod    = prepare_product_image_for_composite(product_img)
    _max_sc = _CV_CAT_MAX_SCALE.get(category, 1.0)
    scale   = min(target_w / max(prod.width, 1), target_h / max(prod.height, 1), _max_sc)
    new_w   = max(30, int(prod.width * scale))
    new_h   = max(30, int(prod.height * scale))
    if (new_w, new_h) != (prod.width, prod.height):
        prod = prod.resize((new_w, new_h), Image.Resampling.LANCZOS)

    alpha_sharp = prod.getchannel("A").copy()

    if edge_feather > 0:
        alpha_arr  = np.array(alpha_sharp)
        ksize      = edge_feather * 2 + 1
        alpha_blur = cv2.GaussianBlur(alpha_arr, (ksize, ksize), 0)
        prod_arr   = np.array(prod)
        prod_arr[:, :, 3] = alpha_blur
        prod = Image.fromarray(prod_arr)

    px       = x1 + (target_w - prod.width) // 2
    py       = y2 - prod.height
    paste_xy = (max(0, px), max(0, py))

    out = base.convert("RGBA")
    out.alpha_composite(prod, paste_xy)

    silhouette = Image.new("L", base.size, 0)
    silhouette.paste(alpha_sharp, paste_xy)

    return out.convert("RGB"), silhouette


def _add_shadows(
    base: Image.Image,
    region: tuple,
    category: str,
    prod_alpha: Image.Image | None = None,
    debug_dir: Path | None = None,
) -> Image.Image:
    x1, y1, x2, y2 = region
    w, h = base.size
    cat  = category.upper()
    pw   = max(1, x2 - x1)
    ph   = max(1, y2 - y1)

    _contact_y  = min(y2, h - 1)
    _contact_x1 = x1
    _contact_x2 = x2

    if prod_alpha is not None and prod_alpha.mode == "RGBA":
        _sc  = min(pw / max(prod_alpha.width, 1), ph / max(prod_alpha.height, 1))
        _sw  = max(1, int(prod_alpha.width  * _sc))
        _sh  = max(1, int(prod_alpha.height * _sc))
        _psc = prod_alpha.resize((_sw, _sh), Image.Resampling.LANCZOS)
        _a   = np.array(_psc.getchannel("A"))
        _px  = x1 + (pw - _sw) // 2

        _rows = np.where((_a > 127).any(axis=1))[0]
        if len(_rows) > 0:
            _obj_bot_local = _rows[-1]
            _contact_y     = min(h - 1, (y2 - _sh) + _obj_bot_local)
            _cols          = np.where(_a[_obj_bot_local] > 127)[0]
            if len(_cols) > 0:
                _contact_x1 = max(0, _px + _cols[0])
                _contact_x2 = min(w, _px + _cols[-1])

    _cx      = (_contact_x1 + _contact_x2) // 2
    _cw      = max(1, _contact_x2 - _contact_x1)
    _cast_ox = max(1, int(pw * 0.12))
    _cast_oy = max(1, int(ph * 0.06))
    _cast_cx = min(w - 1, _cx + _cast_ox)
    _cast_cy = min(h - 1, _contact_y + _cast_oy)

    contact_shadow = np.zeros((h, w), dtype=np.float32)
    cast_shadow    = np.zeros((h, w), dtype=np.float32)

    if cat == "MONITOR":
        _shadow_half_w = max(_cw // 4, pw // 5, 40)
        cv2.ellipse(contact_shadow, (_cx, _contact_y), (_shadow_half_w, 10), 0, 0, 360, 1.0, -1)
        contact_blur_k, contact_str = 13, 0.52
        _cast_hw = max(_shadow_half_w * 2, pw // 3, 60)
        cv2.ellipse(cast_shadow, (_cast_cx, _cast_cy), (_cast_hw, 15), 0, 0, 360, 1.0, -1)
        cast_blur_k, cast_str = 31, 0.15
    elif cat == "KEYBOARD":
        contact_shadow[max(0, _contact_y - 4):min(h, _contact_y + 6),
                       max(0, _contact_x1):min(w, _contact_x2)] = 1.0
        contact_blur_k, contact_str = 11, 0.50
        cast_shadow[max(0, _cast_cy - 5):min(h, _cast_cy + 8),
                    max(0, _contact_x1 + _cast_ox):min(w, _contact_x2 + _cast_ox)] = 1.0
        cast_blur_k, cast_str = 25, 0.14
    elif cat == "MOUSE":
        cv2.ellipse(contact_shadow, (_cx, _contact_y), (max(_cw // 2, 20), 9), 0, 0, 360, 1.0, -1)
        contact_blur_k, contact_str = 9, 0.55
        cv2.ellipse(cast_shadow, (_cast_cx, _cast_cy), (max(_cw, 30), 14), 0, 0, 360, 1.0, -1)
        cast_blur_k, cast_str = 21, 0.15
    elif cat == "DESK_LAMP":
        cv2.ellipse(contact_shadow, (_cx, _contact_y), (max(_cw // 2, 22), 12), 0, 0, 360, 1.0, -1)
        contact_blur_k, contact_str = 13, 0.45
        cv2.ellipse(cast_shadow, (_cast_cx, _cast_cy), (max(_cw, 30), 18), 0, 0, 360, 1.0, -1)
        cast_blur_k, cast_str = 27, 0.13
    else:
        cv2.ellipse(contact_shadow, (_cx, _contact_y), (max(_cw // 3, 18), 9), 0, 0, 360, 1.0, -1)
        contact_blur_k, contact_str = 11, 0.45
        cv2.ellipse(cast_shadow, (_cast_cx, _cast_cy), (max(_cw // 2, 20), 12), 0, 0, 360, 1.0, -1)
        cast_blur_k, cast_str = 19, 0.13

    contact_shadow = cv2.GaussianBlur(contact_shadow, (contact_blur_k, contact_blur_k), 0)
    contact_shadow = np.clip(contact_shadow * contact_str, 0.0, 0.50)
    cast_shadow    = cv2.GaussianBlur(cast_shadow,    (cast_blur_k,    cast_blur_k),    0)
    cast_shadow    = np.clip(cast_shadow    * cast_str,    0.0, 0.30)
    combined       = np.clip(contact_shadow + cast_shadow, 0.0, 0.55)

    if debug_dir is not None:
        try:
            _dd = Path(debug_dir)
            _dd.mkdir(parents=True, exist_ok=True)
            Image.fromarray((contact_shadow * 255).astype(np.uint8)).save(_dd / f"{cat}_contact_shadow_mask.png")
            Image.fromarray((cast_shadow    * 255).astype(np.uint8)).save(_dd / f"{cat}_cast_shadow_mask.png")
            Image.fromarray((combined       * 255).astype(np.uint8)).save(_dd / f"{cat}_combined_shadow_mask.png")
            _shadow_info = {
                "contact_y":              int(_contact_y),
                "contact_x1":             int(_contact_x1),
                "contact_x2":             int(_contact_x2),
                "contact_shadow_opacity": float(contact_str),
                "contact_shadow_blur":    int(contact_blur_k),
                "cast_shadow_opacity":    float(cast_str),
                "cast_shadow_blur":       int(cast_blur_k),
                "cast_shadow_offset":     [int(_cast_ox), int(_cast_oy)],
            }
            _ci_path = _dd / f"{cat}_contact_info.json"
            _existing = json.loads(_ci_path.read_text(encoding="utf-8")) if _ci_path.exists() else {}
            _existing.update(_shadow_info)
            _ci_path.write_text(json.dumps(_existing, indent=2, ensure_ascii=False), encoding="utf-8")
        except Exception:
            pass

    arr = np.array(base.convert("RGB")).astype(np.float32)
    arr = arr * (1.0 - combined[:, :, np.newaxis])
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def _detect_desk_bbox(image: Image.Image) -> tuple | None:
    from .dino_processor import run_grounding_dino
    img_w, img_h = image.size
    scale    = min(512 / img_w, 512 / img_h)
    small    = image.resize((int(img_w * scale), int(img_h * scale)))
    small_bgr = np.array(small.convert("RGB"))[:, :, ::-1].copy()
    dets = run_grounding_dino(small_bgr, "desk. table.", box_threshold=0.20, max_area_ratio=0.98)
    if not dets:
        return None
    best = max(dets, key=lambda d: (d.box_xyxy[2] - d.box_xyxy[0]) * (d.box_xyxy[3] - d.box_xyxy[1]))
    x1, y1, x2, y2 = best.box_xyxy
    return (int(x1 / scale), int(y1 / scale), int(x2 / scale), int(y2 / scale))


def _calc_regions(
    img_w: int, img_h: int, products,
    desk_bbox: tuple | None = None,
    desk_width_mm: int | None = None,
) -> list:
    if desk_bbox:
        dx1, dy1, dx2, dy2 = desk_bbox
    else:
        dx1, dy1 = 0, int(img_h * 0.35)
        dx2, dy2 = img_w, int(img_h * 0.68)
    dy1 = max(dy1, int(img_h * 0.20))
    dy2 = min(dy2, int(img_h * 0.68))

    DW       = dx2 - dx1
    DH       = dy2 - dy1
    cx       = (dx1 + dx2) // 2
    front_y  = dy1 + int(DH * 0.55)
    back_top = dy1

    def clip(x1, y1, x2, y2):
        return max(0, x1), max(0, y1), min(img_w - 1, x2), min(img_h - 1, y2)

    def perspective_scale(center_y: int) -> float:
        return 0.60 + 0.40 * ((center_y - dy1) / max(DH, 1))

    def product_pixel_size(p) -> tuple[int, int]:
        cat     = normalize_category(p.category)
        w_mm    = getattr(p, "width_mm", None) or _CATEGORY_DIMS_MM.get(cat, (100, 100))[0]
        h_ratio = _FRONT_HEIGHT_RATIO.get(cat, 0.80)
        if desk_width_mm and desk_width_mm > 0:
            raw = int(w_mm * DW / desk_width_mm)
        else:
            raw = int(DW * _DESK_W_RATIO.get(cat, 0.15))
        pw = min(raw, int(DW * 0.65))
        return pw, int(pw * h_ratio)

    regions = []
    for p in products:
        cat = normalize_category(p.category)
        if cat not in _CATEGORY_DIMS_MM:
            continue
        base_pw, _ = product_pixel_size(p)

        if cat in ("KEYBOARD", "MOUSE", "MOUSEPAD"):
            ps = perspective_scale(front_y)
            pw = int(base_pw * ps)
            ph = int(pw * _FRONT_HEIGHT_RATIO.get(cat, 0.80))
            if cat == "KEYBOARD":
                x1 = cx - pw // 2;              x2 = x1 + pw
                y2 = front_y;                    y1 = y2 - ph
            elif cat == "MOUSEPAD":
                x1 = cx - pw // 2;              x2 = x1 + pw
                y2 = front_y + int(DH * 0.05);  y1 = y2 - ph
            else:
                x1 = cx + int(DW * 0.22);       x2 = x1 + pw
                y2 = front_y + int(DH * 0.03);  y1 = y2 - ph
        else:
            ps = perspective_scale(back_top + int(DH * 0.15))
            pw = int(base_pw * ps)
            ph = int(pw * _FRONT_HEIGHT_RATIO.get(cat, 0.80))
            if cat == "MONITOR":
                x1 = cx - pw // 2;               x2 = x1 + pw
                y2 = back_top + int(DH * 0.12);  y1 = max(0, y2 - ph)
            elif cat == "SPEAKER":
                x2 = cx - int(DW * 0.20);        x1 = x2 - pw
                y2 = back_top + int(DH * 0.28);  y1 = max(0, y2 - ph)
            elif cat == "DESK_LAMP":
                x1 = cx + int(DW * 0.32);        x2 = x1 + pw
                y2 = back_top + int(DH * 0.22);  y1 = max(0, y2 - ph)
            elif cat == "DESK_SHELF":
                x1 = cx - pw // 2;               x2 = x1 + pw
                y2 = back_top + int(DH * 0.32);  y1 = max(0, y2 - ph)
            elif cat == "LAPTOP_STAND":
                x2 = cx - int(DW * 0.10);        x1 = x2 - pw
                y2 = front_y - int(DH * 0.10);   y1 = y2 - ph
            else:
                x1 = cx + int(DW * 0.36);        x2 = x1 + pw
                y2 = back_top + int(DH * 0.30);  y1 = max(0, y2 - ph)

        x1, y1, x2, y2 = clip(x1, y1, x2, y2)
        if x2 > x1 and y2 > y1:
            regions.append({"product": p, "region": (x1, y1, x2, y2)})

    return regions
