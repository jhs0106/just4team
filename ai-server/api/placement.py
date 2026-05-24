import pickle
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from .config import (
    _CATEGORY_DIMS_MM, _PLACEMENT_ORDER, _PREFERRED_POS,
    _MIN_FRONT_SIZE, _CAT_ASPECT_VALID, _CONTACT_Y_OFFSET,
    _OVERLAP_TOLERANCE, _DEFAULT_OVERLAP_THR,
    _FRONT_HEIGHT_RATIO, _DESK_W_RATIO,
    _DINO_LABEL_TO_CATEGORY, _RANKER_CAT_ID, _RANKER_SKIP_CATS,
    _FRONT_CATS, _BACK_CATS, _REMOVAL_PROMPT, _CAT_RY_RANGE,
)
from .utils import normalize_category
from .composite import _detect_desk_bbox, _calc_regions


def _build_occupied_mask_from_detection(detection: dict, img_w: int, img_h: int) -> np.ndarray:
    occupied = np.zeros((img_h, img_w), dtype=np.uint8)
    for mask_pil in detection.get("individual_masks", []):
        arr = np.array(mask_pil.convert("L"))
        occupied = cv2.bitwise_or(occupied, (arr > 127).astype(np.uint8) * 255)
    return occupied


def _match_products_to_detections(products, detections) -> list:
    available = list(detections)
    matched   = []
    unmatched = []
    for p in products:
        cat   = normalize_category(p.category)
        found = None
        for i, det in enumerate(available):
            det_cat = _DINO_LABEL_TO_CATEGORY.get(det.label.lower(), "")
            if det_cat == cat:
                found = i
                break
        if found is not None:
            det = available.pop(found)
            matched.append({"product": p, "region": det.box_xyxy})
            print(f"[Match] {cat} → DINO '{det.label}' bbox={det.box_xyxy}")
        else:
            unmatched.append(p)
            print(f"[Match] {cat} → 감지된 영역 없음, fallback 사용")
    return matched, unmatched


def _make_rect_mask(img_w: int, img_h: int, x1: int, y1: int, x2: int, y2: int) -> Image.Image:
    mask = Image.new("L", (img_w, img_h), 0)
    ImageDraw.Draw(mask).rectangle([x1, y1, x2, y2], fill=255)
    return mask


def _region_from_detection_center(
    img_w: int, img_h: int,
    det_box: tuple,
    product,
    desk_bbox: tuple | None,
    desk_width_mm: int | None,
) -> tuple:
    x1, y1, x2, y2 = det_box
    cx     = (x1 + x2) // 2
    cy_bot = y2
    cat    = product.category.upper()
    w_mm   = getattr(product, "width_mm", None) or _CATEGORY_DIMS_MM.get(cat, (200, 200))[0]
    if desk_bbox and desk_width_mm:
        DW        = desk_bbox[2] - desk_bbox[0]
        px_per_mm = DW / desk_width_mm
        pw = int(w_mm * px_per_mm)
    else:
        ref_w = _CATEGORY_DIMS_MM.get(cat, (200, 200))[0]
        pw = int(max(x2 - x1, 1) * w_mm / ref_w)
    det_h  = max(y2 - y1, 1)
    ph     = max(det_h * 3, 80)
    pw     = max(pw, 60)
    new_x1 = max(0,     cx - pw // 2)
    new_x2 = min(img_w, new_x1 + pw)
    new_y2 = min(img_h, cy_bot)
    new_y1 = max(0,     new_y2 - ph)
    return new_x1, new_y1, new_x2, new_y2


def _sample_points_in_region(region: dict, n: int = 7) -> list[tuple[float, float]]:
    bx = region["bbox_px"]
    rx1, ry1 = float(bx["x"]), float(bx["y"])
    rx2, ry2 = rx1 + float(bx["width"]), ry1 + float(bx["height"])
    return [
        (rx1 + (rx2 - rx1) * (gi + 0.5) / n,
         ry1 + (ry2 - ry1) * (gj + 0.5) / n)
        for gi in range(n) for gj in range(n)
    ]


def _front_bbox_for_anchor(
    cat: str, rx: float, ry: float, w_mm: int,
    fv_dx1: int, fv_dy1: int, fv_dx2: int, fv_dy2: int,
    fv_dw: int, fv_dh: int, fv_w: int, fv_h: int,
    desk_width_mm: int | None,
    relation_state: dict,
) -> tuple[int, int, int, int] | None:
    # ranker가 선택한 (rx, ry)를 front-view bbox로 변환.
    # ry는 카테고리별 안전 범위(_CAT_RY_RANGE)로 clamp하여 비현실적 배치 방지.
    # 카테고리간 관계 제약(예: KEYBOARD가 MONITOR 아래 와야 함)은 별도 강제.

    # 1. ry를 안전 범위로 clamp
    ry_min, ry_max = _CAT_RY_RANGE.get(cat, (0.10, 0.85))
    ry_clamped     = max(ry_min, min(ry_max, ry))

    # 2. 픽셀 크기 계산 (perspective scale은 clamp된 ry 사용)
    ps = 0.60 + 0.40 * ry_clamped
    fv_pw = max(40, min(int(w_mm * fv_dw / (desk_width_mm or 1200) * ps), int(fv_dw * 0.65)))
    fv_ph = max(20, int(fv_pw * _FRONT_HEIGHT_RATIO.get(cat, 0.80)))
    min_w, min_h = _MIN_FRONT_SIZE.get(cat, (40, 20))
    fv_pw = max(fv_pw, min_w)
    fv_ph = max(fv_ph, min_h)

    # 3. 카테고리별 px 크기 보정 (mm 기반 계산이 너무 작거나 클 때 cap)
    _rx_adj = rx
    if cat == "MONITOR":
        fv_pw = max(min(int(fv_dw * 0.48), 420), 240)
        fv_ph = int(fv_pw * 0.60)
    elif cat == "KEYBOARD":
        fv_pw = max(min(int(fv_dw * 0.38), 360), 220)
        fv_ph = max(int(fv_pw * 0.20), 45)
        if "monitor_rx" in relation_state:
            _rx_adj = relation_state["monitor_rx"]  # 키보드는 모니터 가로축 정렬
    elif cat == "DESK_SHELF":
        fv_pw = max(min(int(fv_dw * 0.40), 320), 150)
        fv_ph = max(int(fv_pw * 0.20), 30)
        if "monitor_rx" in relation_state:
            _rx_adj = relation_state["monitor_rx"]
    elif cat == "DESK_LAMP":
        fv_pw = max(fv_pw, 90)
        fv_ph = max(fv_ph, 130)
        # 책상 중앙 회피 (모니터 가림 방지) — rx만 약간 조정
        _rx_adj = max(rx, 0.12) if rx < 0.5 else min(rx, 0.88)
    elif cat == "SPEAKER":
        fv_pw = max(fv_pw, 60)
        fv_ph = max(fv_ph, 60)
    elif cat == "DECO":
        fv_pw = max(fv_pw, 45)
        fv_ph = max(fv_ph, 45)
    elif cat == "MOUSE":
        fv_pw = max(fv_pw, 60)
        fv_ph = max(int(fv_pw * _FRONT_HEIGHT_RATIO["MOUSE"]), 45)
    elif cat == "MOUSEPAD":
        fv_ph = int(fv_pw * _FRONT_HEIGHT_RATIO["MOUSEPAD"])
    elif cat == "LIGHTING":
        # 모니터 위 가로 라이트바 — 모니터 가로폭과 비슷하게, 매우 얇음
        if "monitor_rx" in relation_state:
            _rx_adj = relation_state["monitor_rx"]
        fv_pw = max(min(int(fv_dw * 0.45), 380), 200)
        fv_ph = max(int(fv_pw * _FRONT_HEIGHT_RATIO.get("LIGHTING", 0.08)), 14)

    # 4. x 좌표 (rx 또는 카테고리 의존성 사용)
    _kb_x2 = relation_state.get("keyboard_front_x2") if cat == "MOUSE" else None
    if _kb_x2 is not None:
        # MOUSE는 KEYBOARD 우측에 붙임 (사용자 손 위치)
        x1 = max(0, _kb_x2 - fv_pw // 4)
        x2 = min(fv_w, x1 + fv_pw)
    else:
        fv_cx = fv_dx1 + _rx_adj * fv_dw
        x1    = max(0, int(fv_cx - fv_pw // 2))
        x2    = min(fv_w, x1 + fv_pw)

    # 5. y 좌표 — clamp된 ry를 base로 사용 (★ 모든 카테고리 통일)
    #    이전엔 카테고리별 하드코딩 % 값으로 덮어써서 ranker 결과 무시됐음
    y2 = min(fv_dy2, max(fv_dy1, int(fv_dy1 + ry_clamped * fv_dh)))
    y1 = max(0, y2 - fv_ph)

    # 6. 카테고리간 관계 강제 (안전판)
    if cat == "MONITOR":
        # 모니터 상단이 책상 위로 충분히 솟아야 함
        y1 = max(int(fv_h * 0.05), y1)
        y2 = y1 + fv_ph
        relation_state["monitor_contact_y"] = y2
    elif cat == "DESK_SHELF":
        # 모니터와 비슷한 깊이
        if "monitor_contact_y" in relation_state:
            _mon_y2 = relation_state["monitor_contact_y"]
            y2 = min(y2, _mon_y2 + int(fv_dh * 0.05))
            y1 = y2 - fv_ph
    elif cat == "KEYBOARD":
        # 모니터 아래로 와야 함 (모니터에 안 가려지게)
        _mon_y2 = relation_state.get("monitor_contact_y", 0)
        if y1 < _mon_y2 + 20:
            y1 = _mon_y2 + 20
            y2 = y1 + fv_ph
        # CONTACT_Y_OFFSET 먼저 적용 (시각적 접지 보정) → 그 다음 cap.
        # 순서 중요: cap 먼저 적용하면 offset이 cap에 막혀 효과 사라짐.
        _offset = _CONTACT_Y_OFFSET.get("KEYBOARD", 0)
        y2 = y2 + _offset
        # 책상 앞 가장자리 cap (마지막)
        y2 = min(y2, int(fv_dy2 - 15))
        y1 = y2 - fv_ph
        relation_state["keyboard_y2"] = y2
        relation_state["keyboard_front_y1"] = y1
    elif cat == "MOUSE":
        # 키보드와 같은 깊이 (책상 면이 같으므로)
        _kb_y2 = relation_state.get("keyboard_y2")
        if _kb_y2 is not None:
            # 키보드 y2 그대로 사용 (이미 offset/cap 적용됨)
            y2 = _kb_y2
            y1 = y2 - fv_ph
            _kb_y1 = relation_state.get("keyboard_front_y1", 0)
            if y1 < _kb_y1:
                y1 = _kb_y1
                y2 = y1 + fv_ph
        else:
            # 키보드 없을 때만 MOUSE 단독 offset 적용
            _offset = _CONTACT_Y_OFFSET.get("MOUSE", 0)
            y2 = min(y2 + _offset, int(fv_dy2 - 15))
            y1 = y2 - fv_ph
    elif cat == "LIGHTING":
        # 모니터 상단에 부착 — monitor_contact_y에서 모니터 높이만큼 위로 가서 위쪽 5~10px
        _mon_y2 = relation_state.get("monitor_contact_y")
        if _mon_y2 is not None:
            # 모니터 본체 위쪽 가장자리 추정 (모니터 높이 ≈ fv_dh × 0.60)
            _mon_top_estimate = max(int(fv_h * 0.05), _mon_y2 - int(fv_dh * 0.60))
            y2 = _mon_top_estimate + max(2, fv_ph // 3)  # 라이트바 하단이 모니터 상단 살짝 가림
            y1 = y2 - fv_ph
        # 화면 상단 안전
        if y1 < int(fv_h * 0.02):
            y1 = int(fv_h * 0.02)
            y2 = y1 + fv_ph

    # 7. 화면 경계 안전화
    y1 = max(0, y1)
    y2 = min(fv_h, y2)

    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)


def bbox_iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1); iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2); iy2 = min(ay2, by2)
    iw  = max(0, ix2 - ix1); ih = max(0, iy2 - iy1)
    inter  = iw * ih
    area_a = max(1, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(1, (bx2 - bx1) * (by2 - by1))
    return inter / (area_a + area_b - inter + 1e-6)


def _overlap_threshold(cat_a: str, cat_b: str) -> float:
    return _OVERLAP_TOLERANCE.get(frozenset({cat_a, cat_b}), _DEFAULT_OVERLAP_THR)


_layout_ranker: dict | None = None


def _get_layout_ranker() -> dict | None:
    global _layout_ranker
    if _layout_ranker is not None:
        return _layout_ranker
    model_path = Path(__file__).parent.parent / "outputs" / "models" / "layout_ranker.pkl"
    if not model_path.exists():
        return None
    try:
        with model_path.open("rb") as f:
            _layout_ranker = pickle.load(f)
        print(f"[LayoutRanker] 로드 완료 type={_layout_ranker['model_type']} AUC={_layout_ranker.get('val_auc')}")
    except Exception as e:
        print(f"[LayoutRanker] 로드 실패: {e}")
    return _layout_ranker


def _make_ranker_feature(
    cat: str, rx: float, ry: float, rw: float, rh: float,
    relation_state: dict, placed_norm: list,
) -> list:
    # 학습된 ranker는 13개 feature 사용 (layout_ranker.pkl의 feature_names 확인됨).
    # 순서: cat_id, rx, ry, rw, rh, dist_to_preferred, edge_margin_x, edge_margin_y,
    #       monitor_rx, monitor_ry, keyboard_rx, keyboard_ry, n_other_objects
    # 추가 feature(dist_mon, is_left, overlap 등)는 모델 재학습 전까지 제외.
    pref         = _PREFERRED_POS.get(cat, {"rx": 0.5, "ry": 0.5})
    dist_pref    = ((rx - pref["rx"])**2 + (ry - pref["ry"])**2) ** 0.5
    edge_x       = min(rx, 1.0 - rx)
    edge_y       = min(ry, 1.0 - ry)
    monitor_rx   = float(relation_state.get("monitor_rx",  -1.0))
    monitor_ry   = float(relation_state.get("monitor_ry",  -1.0))
    keyboard_rx  = float(relation_state.get("keyboard_rx", -1.0))
    keyboard_ry  = float(relation_state.get("keyboard_ry", -1.0))
    n_others     = len([p for p in placed_norm if p.get("cat") != cat])
    return [
        _RANKER_CAT_ID.get(cat, -1),
        round(rx, 4), round(ry, 4), round(rw, 4), round(rh, 4),
        round(dist_pref, 4),
        round(edge_x, 4), round(edge_y, 4),
        round(monitor_rx, 4), round(monitor_ry, 4),
        round(keyboard_rx, 4), round(keyboard_ry, 4),
        n_others,
    ]


def score_region_for_product(
    cat: str,
    rx: float,
    ry: float,
    placed_norm: list,
    relation_state: dict,
    rw: float | None = None,
    rh: float | None = None,
) -> tuple[float, dict]:
    if cat == "KEYBOARD" and "monitor_rx" in relation_state:
        pref_rx = relation_state["monitor_rx"]
        pref_ry = 0.62
    elif cat == "MOUSE":
        pref_rx = min(1.0, relation_state.get("keyboard_rx", 0.50) + 0.20)
        pref_ry = relation_state.get("keyboard_ry", 0.62)
    elif cat == "SPEAKER":
        pref_rx = 0.20 if rx <= 0.50 else 0.80
        pref_ry = relation_state.get("monitor_ry", 0.25)
    else:
        pos = _PREFERRED_POS.get(cat, {"rx": 0.50, "ry": 0.50})
        pref_rx, pref_ry = pos["rx"], pos["ry"]

    dist  = ((rx - pref_rx) ** 2 + (ry - pref_ry) ** 2) ** 0.5
    score = max(0.0, 1.0 - dist * 2.0)

    if cat == "MONITOR" and ry > 0.40:
        score -= 2.0
    if cat == "KEYBOARD" and ry < 0.45:
        score -= 1.5
    if cat == "DESK_LAMP" and 0.20 <= rx <= 0.80:
        score -= 1.5
    if cat in ("DECO", "CLOCK") and 0.30 <= rx <= 0.70 and ry > 0.40:
        score -= 1.0
    if cat == "DECO":
        for _rkey in ("mouse_rx", "keyboard_rx"):
            if _rkey in relation_state:
                _dx = abs(rx - relation_state[_rkey])
                if _dx < 0.15:
                    score -= 2.0
                elif _dx < 0.28:
                    score -= 0.8

    for prev in placed_norm:
        d = ((rx - prev["rx"]) ** 2 + (ry - prev["ry"]) ** 2) ** 0.5
        if d < 0.15:
            score -= 2.0
        elif d < 0.25:
            score -= 0.5

    if min(rx, 1 - rx, ry, 1 - ry) < 0.05:
        score -= 0.3

    rule_score            = score
    learned_score         = None
    ranker_used           = False
    ranker_skipped_reason = None

    if rw is None or rh is None:
        ranker_skipped_reason = "no_rw_rh"
    elif cat in _RANKER_SKIP_CATS:
        ranker_skipped_reason = "skip_cat"
    else:
        _ranker = _get_layout_ranker()
        if _ranker is None:
            ranker_skipped_reason = "no_model"
        else:
            try:
                _feat         = _make_ranker_feature(cat, rx, ry, rw, rh, relation_state, placed_norm)
                _prob         = _ranker["model"].predict_proba([_feat])[0][1]
                learned_score = round((_prob - 0.5) * 3.0, 4)
                score         = rule_score * 0.7 + learned_score * 0.3
                ranker_used   = True
            except Exception:
                ranker_skipped_reason = "error"

    _score_meta = {
        "rule_score":            round(rule_score, 4),
        "learned_layout_score":  learned_score,
        "final_score":           round(score, 4),
        "ranker_used":           ranker_used,
        "ranker_skipped_reason": ranker_skipped_reason,
    }
    return score, _score_meta


def _save_topview_overlays(
    top_view_image: Image.Image,
    available_regions: list,
    placements: list,
    placed_norm: list,
    tv_dx1: int, tv_dy1: int, tv_dw: int, tv_dh: int,
    debug_dir: Path,
) -> None:
    tv_w, tv_h = top_view_image.size

    av_base = top_view_image.copy().convert("RGBA")
    overlay = Image.new("RGBA", (tv_w, tv_h), (0, 0, 0, 0))
    d = ImageDraw.Draw(overlay)
    for r in available_regions:
        bx = r["bbox_px"]
        rx1, ry1 = int(bx["x"]), int(bx["y"])
        rx2, ry2 = rx1 + int(bx["width"]), ry1 + int(bx["height"])
        d.rectangle([rx1, ry1, rx2, ry2], fill=(0, 255, 0, 60), outline=(0, 200, 0, 200))
        d.text((rx1 + 2, ry1 + 2), str(r["region_id"]), fill=(0, 180, 0, 255))
    av_base.alpha_composite(overlay)
    av_base.convert("RGB").save(debug_dir / "top_available_space_overlay.png")

    pl_img = top_view_image.copy()
    d2 = ImageDraw.Draw(pl_img)
    for pn in placed_norm:
        px = int(tv_dx1 + pn["rx"] * tv_dw)
        py = int(tv_dy1 + pn["ry"] * tv_dh)
        d2.ellipse([px - 8, py - 8, px + 8, py + 8], fill=(255, 80, 0))
        d2.text((px + 10, py - 8), pn["cat"], fill=(255, 80, 0))
    pl_img.save(debug_dir / "top_product_placement_debug.png")


def calc_placements_from_available_space(
    front_image: Image.Image,
    top_view_image: Image.Image,
    products: list,
    desk_width_mm: int | None,
    desk_depth_mm: int | None,
    mode,
    remover=None,
    debug_dir: Path | None = None,
) -> list[dict]:
    import json as _dj
    from .space_analysis import (
        analyze_space, make_full_desk_mask, keep_largest_component,
        build_available_mask, draw_available_overlay,
    )
    from .mask_utils import postprocess_occupied_mask
    from .object_removal_processor import get_object_removal_processor

    if remover is None:
        remover = get_object_removal_processor()

    tv_w, tv_h = top_view_image.size
    fv_w, fv_h = front_image.size

    top_det     = remover.detect_with_prompt(
        image=top_view_image, prompt=_REMOVAL_PROMPT, max_area_ratio=0.40,
    )
    occupied_np = _build_occupied_mask_from_detection(top_det, tv_w, tv_h)
    occupied_np = postprocess_occupied_mask(occupied_np)

    desk_mask_np = remover.detect_desk_mask(top_view_image)
    if desk_mask_np is None:
        print("[AvailSpace] desk_mask 실패 → full image")
        desk_mask_np = make_full_desk_mask((tv_h, tv_w))
    else:
        desk_mask_np = keep_largest_component(desk_mask_np)

    desk_width_cm  = (desk_width_mm  / 10) if desk_width_mm  else 120.0
    desk_depth_cm  = (desk_depth_mm  / 10) if desk_depth_mm  else desk_width_cm * 0.55

    # RemoveMode.add → occupied 영역 그대로 유지 (제거 안 함)
    from .models import RemoveMode
    remove_mask = None if mode == RemoveMode.add else occupied_np

    space_info        = analyze_space(
        occupied_mask=occupied_np,
        desk_width_cm=desk_width_cm,
        desk_depth_cm=desk_depth_cm,
        desk_mask=desk_mask_np,
        remove_mask=remove_mask,
    )
    available_regions = space_info.get("available_regions", [])
    cm_per_px         = space_info.get("cm_per_px", {"x": 0.1, "y": 0.1})
    mm_per_px_x       = cm_per_px["x"] * 10  # noqa: F841
    mm_per_px_y       = cm_per_px["y"] * 10  # noqa: F841

    if debug_dir is not None:
        debug_dir = Path(debug_dir)
        debug_dir.mkdir(parents=True, exist_ok=True)

        _occ_bin  = (occupied_np  > 0).astype(np.uint8) * 255
        _desk_bin = (desk_mask_np > 0).astype(np.uint8) * 255
        _avail    = build_available_mask(occupied_np, desk_mask_np, remove_mask)

        if remove_mask is not None:
            _rm_bin   = (remove_mask > 0).astype(np.uint8) * 255
            _kept_occ = cv2.bitwise_and(_occ_bin, cv2.bitwise_not(_rm_bin))
        else:
            _rm_bin   = np.zeros_like(_occ_bin)
            _kept_occ = _occ_bin

        _tv_bgr  = cv2.cvtColor(np.array(top_view_image), cv2.COLOR_RGB2BGR)
        _overlay = draw_available_overlay(_tv_bgr, _avail, desk_mask=_desk_bin)

        Image.fromarray(_desk_bin).save(debug_dir / "desk_mask.png")
        Image.fromarray(_occ_bin).save(debug_dir / "occupied_mask.png")
        Image.fromarray(_rm_bin).save(debug_dir / "remove_mask.png")
        Image.fromarray(_kept_occ).save(debug_dir / "kept_occupied_mask.png")
        Image.fromarray(_avail).save(debug_dir / "available_space_mask.png")
        Image.fromarray(cv2.cvtColor(_overlay, cv2.COLOR_BGR2RGB)).save(debug_dir / "available_overlay.png")

        _space_debug = {
            "remove_mode":            mode.value,
            "desk_area_px":           space_info.get("desk_area_px"),
            "occupied_area_px":       space_info.get("occupied_area_px"),
            "removed_area_px":        int(np.count_nonzero(_rm_bin)),
            "kept_occupied_area_px":  int(np.count_nonzero(_kept_occ)),
            "available_area_px":      space_info.get("available_area_px"),
            "available_area_cm2":     space_info.get("available_area_cm2"),
            "available_region_count": len(available_regions),
        }
        (debug_dir / "space_debug.json").write_text(
            _dj.dumps(_space_debug, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"[AvailSpace] debug masks + space_debug.json 저장: {debug_dir}")

    if not available_regions:
        print("[AvailSpace] 가용 영역 없음 → fallback")
        return []

    ys, xs = np.where(desk_mask_np > 0)
    if len(xs) == 0:
        return []
    tv_dx1, tv_dy1 = int(xs.min()), int(ys.min())
    tv_dx2, tv_dy2 = int(xs.max()), int(ys.max())
    tv_dw = max(1, tv_dx2 - tv_dx1)
    tv_dh = max(1, tv_dy2 - tv_dy1)

    fv_bbox_desk = _detect_desk_bbox(front_image)
    if fv_bbox_desk:
        fv_dx1, fv_dy1, fv_dx2, fv_dy2 = fv_bbox_desk
    else:
        fv_dx1, fv_dy1 = 0, int(fv_h * 0.15)
        fv_dx2, fv_dy2 = fv_w, int(fv_h * 0.72)
    # 책상 표면 끝 근사. 이전(0.68)은 너무 보수적, 그 다음(0.78)은 의자 영역 침범.
    # 0.72는 desk_image2 기준 의자 시작점(0.75~) 직전까지만 허용 → 키보드가 의자 위 안 옴.
    fv_dy2 = min(fv_dy2, int(fv_h * 0.72))
    fv_dw  = max(1, fv_dx2 - fv_dx1)
    fv_dh  = max(1, fv_dy2 - fv_dy1)

    sorted_products = sorted(
        [p for p in products if normalize_category(p.category) in _CATEGORY_DIMS_MM],
        key=lambda p: _PLACEMENT_ORDER.get(normalize_category(p.category), 999),
    )
    for p in products:
        if normalize_category(p.category) not in _CATEGORY_DIMS_MM:
            print(f"[Placement SKIP] unsupported: {p.category}")

    relation_state:     dict = {}
    placed_norm:        list = []
    placed_front_items: list = []
    placements:         list = []

    for p in sorted_products:
        cat  = normalize_category(p.category)
        w_mm = getattr(p, "width_mm", None) or _CATEGORY_DIMS_MM[cat][0]

        best_score       = -999.0
        best_cand        = None
        best_score_meta  = None
        candidate_count  = 0
        overlap_reject   = 0
        low_score_reject = 0

        for region in available_regions:
            for anchor_x, anchor_y in _sample_points_in_region(region, n=7):
                rx = max(0.0, min(1.0, (anchor_x - tv_dx1) / tv_dw))
                ry = max(0.0, min(1.0, (anchor_y - tv_dy1) / tv_dh))
                fv_bbox = _front_bbox_for_anchor(
                    cat, rx, ry, w_mm,
                    fv_dx1, fv_dy1, fv_dx2, fv_dy2, fv_dw, fv_dh, fv_w, fv_h,
                    desk_width_mm, relation_state,
                )
                if fv_bbox is None:
                    continue
                _rej = any(
                    bbox_iou(fv_bbox, prev["region"]) >= _overlap_threshold(cat, prev["cat"])
                    for prev in placed_front_items
                )
                if _rej:
                    overlap_reject += 1
                    continue
                candidate_count += 1
                _rw = (fv_bbox[2] - fv_bbox[0]) / max(fv_dw, 1)
                _rh = (fv_bbox[3] - fv_bbox[1]) / max(fv_dh, 1)
                s, _smeta = score_region_for_product(cat, rx, ry, placed_norm, relation_state, rw=_rw, rh=_rh)
                if s < -0.5:
                    low_score_reject += 1
                    continue
                if s > best_score:
                    best_score      = s
                    best_score_meta = _smeta
                    best_cand       = {"region": fv_bbox, "rx": rx, "ry": ry,
                                       "region_id": region["region_id"]}

        # KEYBOARD 강제 후보: scoring 실패 시 monitor 하단에 강제 배치
        if cat == "KEYBOARD" and best_cand is None:
            _forced_rx = relation_state.get("monitor_rx", 0.50)
            for _fry in [0.65, 0.70, 0.60, 0.75, 0.55]:
                _fb = _front_bbox_for_anchor(
                    cat, _forced_rx, _fry, w_mm,
                    fv_dx1, fv_dy1, fv_dx2, fv_dy2, fv_dw, fv_dh, fv_w, fv_h,
                    desk_width_mm, relation_state,
                )
                if _fb is None:
                    continue
                _non_critical = [i for i in placed_front_items
                                 if i["cat"] not in {"MONITOR", "DESK_SHELF"}]
                if any(bbox_iou(_fb, i["region"]) >= _DEFAULT_OVERLAP_THR for i in _non_critical):
                    continue
                _rw_fb = (_fb[2] - _fb[0]) / max(fv_dw, 1)
                _rh_fb = (_fb[3] - _fb[1]) / max(fv_dh, 1)
                _fs, best_score_meta = score_region_for_product(cat, _forced_rx, _fry, placed_norm, relation_state, rw=_rw_fb, rh=_rh_fb)
                best_score = max(_fs, -0.49)
                best_cand  = {"region": _fb, "rx": _forced_rx, "ry": _fry, "region_id": -1}
                print(f"  [KEYBOARD forced] rx={_forced_rx:.2f} ry={_fry:.2f} → {_fb}")
                break

        if best_cand:
            x1, y1, x2, y2 = best_cand["region"]
            _src = ("forced_keyboard" if best_cand["region_id"] == -1
                    else "available_space_scoring")
            _sel_reason = (
                "forced_keyboard_below_monitor" if _src == "forced_keyboard"
                else "high_score"               if best_score > 0.5
                else "positive_score"           if best_score > 0
                else "zero_score_best_available" if best_score >= -0.01
                else "low_score_best_available"
            )
            print(f"  [Score] {cat} rx={best_cand['rx']:.2f} ry={best_cand['ry']:.2f} "
                  f"score={best_score:.2f} region_id={best_cand['region_id']} "
                  f"cand={candidate_count} ov_rej={overlap_reject} ls_rej={low_score_reject} "
                  f"→ fv({x1},{y1},{x2},{y2})")
            placements.append({
                "product":               p,
                "region":                (x1, y1, x2, y2),
                "score":                 round(best_score, 3),
                "score_meta":            best_score_meta,
                "available_region_id":   best_cand["region_id"],
                "placement_source":      _src,
                "selected_reason":       _sel_reason,
                "candidate_count":       candidate_count,
                "overlap_reject_count":  overlap_reject,
                "low_score_reject_count": low_score_reject,
                "fallback_reason":       None,
                "anchor_rx":             round(best_cand["rx"], 3),
                "anchor_ry":             round(best_cand["ry"], 3),
            })
            placed_front_items.append({"region": (x1, y1, x2, y2), "cat": cat})
            relation_state[f"{cat.lower()}_rx"] = best_cand["rx"]
            relation_state[f"{cat.lower()}_ry"] = best_cand["ry"]
            if cat == "KEYBOARD":
                relation_state["keyboard_front_x2"] = x2
                relation_state["keyboard_front_y1"] = y1
            placed_norm.append({
                "rx":  best_cand["rx"], "ry": best_cand["ry"],
                "rw":  (x2 - x1) / max(fv_dw, 1), "rh": (y2 - y1) / max(fv_dh, 1),
                "cat": cat,
            })
        else:
            reason = ("no_non_overlapping_candidate" if candidate_count == 0
                      else "all_candidates_low_score")
            print(f"  [Score FAIL] {cat} cand={candidate_count} "
                  f"ov_rej={overlap_reject} ls_rej={low_score_reject} → {reason}")
            placements.append({
                "product":               p,
                "region":                None,
                "score":                 None,
                "available_region_id":   None,
                "placement_source":      "fallback",
                "selected_reason":       None,
                "candidate_count":       candidate_count,
                "overlap_reject_count":  overlap_reject,
                "low_score_reject_count": low_score_reject,
                "fallback_reason":       reason,
                "anchor_rx":             None,
                "anchor_ry":             None,
            })

    if debug_dir is not None:
        try:
            _save_topview_overlays(
                top_view_image=top_view_image,
                available_regions=available_regions,
                placements=[i for i in placements if i.get("region") is not None],
                placed_norm=placed_norm,
                tv_dx1=tv_dx1, tv_dy1=tv_dy1, tv_dw=tv_dw, tv_dh=tv_dh,
                debug_dir=debug_dir,
            )
        except Exception as _e:
            print(f"[AvailSpace] top-view overlay 저장 실패: {_e}")

    return placements
