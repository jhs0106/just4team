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
    _PRODUCT_FORM_TIER,
    TABLETOP_BBOX_HARD_CLAMP_ENABLED,
    RANKER_WEIGHT,
)
from .utils import normalize_category, validate_tabletop_bbox
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
    d_mm: int | None = None,           # 제품 depth (mm, 책상 안쪽 방향). flat 카테고리 fv_ph 계산용.
    desk_depth_mm: int | None = None,  # 책상 depth (mm, 사용자 입력). flat 카테고리 perspective용.
) -> tuple[int, int, int, int] | None:
    # ranker가 선택한 (rx, ry)를 front-view bbox로 변환.
    # ry는 카테고리별 안전 범위(_CAT_RY_RANGE)로 clamp하여 비현실적 배치 방지.
    # 카테고리간 관계 제약(예: KEYBOARD가 MONITOR 아래 와야 함)은 별도 강제.

    # 1. ry를 안전 범위로 clamp
    ry_min, ry_max = _CAT_RY_RANGE.get(cat, (0.10, 0.85))
    ry_clamped     = max(ry_min, min(ry_max, ry))

    # 2. 픽셀 크기 계산 — 제품 width_mm/depth_mm를 책상 가로 px_per_mm로 그대로 변환
    # desk_width_mm은 main.py에서 None 체크 후 도착 → 여기 도달 시 항상 정수.
    # 안전판으로만 1mm 최소값 (0 division 방지).
    ps = 0.60 + 0.40 * ry_clamped
    _desk_w_mm_safe = max(1, int(desk_width_mm)) if desk_width_mm else 1
    _px_per_mm = fv_dw / _desk_w_mm_safe
    fv_pw = max(40, min(int(w_mm * _px_per_mm * ps), int(fv_dw * 0.65)))
    if d_mm:
        fv_ph = max(20, int(int(d_mm) * _px_per_mm * ps))
    else:
        # depth_mm 정보가 없으면 정사각 가정
        fv_ph = max(20, int(fv_pw * 0.5))

    min_w, min_h = _MIN_FRONT_SIZE.get(cat, (40, 20))
    fv_pw = max(fv_pw, min_w)
    fv_ph = max(fv_ph, min_h)

    # 3. 카테고리별 px 크기 보정 (mm 기반 계산이 너무 작거나 클 때 cap)
    _rx_adj = rx
    if cat == "MONITOR":
        # 옛 fv_dw*0.48 강제 제거 — 위에서 계산한 실제 width_mm 기반 fv_pw 사용.
        fv_ph = int(fv_pw * 0.60)
    elif cat == "KEYBOARD":
        # w_mm 기반 자연 크기. 옛 hardcoded 0.38 강제 → 작은 키보드(Apple Magic 등)가
        # 비대해지고 horizontal stretch까지 발동해서 세로로 짜부러져 보임.
        if w_mm and desk_width_mm:
            _natural_pw = int(w_mm * fv_dw / desk_width_mm)
            fv_pw = max(140, min(_natural_pw, int(fv_dw * 0.50)))
        else:
            fv_pw = max(min(int(fv_dw * 0.30), 280), 180)
        # fv_ph는 위 perspective 계산 결과 유지 (옛 magic 0.28 제거).
        if "monitor_rx" in relation_state:
            _rx_adj = relation_state["monitor_rx"]  # 키보드는 모니터 가로축 정렬
    elif cat == "DESK_SHELF":
        # 옛 fv_dw*0.40 강제 제거 — 실제 width_mm 기반 fv_pw 사용.
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
        # fv_ph는 perspective 계산 결과 유지 (옛 magic 0.75 제거).
    elif cat == "MOUSEPAD":
        # fv_ph는 perspective 계산 결과 유지 (옛 magic 0.25 제거).
        pass
    elif cat == "LIGHTING":
        # 모니터 위 가로 라이트바 — 매우 얇음. 옛 fv_dw*0.45 강제 제거, 실제 width_mm 기반.
        if "monitor_rx" in relation_state:
            _rx_adj = relation_state["monitor_rx"]
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
        # 모니터 스탠드 베이스(y2)는 책상 back edge에 anchor되어야 함.
        # 이전 버그: y1 = max(fv_h*0.05, y1) 후 y2 = y1+fv_ph로 재계산 → y2가
        #   책상 back에서 분리되어 책상 중간/허공에 떠보이는 현상 발생.
        # 수정: y2를 책상 back ~ ry 위치에 직접 anchor, y1은 화면 밖이어도 OK
        #   (큰 모니터일수록 화면 top 밖으로 일부 잘리는 것은 자연스러움).
        y2 = max(fv_dy1, min(fv_dy2, int(fv_dy1 + ry_clamped * fv_dh)))
        y1 = max(-fv_ph // 2, y2 - fv_ph)
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
        # mousepad_region 있으면 MOUSE를 그 안쪽에 강제 배치 (마우스패드 위에 마우스).
        # x/y 좌표 모두 mousepad 영역으로 clamp — score/ranker 결과 무시하고 강제.
        _mp_region = relation_state.get("mousepad_region")
        if _mp_region is not None:
            _mp_x1, _mp_y1, _mp_x2, _mp_y2 = _mp_region
            _mp_w, _mp_h = _mp_x2 - _mp_x1, _mp_y2 - _mp_y1
            # MOUSE 폭이 mousepad 폭의 40% 넘으면 강제로 줄임 (시각적 비례)
            if fv_pw > int(_mp_w * 0.40):
                fv_pw = max(int(_mp_w * 0.40), 50)
                fv_ph = max(int(fv_pw * _FRONT_HEIGHT_RATIO["MOUSE"]), 40)
            # x: mousepad 가로 중앙에서 약간 우측 (오른손잡이 가정)
            _target_cx = _mp_x1 + int(_mp_w * 0.60)
            x1 = max(_mp_x1, _target_cx - fv_pw // 2)
            x2 = min(_mp_x2, x1 + fv_pw)
            # y: mousepad 하단에 마우스 발 닿게
            y2 = _mp_y2
            y1 = max(_mp_y1, y2 - fv_ph)
        else:
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
                # 키보드도 mousepad도 없을 때만 MOUSE 단독 offset 적용
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


def _front_bbox_from_corners(
    cat: str, rx: float, ry: float, w_mm: int, d_mm: int | None,
    desk_corners,                  # 4x2 TL,TR,BR,BL (front px) — 사용자 클릭 책상 윗면
    desk_width_mm: int | None, desk_depth_mm: int | None,
    relation_state: dict,
) -> tuple[int, int, int, int] | None:
    # 사용자 클릭 4점으로 책상평면(0~1) → front 픽셀 perspective 매핑.
    # flat은 footprint를 책상면에 깔고, upright는 footprint 앞모서리 접지 + 위로 높이.
    ry_min, ry_max = _CAT_RY_RANGE.get(cat, (0.10, 0.85))
    ry_c = max(ry_min, min(ry_max, ry))
    rx_adj = rx
    if cat in ("KEYBOARD", "DESK_SHELF", "LIGHTING") and "monitor_rx" in relation_state:
        rx_adj = relation_state["monitor_rx"]

    corners = np.asarray(desk_corners, dtype=np.float32)
    norm = np.float32([[0, 0], [1, 0], [1, 1], [0, 1]])
    M = cv2.getPerspectiveTransform(norm, corners)

    _dim = _CATEGORY_DIMS_MM.get(cat, (100, 100))
    dw_mm = max(1, int(desk_width_mm)) if desk_width_mm else 1
    dd_mm = max(1, int(desk_depth_mm)) if desk_depth_mm else max(1, int(dw_mm * 0.55))
    rw = min(0.95, (w_mm or _dim[0]) / dw_mm)
    rd = min(0.60, (d_mm or _dim[1]) / dd_mm)

    # footprint 4점 (앞 ry_c, 뒤 ry_c-rd) → front
    yb, yt = ry_c, max(0.0, ry_c - rd)
    xl, xr = max(0.0, rx_adj - rw / 2), min(1.0, rx_adj + rw / 2)
    fp  = np.float32([[xl, yt], [xr, yt], [xr, yb], [xl, yb]]).reshape(-1, 1, 2)
    fpf = cv2.perspectiveTransform(fp, M).reshape(-1, 2)

    tier = _PRODUCT_FORM_TIER.get(cat, "upright")
    if tier == "flat":
        x1, x2 = float(fpf[:, 0].min()), float(fpf[:, 0].max())
        y1, y2 = float(fpf[:, 1].min()), float(fpf[:, 1].max())
    else:
        bl, br = fpf[3], fpf[2]
        foot_w = max(1.0, abs(br[0] - bl[0]))
        h_px   = foot_w * _FRONT_HEIGHT_RATIO.get(cat, 0.60)
        x1, x2 = float(min(bl[0], br[0])), float(max(bl[0], br[0]))
        y2     = float(max(bl[1], br[1]))
        y1     = y2 - h_px

    # 카테고리 관계 (모니터 접지 기록 → 키보드는 그 아래로)
    if cat == "MONITOR":
        relation_state["monitor_contact_y"] = y2
    elif cat == "KEYBOARD":
        _mon_y2 = relation_state.get("monitor_contact_y", 0)
        if y1 < _mon_y2 + 10:
            _shift = (_mon_y2 + 10) - y1
            y1 += _shift
            y2 += _shift
        relation_state["keyboard_y2"] = y2

    # 최소 크기 보정
    mw, mh = _MIN_FRONT_SIZE.get(cat, (40, 20))
    if x2 - x1 < mw:
        cx = (x1 + x2) / 2
        x1, x2 = cx - mw / 2, cx + mw / 2
    if y2 - y1 < mh:
        y1 = y2 - mh

    x1, y1, x2, y2 = max(0, int(x1)), max(0, int(y1)), int(x2), int(y2)
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)


def compute_relational_layout(products, desk_corners, desk_width_mm, desk_depth_mm):
    # 정석 구도 관계 체인. 위치를 제품 크기·책상에서 계산 (top-available/기존물체 검출 미사용).
    # 모니터 중앙뒤 → 키보드 모니터앞 → 마우스 키보드오른쪽 → 마우스패드 둘감쌈 → 스피커 모니터옆.
    dw_mm = max(1, int(desk_width_mm)) if desk_width_mm else 1400
    dd_mm = max(1, int(desk_depth_mm)) if desk_depth_mm else int(dw_mm * 0.5)

    prod_by_cat: dict = {}
    for p in products:
        c = normalize_category(p.category)
        if c in _CATEGORY_DIMS_MM and c not in prod_by_cat:
            prod_by_cat[c] = p

    def _w(cat, p):
        return getattr(p, "width_mm", None) or _CATEGORY_DIMS_MM[cat][0]
    def _rw(cat, p):
        return min(0.95, _w(cat, p) / dw_mm)

    tgt: dict = {}
    # 모니터: 가로 중앙, 책상 뒤
    if "MONITOR" in prod_by_cat:
        tgt["MONITOR"] = (0.50, _CAT_RY_RANGE["MONITOR"][0] + 0.02)
    mon_rx = tgt.get("MONITOR", (0.50, 0.0))[0]
    # 선반: 모니터 가로 정렬, 비슷한 깊이
    if "DESK_SHELF" in prod_by_cat:
        _r = _CAT_RY_RANGE.get("DESK_SHELF", (0.15, 0.32))
        tgt["DESK_SHELF"] = (mon_rx, _r[0] + 0.02)
    # 키보드: 모니터 가로 정렬, 책상 앞쪽
    kb_ry = _CAT_RY_RANGE["KEYBOARD"][1] - 0.03
    if "KEYBOARD" in prod_by_cat:
        tgt["KEYBOARD"] = (mon_rx, kb_ry)
        kb_rw = _rw("KEYBOARD", prod_by_cat["KEYBOARD"])
    else:
        kb_rw = 0.30
    # 마우스: 키보드 오른쪽 끝 + 마우스폭 (제품 크기로 계산)
    if "MOUSE" in prod_by_cat:
        m_rw = _rw("MOUSE", prod_by_cat["MOUSE"])
        mouse_rx = min(0.95, mon_rx + kb_rw / 2 + m_rw / 2 + 0.015)
        tgt["MOUSE"] = (mouse_rx, kb_ry)
    else:
        m_rw, mouse_rx = 0.05, mon_rx + kb_rw / 2
    # 마우스패드: 키보드 왼쪽 ~ 마우스 오른쪽을 감싸는 중심 (아래 레이어)
    if "MOUSEPAD" in prod_by_cat:
        left  = mon_rx - kb_rw / 2
        right = mouse_rx + m_rw / 2
        tgt["MOUSEPAD"] = ((left + right) / 2, kb_ry)
    # 스피커: 모니터 왼쪽 바깥 (단일 제품 가정)
    if "SPEAKER" in prod_by_cat:
        _r = _CAT_RY_RANGE.get("SPEAKER", (0.18, 0.45))
        tgt["SPEAKER"] = (max(0.08, mon_rx - 0.30), _r[0] + 0.05)
    # 나머지(조명/시계/소품 등): _PREFERRED_POS 기본 상대 위치
    for c in prod_by_cat:
        if c not in tgt:
            pref = _PREFERRED_POS.get(c, {"rx": 0.50, "ry": 0.40})
            tgt[c] = (pref["rx"], pref["ry"])

    # z-order/배치 순서대로 front bbox 산출 (마우스패드가 키보드보다 먼저=아래)
    rel_state: dict = {"monitor_rx": mon_rx}
    placements: list = []
    for cat in sorted(prod_by_cat, key=lambda c: _PLACEMENT_ORDER.get(c, 999)):
        p      = prod_by_cat[cat]
        rx, ry = tgt[cat]
        w_mm   = _w(cat, p)
        d_mm   = getattr(p, "depth_mm", None) or _CATEGORY_DIMS_MM[cat][1]
        bbox   = _front_bbox_from_corners(cat, rx, ry, w_mm, d_mm, desk_corners, dw_mm, dd_mm, rel_state)
        if bbox is None:
            continue
        rel_state[f"{cat.lower()}_rx"] = rx
        placements.append({
            "product": p, "region": bbox,
            "placement_source": "relational_layout",
            "selected_reason": "relational", "fallback_reason": None,
            "anchor_rx": round(rx, 3), "anchor_ry": round(ry, 3),
            "score": None, "score_meta": None, "available_region_id": None,
            "candidate_count": None, "overlap_reject_count": None,
            "low_score_reject_count": None,
        })
    return placements


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
    # pref_ry는 카테고리별 _PREFERRED_POS에서 가져와 통일. 카테고리간 의존은 rx에만 적용.
    # 옛 코드는 ry까지 별도 magic (0.62/0.72)을 박아 _PREFERRED_POS가 사실상 무시됐음.
    _pref = _PREFERRED_POS.get(cat, {"rx": 0.50, "ry": 0.50})
    if cat == "KEYBOARD" and "monitor_rx" in relation_state:
        pref_rx = relation_state["monitor_rx"]
        pref_ry = _pref["ry"]
    elif cat == "MOUSE":
        pref_rx = min(1.0, relation_state.get("keyboard_rx", 0.50) + 0.20)
        pref_ry = relation_state.get("keyboard_ry", _pref["ry"])
    elif cat == "MOUSEPAD":
        # MOUSEPAD가 placement 순서상 MOUSE보다 먼저 배치되니 MOUSE 위치를 직접 참조 못 함
        # → KEYBOARD 위치 기반으로 동일하게 계산.
        pref_rx = min(1.0, relation_state.get("keyboard_rx", 0.50) + 0.20)
        pref_ry = relation_state.get("keyboard_ry", _pref["ry"])
    elif cat == "SPEAKER":
        pref_rx = 0.20 if rx <= 0.50 else 0.80
        pref_ry = relation_state.get("monitor_ry", _pref["ry"])
    else:
        pref_rx, pref_ry = _pref["rx"], _pref["ry"]

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
        # MOUSE ↔ MOUSEPAD는 겹치는 게 정상 (마우스가 마우스패드 위에 얹힘)
        if {cat, prev.get("cat", "")} == {"MOUSE", "MOUSEPAD"}:
            continue
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
                score         = rule_score * (1.0 - RANKER_WEIGHT) + learned_score * RANKER_WEIGHT
                ranker_used   = (RANKER_WEIGHT > 0.0)
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
    front_desk_bbox_override: tuple | None = None,  # (x1,y1,x2,y2) — SAM2 클릭 mask에서 추출한 책상 윗면 bbox
    desk_corners: list | None = None,               # 4x2 TL,TR,BR,BL — 사용자가 직접 클릭한 책상 윗면 4점
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
        return [], {"tabletop_valid": True, "tabletop_invalid_reason": None,
                    "no_available_regions": True}

    ys, xs = np.where(desk_mask_np > 0)
    if len(xs) == 0:
        return [], {"tabletop_valid": True, "tabletop_invalid_reason": None,
                    "empty_top_desk_mask": True}
    tv_dx1, tv_dy1 = int(xs.min()), int(ys.min())
    tv_dx2, tv_dy2 = int(xs.max()), int(ys.max())
    tv_dw = max(1, tv_dx2 - tv_dx1)
    tv_dh = max(1, tv_dy2 - tv_dy1)

    # === Tabletop bbox 결정 + validation (1차 commit, 2026-05-26) ===
    # 후보 우선순위: sam2_override → dino_clamped(flag ON) → dino_raw
    # 각 후보를 validate_tabletop_bbox로 검증, 첫 valid 사용. 모두 invalid이면 placement 강행 X.
    #
    # 회귀 원인:
    #   1) DINO raw bbox에 0.45/0.72 hard clamp가 책상 하단을 잘라먹어 fv_dh=124 (이미지의 23%)로
    #      줄여 top-view anchor 투영이 책상 뒤 벽으로 튕김 → 기본 OFF (config flag).
    #   2) SAM2 override의 derivative-based band가 윗면 위쪽 edge만 67px 잡음 → extract_desk_top_band
    #      개선(full_bbox fallback) + 여기서 sanity check 한 번 더.
    #   3) 빈 책상 모드에서 SAM2가 desk body/다리/바닥까지 잡으면 bbox가 너무 두꺼움 → max ratio check.
    #
    # heuristic(0.45h~0.72h)은 실제 책상과 무관한 magic bbox이므로 후보에서 제외.
    _candidates: list[tuple[str, tuple]] = []
    if desk_corners is not None:
        # 사용자 클릭 4점 → bounding box를 최우선 후보로 (검출 우회). 배치는 4점 perspective 사용.
        _dc = np.asarray(desk_corners, dtype=np.float32)
        _candidates.append(("user_corners", (
            int(_dc[:, 0].min()), int(_dc[:, 1].min()),
            int(_dc[:, 0].max()), int(_dc[:, 1].max()),
        )))
    if front_desk_bbox_override is not None:
        _candidates.append(("sam2_override", tuple(front_desk_bbox_override)))
    _dino_raw = _detect_desk_bbox(front_image)
    if _dino_raw is not None:
        _dr = tuple(_dino_raw)
        if TABLETOP_BBOX_HARD_CLAMP_ENABLED:
            _dx1, _dy1, _dx2, _dy2 = _dr
            _dy1c = max(_dy1, int(fv_h * 0.45))
            _dy2c = min(_dy2, int(fv_h * 0.72))
            if _dy2c > _dy1c:
                _candidates.append(("dino_clamped", (_dx1, _dy1c, _dx2, _dy2c)))
        _candidates.append(("dino_raw", _dr))

    _attempts: list[dict] = []
    chosen_bbox   = None
    chosen_source = None
    chosen_vmeta  = None
    for _src, _bb in _candidates:
        _is_valid, _reason, _vmeta = validate_tabletop_bbox(_bb, fv_w, fv_h)
        _attempts.append({"source": _src, "bbox": list(_bb), "valid": _is_valid,
                          "reason": _reason, **_vmeta})
        print(f"[Desk bbox] candidate={_src} bbox={_bb} valid={_is_valid} "
              f"reason={_reason} fv_dh={_vmeta['fv_dh']} ratio={_vmeta['fv_dh_ratio']}")
        if _is_valid and chosen_bbox is None:
            chosen_bbox, chosen_source, chosen_vmeta = _bb, _src, _vmeta

    tabletop_meta: dict = {
        "tabletop_valid":          chosen_bbox is not None,
        "tabletop_source":         chosen_source,
        "tabletop_bbox":           list(chosen_bbox) if chosen_bbox else None,
        "tabletop_invalid_reason": None if chosen_bbox else "all_candidates_invalid",
        "tabletop_clamp_flag":     TABLETOP_BBOX_HARD_CLAMP_ENABLED,
        "tabletop_candidates":     _attempts,
        "image_w":                 fv_w,
        "image_h":                 fv_h,
    }
    if chosen_vmeta is not None:
        tabletop_meta.update({
            "tabletop_fv_dh":       chosen_vmeta["fv_dh"],
            "tabletop_fv_dw":       chosen_vmeta["fv_dw"],
            "tabletop_fv_dh_ratio": chosen_vmeta["fv_dh_ratio"],
        })

    if debug_dir is not None:
        (debug_dir / "front_desk_bbox_debug.json").write_text(
            _dj.dumps(tabletop_meta, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    # 책상 평면 확보: 검증 bbox > desk_corners > dino_raw > 이미지 비율 (관계 배치는 평면만 있으면 가능)
    if chosen_bbox is None:
        if desk_corners is not None:
            _dc = np.asarray(desk_corners, dtype=np.float32)
            chosen_bbox = (int(_dc[:, 0].min()), int(_dc[:, 1].min()),
                           int(_dc[:, 0].max()), int(_dc[:, 1].max()))
            chosen_source = "user_corners_fallback"
        elif _dino_raw is not None:
            chosen_bbox = tuple(int(v) for v in _dino_raw)
            chosen_source = "dino_raw_fallback"
        else:
            chosen_bbox = (0, int(fv_h * 0.40), fv_w, int(fv_h * 0.72))
            chosen_source = "image_ratio_fallback"
        print(f"[Desk bbox] 검증 실패 → 관계배치용 평면 fallback: {chosen_source} {chosen_bbox}")

    fv_dx1, fv_dy1, fv_dx2, fv_dy2 = chosen_bbox
    fv_dw = max(1, fv_dx2 - fv_dx1)
    fv_dh = max(1, fv_dy2 - fv_dy1)
    print(f"[Desk bbox] chosen source={chosen_source} bbox=({fv_dx1},{fv_dy1},{fv_dx2},{fv_dy2}) "
          f"fv_dh={fv_dh} ratio={fv_dh/max(fv_h,1):.3f} clamp_flag={TABLETOP_BBOX_HARD_CLAMP_ENABLED}")

    # ★ 관계 체인 배치 — top-available/기존물체 검출/위치 magic 전부 미사용.
    # 평면: 사용자 4점 우선, 없으면 책상 bbox. 위치는 제품 크기·관계로만 결정.
    _plane = desk_corners if desk_corners is not None else [
        [fv_dx1, fv_dy1], [fv_dx2, fv_dy1], [fv_dx2, fv_dy2], [fv_dx1, fv_dy2],
    ]
    _rel = compute_relational_layout(products, _plane, desk_width_mm, desk_depth_mm)
    if _rel:
        print(f"[Placement] 관계 체인 배치 {len(_rel)}개 — top-available 미사용")
        return _rel, tabletop_meta

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
        d_mm = getattr(p, "depth_mm", None) or _CATEGORY_DIMS_MM[cat][1]

        best_score       = -999.0
        best_cand        = None
        best_score_meta  = None
        candidate_count  = 0
        overlap_reject   = 0
        low_score_reject = 0

        # ry pre-filter: 카테고리별 안전 범위 밖 후보는 score 무의미 (어차피 clamp됨).
        # 이전엔 모든 후보 scoring → ry=0.07 같은 out-of-range 값이 best로 뜨고,
        #   _front_bbox_for_anchor에서 0.18로 clamp되어 score 의미 불일치.
        _ry_min, _ry_max = _CAT_RY_RANGE.get(cat, (0.05, 0.95))
        ry_filtered_reject = 0

        for region in available_regions:
            for anchor_x, anchor_y in _sample_points_in_region(region, n=7):
                rx = max(0.0, min(1.0, (anchor_x - tv_dx1) / tv_dw))
                ry = max(0.0, min(1.0, (anchor_y - tv_dy1) / tv_dh))
                if not (_ry_min <= ry <= _ry_max):
                    ry_filtered_reject += 1
                    continue
                if desk_corners is not None:
                    fv_bbox = _front_bbox_from_corners(
                        cat, rx, ry, w_mm, d_mm,
                        desk_corners, desk_width_mm, desk_depth_mm, relation_state,
                    )
                else:
                    fv_bbox = _front_bbox_for_anchor(
                        cat, rx, ry, w_mm,
                        fv_dx1, fv_dy1, fv_dx2, fv_dy2, fv_dw, fv_dh, fv_w, fv_h,
                        desk_width_mm, relation_state,
                        d_mm=d_mm, desk_depth_mm=desk_depth_mm,
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
                if desk_corners is not None:
                    _fb = _front_bbox_from_corners(
                        cat, _forced_rx, _fry, w_mm, d_mm,
                        desk_corners, desk_width_mm, desk_depth_mm, relation_state,
                    )
                else:
                    _fb = _front_bbox_for_anchor(
                        cat, _forced_rx, _fry, w_mm,
                        fv_dx1, fv_dy1, fv_dx2, fv_dy2, fv_dw, fv_dh, fv_w, fv_h,
                        desk_width_mm, relation_state,
                        d_mm=d_mm, desk_depth_mm=desk_depth_mm,
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
                  f"ry_rej={ry_filtered_reject} → fv({x1},{y1},{x2},{y2})")
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
            elif cat == "MOUSEPAD":
                # MOUSE는 이 영역 안에 강제 배치 (_front_bbox_for_anchor에서 사용)
                relation_state["mousepad_region"] = (x1, y1, x2, y2)
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

    return placements, tabletop_meta


def analyze_top_view_only(
    top_view_image: Image.Image,
    desk_width_mm: int,
    desk_depth_mm: int,
    mode=None,
    remover=None,
) -> dict | None:
    # calc_placements_from_available_space의 분석 단계만 추출.
    # 추천 단계에 size 제약 전달용. (placement 호출 시점에 다시 분석되므로 결과는 일회성)
    from .models import RemoveMode
    from .space_analysis import (
        analyze_space, make_full_desk_mask, keep_largest_component,
    )
    from .mask_utils import postprocess_occupied_mask
    from .object_removal_processor import get_object_removal_processor

    if mode is None:
        mode = RemoveMode.own_desk
    if remover is None:
        remover = get_object_removal_processor()

    tv_w, tv_h = top_view_image.size

    top_det = remover.detect_with_prompt(
        image=top_view_image, prompt=_REMOVAL_PROMPT, max_area_ratio=0.40,
    )
    occupied_np = _build_occupied_mask_from_detection(top_det, tv_w, tv_h)
    occupied_np = postprocess_occupied_mask(occupied_np)

    desk_mask_np = remover.detect_desk_mask(top_view_image)
    if desk_mask_np is None:
        desk_mask_np = make_full_desk_mask((tv_h, tv_w))
    else:
        desk_mask_np = keep_largest_component(desk_mask_np)

    desk_width_cm = desk_width_mm / 10
    desk_depth_cm = desk_depth_mm / 10
    remove_mask = None if mode == RemoveMode.add else occupied_np

    space_info = analyze_space(
        occupied_mask=occupied_np,
        desk_width_cm=desk_width_cm,
        desk_depth_cm=desk_depth_cm,
        desk_mask=desk_mask_np,
        remove_mask=remove_mask,
    )

    ys, xs = np.where(desk_mask_np > 0)
    if len(xs) == 0:
        return None
    tv_dx1, tv_dy1 = int(xs.min()), int(ys.min())
    tv_dx2, tv_dy2 = int(xs.max()), int(ys.max())
    space_info["desk_bbox_tv"] = (tv_dx1, tv_dy1, tv_dx2, tv_dy2)
    return space_info


def compute_size_constraints_from_space(space_info: dict) -> dict[str, list[int]]:
    # 각 카테고리의 _CAT_RY_RANGE 안에서 최대 가용 (width_mm, depth_mm) 계산.
    # recommendation에 보내서 후보 검색 시 사이즈 필터로 사용.
    available_regions = space_info.get("available_regions", [])
    cm_per_px = space_info.get("cm_per_px", {"x": 0.1, "y": 0.1})
    desk_bbox_tv = space_info.get("desk_bbox_tv")
    if not desk_bbox_tv:
        return {}

    tv_dx1, tv_dy1, tv_dx2, tv_dy2 = desk_bbox_tv
    tv_dh = max(1, tv_dy2 - tv_dy1)
    mm_per_px_x = cm_per_px["x"] * 10
    mm_per_px_y = cm_per_px["y"] * 10

    constraints: dict[str, list[int]] = {}
    for cat, (ry_min, ry_max) in _CAT_RY_RANGE.items():
        y_start = tv_dy1 + int(ry_min * tv_dh)
        y_end = tv_dy1 + int(ry_max * tv_dh)
        max_w_mm = 0
        max_d_mm = 0
        for region in available_regions:
            x1, y1, x2, y2 = region["bbox"]
            inter_y1 = max(y1, y_start)
            inter_y2 = min(y2, y_end)
            if inter_y2 <= inter_y1:
                continue
            w_mm = int((x2 - x1) * mm_per_px_x)
            d_mm = int((inter_y2 - inter_y1) * mm_per_px_y)
            if w_mm > max_w_mm:
                max_w_mm = w_mm
            if d_mm > max_d_mm:
                max_d_mm = d_mm
        if max_w_mm > 0 and max_d_mm > 0:
            constraints[cat] = [max_w_mm, max_d_mm]
    return constraints
