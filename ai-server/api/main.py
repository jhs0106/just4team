import ast
import csv
import uuid
import asyncio
import base64
import logging
import traceback
import torch
import numpy as np
import cv2
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image

import sys as _sys

Path("logs").mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.ERROR,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("logs/server_errors.log", encoding="utf-8"),
    ],
)
log = logging.getLogger("ai-server")

# stdout을 파일에도 동시 기록 — 서버 터미널 접근 없이 print 로그 확인용
class _Tee:
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            try:
                f.write(obj)
                f.flush()
            except Exception:
                pass
    def flush(self):
        for f in self.files:
            try:
                f.flush()
            except Exception:
                pass
    def isatty(self):
        return False
    def fileno(self):
        return self.files[0].fileno() if self.files else 1

_stdout_log_file = open("logs/stdout.log", "a", encoding="utf-8")
_sys.stdout = _Tee(_sys.__stdout__, _stdout_log_file)

from .models import (
    StyleName, RemoveMode, JobStatus,
    ObjectRemovalRequest, ObjectRemovalResult,
    ProductPlaceRequest, ProductPlaceResult,
    SegmentRequest, SegmentResult,
    GenerateRequest, GenerateResult,
)
from .space_analysis import (
    analyze_space,
    make_full_desk_mask, keep_largest_component,
)
from .mask_utils import postprocess_occupied_mask
from .object_removal_processor import get_object_removal_processor
from .lama_processor import get_lama_processor
from .product_inpaint_processor import get_product_inpaint_processor
from .controlnet_inpaint_processor import get_controlnet_inpaint_processor
from .sam2_processor import get_sam2_processor


def b64_to_image(b64: str) -> Image.Image:
    return Image.open(BytesIO(base64.b64decode(b64))).convert("RGB")


def image_to_b64(image: Image.Image) -> str:
    buf = BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


job_store: dict = {}
executor = ThreadPoolExecutor(max_workers=1)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    print("서버 시작 완료. 모든 모델은 첫 요청 시 로드됩니다.")
    yield
    executor.shutdown(wait=False)


app = FastAPI(
    title="Deskterior AI API",
    description="데스크테리어 시뮬레이션 AI 서버",
    version="5.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")


@app.get("/health")
async def health_check():
    return {"status": "ok"}


@app.get("/styles")
async def list_styles():
    return {"styles": [s.value for s in StyleName]}


@app.get("/jobs/{job_id}")
async def get_job(job_id: str):
    if job_id not in job_store:
        raise HTTPException(status_code=404, detail="존재하지 않는 job_id입니다.")
    return job_store[job_id]


PRODUCT_IMAGE_DIR = Path("data/test/processed_images")
_CATALOG_CSV      = Path("data/test/products.csv")


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


_CATEGORY_ALIASES = {
    "KEYBOARD":      "KEYBOARD",
    "MOUSE":         "MOUSE",
    "MOUSEPAD":      "MOUSEPAD",
    "MOUSE_PAD":     "MOUSEPAD",
    "MOUSE PAD":     "MOUSEPAD",
    "MONITOR":       "MONITOR",
    "SPEAKER":       "SPEAKER",
    "LAMP":          "DESK_LAMP",
    "DESK LAMP":     "DESK_LAMP",
    "DESK_LAMP":     "DESK_LAMP",
    "DESK SHELF":    "DESK_SHELF",
    "DESK_SHELF":    "DESK_SHELF",
    "MONITOR RISER": "DESK_SHELF",
    "LAPTOP STAND":  "LAPTOP_STAND",
    "LAPTOP_STAND":  "LAPTOP_STAND",
    "DECO":          "DECO",
    "DECOR":         "DECO",
    "CLOCK":         "CLOCK",
}


def normalize_category(category: str) -> str:
    key = category.strip().upper().replace("-", "_")
    key_space = key.replace("_", " ")
    return _CATEGORY_ALIASES.get(key, _CATEGORY_ALIASES.get(key_space, key))


# ── products.csv 임시 DB ─────────────────────────────────────────────────────

_product_catalog: dict = {}


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
    """req.products에 width_mm/depth_mm가 없으면 products.csv 값으로 보완. category도 정규화."""
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


# 카테고리별 기본 치수 (mm) — products.csv metadata 없을 때 fallback
_CATEGORY_DIMS_MM = {
    "KEYBOARD":     (440, 130),
    "MOUSE":        (70,  120),
    "MOUSEPAD":     (900, 400),
    "MONITOR":      (600, 200),
    "SPEAKER":      (90,  120),
    "DESK_LAMP":    (80,  400),
    "DESK_SHELF":   (600, 200),
    "LAPTOP_STAND": (280, 250),
    "DECO":         (80,  80),
    "CLOCK":        (100, 100),
}

# MONITOR→KEYBOARD→MOUSE 순서여야 관계 기반 배치가 작동함
# KEYBOARD(15)가 DESK_SHELF(20)보다 먼저: 핵심 제품이 모니터 아래 공간 선점
_PLACEMENT_ORDER = {
    "MONITOR":      10,
    "KEYBOARD":     15,
    "DESK_SHELF":   20,
    "MOUSEPAD":     40,
    "MOUSE":        50,
    "SPEAKER":      60,
    "DESK_LAMP":    70,
    "CLOCK":        80,
    "DECO":         90,
}

# CV 합성 전용 카테고리 (ControlNet 생성 금지)
_CV_ONLY_CATS = {"MONITOR", "DESK_SHELF", "KEYBOARD", "DESK_LAMP"}

# cv_composite 모드에서 composite_product_simple에 적용할 카테고리별 최대 scale
_CV_CAT_MAX_SCALE: dict[str, float] = {
    "KEYBOARD":   4.0,
    "MOUSE":      3.0,
    "MONITOR":    2.5,
    "SPEAKER":    2.5,
    "DESK_LAMP":  2.5,
    "DESK_SHELF": 2.0,
}

# top-view 기준 선호 위치 (rx=0 좌/1 우, ry=0 뒤/1 앞)
_PREFERRED_POS = {
    "MONITOR":      {"rx": 0.50, "ry": 0.20},
    "DESK_SHELF":   {"rx": 0.50, "ry": 0.20},
    "KEYBOARD":     {"rx": 0.50, "ry": 0.62},
    "MOUSEPAD":     {"rx": 0.50, "ry": 0.65},
    "MOUSE":        {"rx": 0.70, "ry": 0.62},
    "SPEAKER":      {"rx": 0.25, "ry": 0.25},
    "DESK_LAMP":    {"rx": 0.12, "ry": 0.30},
    "DECO":         {"rx": 0.75, "ry": 0.35},
    "CLOCK":        {"rx": 0.80, "ry": 0.30},
    "LAPTOP_STAND": {"rx": 0.50, "ry": 0.45},
}

# 제품별 최소 front-view bbox 크기 (px)
_MIN_FRONT_SIZE = {
    "MONITOR":   (220, 150),
    "KEYBOARD":  (180, 45),
    "MOUSE":     (60, 45),
    "SPEAKER":   (55, 55),
    "DESK_LAMP": (80, 120),
    "DECO":      (45, 45),
    "CLOCK":     (55, 40),
}

# 카테고리별 tight_crop aspect ratio (w/h) 허용 범위 — 범위 밖이면 잘못된 제품 이미지
_CAT_ASPECT_VALID: dict[str, tuple[float, float]] = {
    "KEYBOARD":  (3.0, 99.0),   # 키보드는 가로로 매우 긴 형태여야 함
    "MOUSE":     (0.6, 1.8),    # 마우스는 정방형에 가까운 형태여야 함
    "MONITOR":   (1.2, 3.0),    # 모니터는 가로가 세로보다 넓어야 함
    "SPEAKER":   (0.4, 2.0),
    "DESK_LAMP": (0.3, 2.5),
}

# scoring으로 선택된 anchor → front-view bbox 변환 시 카테고리별 y 보정 (px)
_CONTACT_Y_OFFSET = {
    "KEYBOARD": 32,
    "MOUSE":    20,
}

# 카테고리 쌍별 허용 IoU 상한 — 이 값 이상이면 overlap으로 거부
_OVERLAP_TOLERANCE: dict = {
    frozenset({"MONITOR",    "KEYBOARD"}):  0.35,  # 키보드가 모니터 하단과 살짝 겹쳐도 허용
    frozenset({"MONITOR",    "DESK_SHELF"}):0.30,
    frozenset({"DESK_SHELF", "KEYBOARD"}):  0.40,  # 모니터 받침대 앞에 키보드 배치 허용
    frozenset({"KEYBOARD",   "MOUSEPAD"}):  0.50,  # 키보드가 마우스패드 위에 놓임
    frozenset({"MOUSE",      "MOUSEPAD"}):  0.60,  # 마우스가 마우스패드 위에 놓임
}
_DEFAULT_OVERLAP_THR = 0.10


def _overlap_threshold(cat_a: str, cat_b: str) -> float:
    return _OVERLAP_TOLERANCE.get(frozenset({cat_a, cat_b}), _DEFAULT_OVERLAP_THR)


# 45도 앵글 뷰에서 제품의 높이/너비 비율
# 수직 제품(모니터·스탠드·램프)은 크고, 수평 제품(키보드·마우스패드)은 작음
_FRONT_HEIGHT_RATIO = {
    "MONITOR":      0.68,
    "KEYBOARD":     0.22,
    "MOUSE":        0.75,
    "MOUSEPAD":     0.25,
    "SPEAKER":      1.10,
    "DESK_LAMP":    1.70,
    "DESK_SHELF":   0.20,
    "LAPTOP_STAND": 0.40,
    "DECO":         0.90,
    "CLOCK":        0.90,
}

# desk_width_mm 없을 때 책상 너비 대비 제품 너비 비율
_DESK_W_RATIO = {
    "KEYBOARD":     0.33,
    "MOUSE":        0.07,
    "MOUSEPAD":     0.55,
    "MONITOR":      0.55,   # 넓게 잡아야 512×512 압축 후에도 가로 모니터로 보임
    "SPEAKER":      0.09,
    "DESK_LAMP":    0.06,
    "DESK_SHELF":   0.45,
    "LAPTOP_STAND": 0.22,
    "DECO":         0.07,
    "CLOCK":        0.08,
}

# DINO가 반환하는 label 문자열 → 우리 카테고리 매핑
_DINO_LABEL_TO_CATEGORY = {
    "keyboard":     "KEYBOARD",
    "mouse":        "MOUSE",
    "mouse pad":    "MOUSEPAD",
    "mousepad":     "MOUSEPAD",
    "monitor":      "MONITOR",
    "speaker":      "SPEAKER",
    "desk lamp":    "DESK_LAMP",
    "lamp":         "DESK_LAMP",
    "headset":      "HEADSET",
    "desk shelf":   "DESK_SHELF",
    "monitor riser":"DESK_SHELF",
    "laptop stand": "LAPTOP_STAND",
    "clock":        "CLOCK",
    "cup":          "DECO",
    "mug":          "DECO",
}


def _match_products_to_detections(products, detections) -> list:
    # 요청 제품과 DINO 감지 결과를 카테고리 기준으로 매칭 → [{"product", "region"}]
    # 매칭된 detection은 재사용하지 않음 (1:1 매칭)
    available = list(detections)
    matched = []
    unmatched = []

    for p in products:
        cat = normalize_category(p.category)
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
    from PIL import ImageDraw
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
    # DINO bbox → 중심 위치만 참조, 크기는 제품 실제 치수(mm)로 재계산
    x1, y1, x2, y2 = det_box
    cx     = (x1 + x2) // 2
    cy_bot = y2  # 하단 기준 정렬

    cat  = product.category.upper()
    w_mm = getattr(product, "width_mm", None) or _CATEGORY_DIMS_MM.get(cat, (200, 200))[0]

    if desk_bbox and desk_width_mm:
        DW        = desk_bbox[2] - desk_bbox[0]
        px_per_mm = DW / desk_width_mm
        pw = int(w_mm * px_per_mm)
    else:
        ref_w = _CATEGORY_DIMS_MM.get(cat, (200, 200))[0]
        pw = int(max(x2 - x1, 1) * w_mm / ref_w)

    # height: DINO 감지 높이의 3배 기준으로 잡되 최소 80px 보장
    # (고정 비율보다 실제 이미지 스케일 기반이 더 안정적)
    det_h = max(y2 - y1, 1)
    ph    = max(det_h * 3, 80)
    pw    = max(pw, 60)

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
    ps     = 0.60 + 0.40 * ry
    fv_pw  = max(40, min(int(w_mm * fv_dw / (desk_width_mm or 1200) * ps), int(fv_dw * 0.65)))
    fv_ph  = max(20, int(fv_pw * _FRONT_HEIGHT_RATIO.get(cat, 0.80)))
    min_w, min_h = _MIN_FRONT_SIZE.get(cat, (40, 20))
    fv_pw  = max(fv_pw, min_w)
    fv_ph  = max(fv_ph, min_h)
    _y_override = False
    _rx_adj = rx

    if cat == "MONITOR":
        fv_pw = max(min(int(fv_dw * 0.48), 420), 240)
        fv_ph = int(fv_pw * 0.60)
        # contact point: stand/base가 책상 뒷면(back line)에 닿는 위치
        _monitor_contact_y = int(fv_dy1 + fv_dh * 0.28)
        y2 = _monitor_contact_y
        y1 = max(int(fv_h * 0.05), y2 - fv_ph)
        relation_state["monitor_contact_y"] = y2
        _y_override = True
    elif cat == "KEYBOARD":
        fv_pw = max(min(int(fv_dw * 0.38), 360), 220)
        fv_ph = max(int(fv_pw * 0.20), 45)
        if "monitor_rx" in relation_state:
            _rx_adj = relation_state["monitor_rx"]
        # 사용자 쪽으로 +32px 내려서 모니터 바로 아래 붙는 현상 완화
        _kb_base_y = int(fv_dy1 + fv_dh * 0.62) + _CONTACT_Y_OFFSET["KEYBOARD"]
        y2 = min(_kb_base_y, int(fv_dy2 - 15))
        y1 = y2 - fv_ph
        relation_state["keyboard_y2"] = y2  # MOUSE y 정렬용
        _y_override = True
    elif cat == "DESK_LAMP":
        fv_pw = max(fv_pw, 90)
        fv_ph = max(fv_ph, 130)
        _rx_adj = max(rx, 0.12) if rx < 0.5 else min(rx, 0.88)
        y2 = int(fv_dy1 + fv_dh * 0.45)
        y1 = y2 - fv_ph
        _y_override = True
    elif cat == "DESK_SHELF":
        fv_pw = max(min(int(fv_dw * 0.40), 320), 150)
        fv_ph = max(int(fv_pw * 0.20), 30)
        if "monitor_rx" in relation_state:
            _rx_adj = relation_state["monitor_rx"]
        y2 = int(fv_dy1 + fv_dh * 0.35)
        y1 = y2 - fv_ph
        _y_override = True
    elif cat == "SPEAKER":
        fv_pw = max(fv_pw, 60)
        fv_ph = max(fv_ph, 60)
    elif cat == "DECO":
        fv_pw = max(fv_pw, 45)
        fv_ph = max(fv_ph, 45)
    elif cat == "MOUSE":
        fv_pw = max(fv_pw, 60)
        fv_ph = max(int(fv_pw * _FRONT_HEIGHT_RATIO["MOUSE"]), 45)
        # keyboard y2 기준으로 정렬 (+20px), chair 영역 침범 방지
        _mouse_base_y = relation_state.get("keyboard_y2", int(fv_dy1 + fv_dh * 0.62))
        y2 = min(_mouse_base_y + _CONTACT_Y_OFFSET["MOUSE"], int(fv_dy2 - 15))
        y1 = y2 - fv_ph
        _y_override = True
    elif cat == "MOUSEPAD":
        fv_ph = int(fv_pw * _FRONT_HEIGHT_RATIO["MOUSEPAD"])

    fv_cx = fv_dx1 + _rx_adj * fv_dw
    x1 = max(0, int(fv_cx - fv_pw // 2))
    x2 = min(fv_w, x1 + fv_pw)
    if not _y_override:
        y2 = min(fv_dy2, int(fv_dy1 + ry * fv_dh))
        y1 = max(0, y2 - fv_ph)
    else:
        y1 = max(0, y1)
        y2 = min(fv_h, y2)

    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)


def calc_placements_from_available_space(
    front_image: Image.Image,
    top_view_image: Image.Image,
    products: list,
    desk_width_mm: int | None,
    desk_depth_mm: int | None,
    mode: RemoveMode = RemoveMode.own_desk,
    remover=None,
    debug_dir: Path | None = None,
) -> list[dict]:
    if remover is None:
        remover = get_object_removal_processor()

    tv_w, tv_h = top_view_image.size
    fv_w, fv_h = front_image.size

    # 1. top-view occupied_mask
    top_det    = remover.detect_with_prompt(
        image=top_view_image, prompt=_REMOVAL_PROMPT, max_area_ratio=0.40,
    )
    occupied_np = _build_occupied_mask_from_detection(top_det, tv_w, tv_h)
    occupied_np = postprocess_occupied_mask(occupied_np)

    # 2. desk_mask (DINO+SAM2, 실패 시 full image)
    desk_mask_np = remover.detect_desk_mask(top_view_image)
    if desk_mask_np is None:
        print("[AvailSpace] desk_mask 실패 → full image")
        desk_mask_np = make_full_desk_mask((tv_h, tv_w))
    else:
        desk_mask_np = keep_largest_component(desk_mask_np)

    # 3. available space 분석
    desk_width_cm = (desk_width_mm / 10) if desk_width_mm else 120.0
    desk_depth_cm = (desk_depth_mm / 10) if desk_depth_mm else desk_width_cm * 0.55

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
    mm_per_px_x       = cm_per_px["x"] * 10
    mm_per_px_y       = cm_per_px["y"] * 10

    if not available_regions:
        print("[AvailSpace] 가용 영역 없음 → fallback")
        return []

    # 4. top-view desk bbox (정규화 기준)
    ys, xs = np.where(desk_mask_np > 0)
    if len(xs) == 0:
        return []
    tv_dx1, tv_dy1 = int(xs.min()), int(ys.min())
    tv_dx2, tv_dy2 = int(xs.max()), int(ys.max())
    tv_dw = max(1, tv_dx2 - tv_dx1)
    tv_dh = max(1, tv_dy2 - tv_dy1)

    # 5. front-view desk bbox
    fv_bbox = _detect_desk_bbox(front_image)
    if fv_bbox:
        fv_dx1, fv_dy1, fv_dx2, fv_dy2 = fv_bbox
    else:
        fv_dx1, fv_dy1 = 0, int(fv_h * 0.15)
        fv_dx2, fv_dy2 = fv_w, int(fv_h * 0.68)
    # DINO가 의자 포함 전체를 책상으로 인식할 수 있으므로 항상 68%로 cap
    fv_dy2 = min(fv_dy2, int(fv_h * 0.68))
    fv_dw = max(1, fv_dx2 - fv_dx1)
    fv_dh = max(1, fv_dy2 - fv_dy1)

    # _PLACEMENT_ORDER 순으로 정렬 후 scoring 기반 배치
    sorted_products = sorted(
        [p for p in products if normalize_category(p.category) in _CATEGORY_DIMS_MM],
        key=lambda p: _PLACEMENT_ORDER.get(normalize_category(p.category), 999),
    )
    for p in products:
        if normalize_category(p.category) not in _CATEGORY_DIMS_MM:
            print(f"[Placement SKIP] unsupported: {p.category}")

    relation_state:     dict = {}
    placed_norm:        list = []
    placed_front_items: list = []  # {region, cat} — 카테고리별 overlap 임계값 적용용
    placements:         list = []

    for p in sorted_products:
        cat  = normalize_category(p.category)
        w_mm = getattr(p, "width_mm", None) or _CATEGORY_DIMS_MM[cat][0]

        best_score       = -999.0
        best_cand        = None
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
                # 카테고리 쌍별 허용 IoU 상한으로 overlap 체크
                _rej = any(
                    bbox_iou(fv_bbox, prev["region"]) >= _overlap_threshold(cat, prev["cat"])
                    for prev in placed_front_items
                )
                if _rej:
                    overlap_reject += 1
                    continue
                candidate_count += 1
                s = score_region_for_product(cat, rx, ry, placed_norm, relation_state)
                if s < -0.5:
                    low_score_reject += 1
                    continue
                if s > best_score:
                    best_score = s
                    best_cand  = {"region": fv_bbox, "rx": rx, "ry": ry,
                                  "region_id": region["region_id"]}

        # KEYBOARD 강제 후보: scoring 실패 시 monitor 하단 위치에 강제 생성
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
                # MONITOR, DESK_SHELF와의 overlap 허용 — 키보드는 받침대 앞에 배치 가능
                _non_critical = [i for i in placed_front_items
                                 if i["cat"] not in {"MONITOR", "DESK_SHELF"}]
                if any(bbox_iou(_fb, i["region"]) >= _DEFAULT_OVERLAP_THR for i in _non_critical):
                    continue
                _fs = score_region_for_product(cat, _forced_rx, _fry, placed_norm, relation_state)
                best_score = max(_fs, -0.49)  # 강제 후보는 score 하한 보장
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
                "product":              p,
                "region":               (x1, y1, x2, y2),
                "score":                round(best_score, 3),
                "available_region_id":  best_cand["region_id"],
                "placement_source":     _src,
                "selected_reason":      _sel_reason,
                "candidate_count":      candidate_count,
                "overlap_reject_count": overlap_reject,
                "low_score_reject_count": low_score_reject,
                "fallback_reason":      None,
                "anchor_rx":            round(best_cand["rx"], 3),
                "anchor_ry":            round(best_cand["ry"], 3),
            })
            placed_front_items.append({"region": (x1, y1, x2, y2), "cat": cat})
            relation_state[f"{cat.lower()}_rx"] = best_cand["rx"]
            relation_state[f"{cat.lower()}_ry"] = best_cand["ry"]
            placed_norm.append({"rx": best_cand["rx"], "ry": best_cand["ry"], "cat": cat})
        else:
            reason = ("no_non_overlapping_candidate" if candidate_count == 0
                      else "all_candidates_low_score")
            print(f"  [Score FAIL] {cat} cand={candidate_count} "
                  f"ov_rej={overlap_reject} ls_rej={low_score_reject} → {reason}")
            placements.append({
                "product":              p,
                "region":               None,
                "score":                None,
                "available_region_id":  None,
                "placement_source":     "fallback",
                "selected_reason":      None,
                "candidate_count":      candidate_count,
                "overlap_reject_count": overlap_reject,
                "low_score_reject_count": low_score_reject,
                "fallback_reason":      reason,
                "anchor_rx":            None,
                "anchor_ry":            None,
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


def score_region_for_product(
    cat: str,
    rx: float,
    ry: float,
    placed_norm: list,
    relation_state: dict,
) -> float:
    # 관계 기반 선호 위치 조정
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

    dist = ((rx - pref_rx) ** 2 + (ry - pref_ry) ** 2) ** 0.5
    score = max(0.0, 1.0 - dist * 2.0)

    # 하드 제약 penalty
    if cat == "MONITOR" and ry > 0.40:
        score -= 2.0
    if cat == "KEYBOARD" and ry < 0.45:
        score -= 1.5
    if cat == "DESK_LAMP" and 0.20 <= rx <= 0.80:
        score -= 1.5
    if cat in ("DECO", "CLOCK") and 0.30 <= rx <= 0.70 and ry > 0.40:
        score -= 1.0
    # DECO는 MOUSE/KEYBOARD 작업 영역 x축 근처 강한 penalty
    if cat == "DECO":
        for _rkey in ("mouse_rx", "keyboard_rx"):
            if _rkey in relation_state:
                _dx = abs(rx - relation_state[_rkey])
                if _dx < 0.15:
                    score -= 2.0
                elif _dx < 0.28:
                    score -= 0.8

    # 이미 배치된 제품과의 거리 penalty
    for prev in placed_norm:
        d = ((rx - prev["rx"]) ** 2 + (ry - prev["ry"]) ** 2) ** 0.5
        if d < 0.15:
            score -= 2.0
        elif d < 0.25:
            score -= 0.5

    # 책상 가장자리 너무 가까우면 소폭 penalty
    if min(rx, 1 - rx, ry, 1 - ry) < 0.05:
        score -= 0.3

    return score


def _save_topview_overlays(
    top_view_image: Image.Image,
    available_regions: list,
    placements: list,
    placed_norm: list,
    tv_dx1: int, tv_dy1: int, tv_dw: int, tv_dh: int,
    debug_dir: Path,
) -> None:
    from PIL import ImageDraw as _IDraw
    tv_w, tv_h = top_view_image.size

    # available space overlay
    av_base = top_view_image.copy().convert("RGBA")
    overlay = Image.new("RGBA", (tv_w, tv_h), (0, 0, 0, 0))
    d = _IDraw.Draw(overlay)
    for r in available_regions:
        bx = r["bbox_px"]
        rx1, ry1 = int(bx["x"]), int(bx["y"])
        rx2, ry2 = rx1 + int(bx["width"]), ry1 + int(bx["height"])
        d.rectangle([rx1, ry1, rx2, ry2], fill=(0, 255, 0, 60), outline=(0, 200, 0, 200))
        d.text((rx1 + 2, ry1 + 2), str(r["region_id"]), fill=(0, 180, 0, 255))
    av_base.alpha_composite(overlay)
    av_base.convert("RGB").save(debug_dir / "top_available_space_overlay.png")

    # product placement debug on top-view
    pl_img = top_view_image.copy()
    d2 = _IDraw.Draw(pl_img)
    for pn in placed_norm:
        px = int(tv_dx1 + pn["rx"] * tv_dw)
        py = int(tv_dy1 + pn["ry"] * tv_dh)
        d2.ellipse([px - 8, py - 8, px + 8, py + 8], fill=(255, 80, 0))
        d2.text((px + 10, py - 8), pn["cat"], fill=(255, 80, 0))
    pl_img.save(debug_dir / "top_product_placement_debug.png")


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
    arr = np.array(rgba)

    rgb = arr[:, :, :3]
    white = (
        (rgb[:, :, 0] >= threshold) &
        (rgb[:, :, 1] >= threshold) &
        (rgb[:, :, 2] >= threshold)
    )

    mask = white.astype(np.uint8) * 255
    h, w = mask.shape
    flood = mask.copy()
    ff_mask = np.zeros((h + 2, w + 2), np.uint8)

    for sx, sy in [(0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)]:
        if flood[sy, sx] > 0:
            cv2.floodFill(flood, ff_mask, (sx, sy), 128)

    bg = flood == 128
    alpha = arr[:, :, 3]
    alpha[bg] = 0

    if feather > 0:
        alpha = cv2.GaussianBlur(alpha, (feather * 2 + 1, feather * 2 + 1), 0)

    arr[:, :, 3] = alpha
    return Image.fromarray(arr)


def _tight_crop_rgba(img: Image.Image, padding: int = 4) -> Image.Image:
    """alpha 채널 기준 오브젝트 영역만 tight crop — 흰 여백 제거."""
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
    rgba = img.convert("RGBA")
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

    prod = prepare_product_image_for_composite(product_img)

    # 비율 유지 리사이즈, 최소 시각 크기 30px 보장 — category_max_scale까지 확대 허용
    _max_sc = _CV_CAT_MAX_SCALE.get(category, 1.0)
    scale = min(target_w / max(prod.width, 1), target_h / max(prod.height, 1), _max_sc)
    new_w = max(30, int(prod.width * scale))
    new_h = max(30, int(prod.height * scale))
    if (new_w, new_h) != (prod.width, prod.height):
        prod = prod.resize((new_w, new_h), Image.Resampling.LANCZOS)

    # alpha 경계 feather — 스티커 경계 완화
    if edge_feather > 0:
        alpha_arr = np.array(prod.getchannel("A"))
        ksize     = edge_feather * 2 + 1
        alpha_blur = cv2.GaussianBlur(alpha_arr, (ksize, ksize), 0)
        prod_arr   = np.array(prod)
        prod_arr[:, :, 3] = alpha_blur
        prod = Image.fromarray(prod_arr)

    px = x1 + (target_w - prod.width) // 2
    py = y2 - prod.height  # 하단 기준 정렬

    out = base.convert("RGBA")
    out.alpha_composite(prod, (max(0, px), max(0, py)))
    return out.convert("RGB")


def _add_contact_shadow(
    base: Image.Image,
    region: tuple,
    category: str,
) -> Image.Image:
    """제품 하단 contact shadow 합성 (multiply 방식, 카테고리별 타원/직사각형)."""
    x1, y1, x2, y2 = region
    w, h = base.size
    cat = category.upper()
    pw = max(1, x2 - x1)

    shadow = np.zeros((h, w), dtype=np.float32)

    if cat == "MONITOR":
        cx, cy = (x1 + x2) // 2, min(y2, h - 1)
        cv2.ellipse(shadow, (cx, cy), (max(pw // 4, 35), 10), 0, 0, 360, 1.0, -1)
        blur_k, strength = 25, 0.30
    elif cat == "KEYBOARD":
        shadow[max(0, y2 - 6):min(h, y2 + 8), max(0, x1):min(w, x2)] = 1.0
        blur_k, strength = 19, 0.28
    elif cat == "MOUSE":
        cx, cy = (x1 + x2) // 2, min(y2, h - 1)
        cv2.ellipse(shadow, (cx, cy), (max(pw // 2, 20), 9), 0, 0, 360, 1.0, -1)
        blur_k, strength = 19, 0.32
    elif cat == "DESK_LAMP":
        cx, cy = (x1 + x2) // 2, min(y2, h - 1)
        cv2.ellipse(shadow, (cx, cy), (max(pw // 2, 22), 12), 0, 0, 360, 1.0, -1)
        blur_k, strength = 21, 0.28
    else:
        cx, cy = (x1 + x2) // 2, min(y2, h - 1)
        cv2.ellipse(shadow, (cx, cy), (max(pw // 3, 18), 9), 0, 0, 360, 1.0, -1)
        blur_k, strength = 17, 0.25

    shadow = cv2.GaussianBlur(shadow, (blur_k, blur_k), 0)
    shadow = np.clip(shadow * strength, 0.0, 0.45)

    arr = np.array(base.convert("RGB")).astype(np.float32)
    arr = arr * (1.0 - shadow[:, :, np.newaxis])
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def _detect_desk_bbox(image: Image.Image) -> tuple | None:
    # DINO로 책상 영역 감지. 실패 시 None → _calc_regions fallback 사용
    import numpy as np
    from .dino_processor import run_grounding_dino

    img_w, img_h = image.size
    scale = min(512 / img_w, 512 / img_h)
    small = image.resize((int(img_w * scale), int(img_h * scale)))
    small_bgr = np.array(small.convert("RGB"))[:, :, ::-1].copy()

    dets = run_grounding_dino(
        small_bgr, "desk. table.",
        box_threshold=0.20,
        max_area_ratio=0.98,  # 책상은 이미지 대부분을 차지할 수 있음
    )
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
    # 상단 cap: 책상이 이미지 상단 20% 이내에서 시작한다고 감지되면 무시
    # → 후면 제품(모니터 등)이 벽 영역에 배치되는 것을 방지
    dy1 = max(dy1, int(img_h * 0.20))
    # 하단 cap: 의자 영역 침범 방지
    dy2 = min(dy2, int(img_h * 0.68))

    DW = dx2 - dx1
    DH = dy2 - dy1
    cx = (dx1 + dx2) // 2

    # 전면(카메라 가까운 쪽): 책상 표면 55% 지점 — 0.78은 의자 영역까지 내려감
    front_y = dy1 + int(DH * 0.55)
    # 후면(벽 쪽): 책상 상단 경계 — 후면 제품은 위쪽(벽)으로 솟아오름
    back_top = dy1

    def clip(x1, y1, x2, y2):
        return max(0, x1), max(0, y1), min(img_w - 1, x2), min(img_h - 1, y2)

    def perspective_scale(center_y: int) -> float:
        ratio = (center_y - dy1) / max(DH, 1)
        return 0.60 + 0.40 * ratio

    def product_pixel_size(p) -> tuple[int, int]:
        cat     = normalize_category(p.category)
        w_mm    = getattr(p, "width_mm", None) or _CATEGORY_DIMS_MM.get(cat, (100, 100))[0]
        h_ratio = _FRONT_HEIGHT_RATIO.get(cat, 0.80)

        if desk_width_mm and desk_width_mm > 0:
            raw = int(w_mm * DW / desk_width_mm)
        else:
            # 책상 너비 대비 카테고리별 비율로 fallback
            raw = int(DW * _DESK_W_RATIO.get(cat, 0.15))
        pw = min(raw, int(DW * 0.65))  # 책상 너비 65% cap

        ph = int(pw * h_ratio)
        return pw, ph

    regions = []
    for p in products:
        cat = normalize_category(p.category)
        if cat not in _CATEGORY_DIMS_MM:
            continue

        base_pw, _ = product_pixel_size(p)

        if cat in ("KEYBOARD", "MOUSE", "MOUSEPAD"):
            # 전면 제품: 책상 앞쪽, 원근 적용
            ps = perspective_scale(front_y)
            pw = int(base_pw * ps)
            ph = int(pw * _FRONT_HEIGHT_RATIO.get(cat, 0.80))  # pw 기준 비율

            if cat == "KEYBOARD":
                x1 = cx - pw // 2;              x2 = x1 + pw
                y2 = front_y;                    y1 = y2 - ph
            elif cat == "MOUSEPAD":
                x1 = cx - pw // 2;              x2 = x1 + pw
                y2 = front_y + int(DH * 0.05);  y1 = y2 - ph
            else:  # MOUSE
                x1 = cx + int(DW * 0.22);       x2 = x1 + pw
                y2 = front_y + int(DH * 0.03);  y1 = y2 - ph
        else:
            # 후면 제품: 책상 뒤쪽, 위로 솟아오름 (y1은 책상 위 벽 방향)
            ps = perspective_scale(back_top + int(DH * 0.15))
            pw = int(base_pw * ps)
            ph = int(pw * _FRONT_HEIGHT_RATIO.get(cat, 0.80))  # pw 기준 비율

            if cat == "MONITOR":
                # 스탠드는 책상 위에, 화면은 위로 솟아오름
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
            else:  # DECO, CLOCK
                x1 = cx + int(DW * 0.36);        x2 = x1 + pw
                y2 = back_top + int(DH * 0.30);  y1 = max(0, y2 - ph)

        x1, y1, x2, y2 = clip(x1, y1, x2, y2)
        if x2 > x1 and y2 > y1:
            regions.append({"product": p, "region": (x1, y1, x2, y2)})

    return regions


# ── Step 1: Object Removal ──────────────────────────────
@app.post("/remove", response_model=ObjectRemovalResult)
async def run_object_removal(req: ObjectRemovalRequest):
    job_id = str(uuid.uuid4())
    job_store[job_id] = ObjectRemovalResult(job_id=job_id, status=JobStatus.pending)
    asyncio.get_event_loop().run_in_executor(executor, _run_object_removal, job_id, req)
    return job_store[job_id]


def _run_object_removal(job_id: str, req: ObjectRemovalRequest):
    job_store[job_id].status = JobStatus.running
    try:
        remover = get_object_removal_processor()
        image = b64_to_image(req.image_base64)

        if req.prompt:
            detection = remover.detect_with_prompt(image=image, prompt=req.prompt, max_area_ratio=req.max_area_ratio)
        elif req.top_view_image_base64:
            top_view = b64_to_image(req.top_view_image_base64)
            detection = remover.detect_from_top_view(front_view=image, top_view=top_view)
        else:
            detection = remover.detect_and_mask(image=image, max_area_ratio=req.max_area_ratio)

        job_store[job_id].mask_image = detection["mask"]
        job_store[job_id].detection_overlay = detection["overlay"]
        job_store[job_id].num_objects = detection["num_objects"]

        if detection["num_objects"] == 0:
            job_store[job_id].cleaned_image = image_to_b64(image)
            job_store[job_id].status = JobStatus.done
            return

        lama = get_lama_processor()
        current = image
        for mask_pil in detection["individual_masks"]:
            current = lama.inpaint(image=current, mask=mask_pil)
        torch.cuda.empty_cache()

        job_store[job_id].cleaned_image = image_to_b64(current)
        job_store[job_id].status = JobStatus.done
    except Exception as e:
        job_store[job_id].status = JobStatus.failed
        job_store[job_id].error = str(e)
        log.error("job_id=%s\n%s", job_id, traceback.format_exc())


# ── Step 2: Product Placement ───────────────────────────
@app.post("/product_place", response_model=ProductPlaceResult)
async def run_product_place(req: ProductPlaceRequest):
    job_id = str(uuid.uuid4())
    job_store[job_id] = ProductPlaceResult(job_id=job_id, status=JobStatus.pending)
    asyncio.get_event_loop().run_in_executor(executor, _run_product_place, job_id, req)
    return job_store[job_id]


def _run_product_place(job_id: str, req: ProductPlaceRequest):
    job_store[job_id].status = JobStatus.running
    try:
        import numpy as np

        processor = get_product_inpaint_processor()
        image = b64_to_image(req.image_base64)
        img_w, img_h = image.size

        # 마스크에서 배치 영역(bbox) 계산
        mask_np = np.array(b64_to_image(req.mask_base64).convert("L"))
        ys, xs = np.where(mask_np > 128)
        if len(xs) == 0:
            job_store[job_id].status = JobStatus.done
            job_store[job_id].result_image = image_to_b64(image)
            return
        x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())

        # 제품 이미지 있으면 CV 합성
        if req.product_image_base64:
            import tempfile, base64, os
            raw = base64.b64decode(req.product_image_base64)
            tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            tmp.write(raw)
            tmp.close()
            try:
                composited = processor.composite_products(
                    image=image,
                    products=[{"image_path": tmp.name, "region": (x1, y1, x2, y2), "category": ""}],
                )
            finally:
                os.unlink(tmp.name)
        else:
            composited = image

        # LoRA img2img 자연화
        processor.pipe.to("cuda")
        result = processor.refine_with_style(
            image=composited,
            style=req.style or "general",
            desk_bbox=(x1, y1, x2, y2),
            num_inference_steps=req.num_inference_steps,
            guidance_scale=req.guidance_scale,
        )
        processor.pipe.to("cpu")
        torch.cuda.empty_cache()

        job_store[job_id].status = JobStatus.done
        job_store[job_id].result_image = image_to_b64(result)
    except Exception as e:
        job_store[job_id].status = JobStatus.failed
        job_store[job_id].error = str(e)
        log.error("job_id=%s\n%s", job_id, traceback.format_exc())


# ── Generate (전체 파이프라인 단일 호출) ⚠️ 임시 스펙 ──────────
@app.post("/generate", response_model=GenerateResult)
async def run_generate(req: GenerateRequest):
    job_id = str(uuid.uuid4())
    job_store[job_id] = GenerateResult(job_id=job_id, status=JobStatus.pending)
    asyncio.get_event_loop().run_in_executor(executor, _run_generate, job_id, req)
    return job_store[job_id]


_REMOVAL_PROMPT = (
    "laptop. laptop computer. notebook computer. monitor. keyboard. mouse. "
    "mouse pad. mousepad. headset. cup. mug. book. books. book stack. "
    "notebook. notepad. paper. document. folder. file. binder. "
    "speaker. desk lamp. lamp. stand lamp. table lamp. desk light. "
    "cable. pen. pencil. pen holder. pencil cup. stationery. desk organizer. "
    "phone. tablet. controller. box. bottle. plant. potted plant. "
    "clock. digital clock. diffuser. perfume bottle. vase."
)

_FRONT_CATS = {"KEYBOARD", "MOUSE", "MOUSEPAD"}
_BACK_CATS  = {"MONITOR", "SPEAKER", "DESK_LAMP", "DESK_SHELF", "LAPTOP_STAND", "DECO", "CLOCK"}


def _build_occupied_mask_from_detection(detection: dict, img_w: int, img_h: int) -> np.ndarray:
    """detect_with_prompt 결과 → numpy occupied_mask (uint8)."""
    occupied = np.zeros((img_h, img_w), dtype=np.uint8)
    for mask_pil in detection.get("individual_masks", []):
        arr = np.array(mask_pil.convert("L"))
        occupied = cv2.bitwise_or(occupied, (arr > 127).astype(np.uint8) * 255)
    return occupied


def _run_generate(job_id: str, req: GenerateRequest):
    job_store[job_id].status = JobStatus.running
    try:
        from datetime import datetime
        _debug_dir = Path("outputs/debug") / datetime.now().strftime("%Y%m%d_%H%M%S")
        _debug_dir.mkdir(parents=True, exist_ok=True)
        print(f"[Generate] mode={req.mode}")

        image = b64_to_image(req.image_base64)
        img_w, img_h = image.size
        mode = req.mode  # RemoveMode enum

        remover = get_object_removal_processor()

        # ── Step 1: front-view 물체 감지 ───────────────────────────
        detection = remover.detect_with_prompt(
            image=image,
            prompt=_REMOVAL_PROMPT,
            max_area_ratio=0.40,
        )
        front_instances   = detection.get("individual_masks", [])
        front_detections  = detection.get("detections", [])
        job_store[job_id].num_removed = detection["num_objects"]

        # detection 로그
        print(f"[Removal] num_objects={detection.get('num_objects')}")
        for _det in front_detections:
            print(f"[Removal] label={_det.label}, score={_det.score:.3f}, bbox={_det.box_xyxy}")
        print(f"[Removal] front_instances={len(front_instances)}")

        # front detection overlay 저장
        from PIL import ImageDraw as _IDraw
        _ov = image.copy()
        _ov_draw = _IDraw.Draw(_ov)
        for _det in front_detections:
            _bx1, _by1, _bx2, _by2 = _det.box_xyxy
            _ov_draw.rectangle([_bx1, _by1, _bx2, _by2], outline=(255, 0, 0), width=3)
            _ov_draw.text((_bx1 + 4, _by1 + 4), f"{_det.label}:{_det.score:.2f}", fill=(255, 0, 0))
        _ov.save(_debug_dir / "front_detection_overlay.png")

        # combined remove mask 저장
        _combined_np = np.zeros((img_h, img_w), dtype=np.uint8)
        for _mp in front_instances:
            _arr = np.array(_mp.convert("L"))
            _combined_np = cv2.bitwise_or(_combined_np, (_arr > 127).astype(np.uint8) * 255)
        Image.fromarray(_combined_np).save(_debug_dir / "front_remove_mask.png")

        # ── 제거 전략 결정 ──────────────────────────────────────────
        removal_strategy = getattr(req, "removal_strategy", "combined")
        if mode == RemoveMode.add:
            removal_strategy = "none"
            print("[Removal] add 모드 — 기존 물체 제거 안 함")

        if removal_strategy != "none" and len(front_instances) == 0:
            print("[Removal WARNING] remove mode인데 감지된 제거 대상이 없습니다.")

        _combined_mask_pil = Image.fromarray(_combined_np)

        if removal_strategy == "none":
            current = image

        elif removal_strategy == "sequential":
            lama   = get_lama_processor()
            current = image
            for _mask_pil in front_instances:
                current = lama.inpaint(image=current, mask=_mask_pil)
            current.save(_debug_dir / "cleaned_front_sequential.png")
            torch.cuda.empty_cache()

        else:  # combined (default)
            lama              = get_lama_processor()
            _cleaned_combined = lama.inpaint(image=image, mask=_combined_mask_pil)
            _cleaned_combined.save(_debug_dir / "cleaned_front_combined.png")
            current = _cleaned_combined
            torch.cuda.empty_cache()

        current.save(_debug_dir / "cleaned_front.png")
        job_store[job_id].cleaned_image = image_to_b64(current)

        # ── Step 2→3: ControlNet 제품 생성 ────────────────────────
        # top-view 공간 분석은 calc_placements_from_available_space() 내부에서 수행
        cn_proc = get_controlnet_inpaint_processor()

        # products.csv로 width_mm/depth_mm 보완 + category 정규화
        products = enrich_products_from_csv(req.products)

        print(f"[Generate] products={len(products)}")
        for _p in products:
            print(f"  product: category={_p.category}, image_id={_p.image_id}, "
                  f"size={_p.width_mm}x{_p.depth_mm}")

        num_placed = 0
        run_errors: list[str] = []

        def _run_cn(p, x1, y1, x2, y2):
            nonlocal current, num_placed, run_errors
            if p.image_id is None:
                msg = f"{p.category}: image_id=None"
                run_errors.append(msg)
                print(f"  [_run_cn SKIP] {msg}")
                return
            prod_path = find_product_image(p.image_id)
            if prod_path is None:
                msg = f"{p.category}: product image not found for image_id={p.image_id}"
                run_errors.append(msg)
                print(f"  [_run_cn SKIP] {msg}")
                log.error("_run_cn SKIP %s", msg)
                return
            print(f"  [_run_cn] prod_path={prod_path}")
            try:
                prod_img = Image.open(prod_path)
                _raw_w, _raw_h = prod_img.size
                cat = normalize_category(p.category)
                _prod_debug_dir = _debug_dir / "products"
                _prod_debug_dir.mkdir(exist_ok=True)
                prod_img.save(_prod_debug_dir / f"{cat}_{p.image_id}_raw.png")
                prod_alpha = prepare_product_image_for_composite(prod_img)
                prod_alpha.save(_prod_debug_dir / f"{cat}_{p.image_id}_alpha.png")
                _alpha_cov = float((np.array(prod_alpha.getchannel("A")) > 127).mean())
                if _alpha_cov > 0.80:
                    print(f"  [WARNING] {cat} id={p.image_id}: alpha_coverage={_alpha_cov:.2f}"
                          f" — multi-object/lifestyle 이미지 의심, 단품 이미지로 교체 필요")

                # aspect ratio validation: 잘못된 제품 이미지 조기 차단
                _ar = prod_alpha.width / max(prod_alpha.height, 1)
                _ar_range = _CAT_ASPECT_VALID.get(cat)
                if _ar_range is not None:
                    _ar_min, _ar_max = _ar_range
                    if not (_ar_min <= _ar <= _ar_max):
                        msg = (f"{cat} id={p.image_id}: aspect_ratio={_ar:.2f} "
                               f"out of [{_ar_min}, {_ar_max}] — 잘못된 제품 이미지, skip")
                        run_errors.append(msg)
                        print(f"  [_run_cn SKIP] {msg}")
                        return

                # cv_composite 모드: _CV_ONLY_CATS 전체 CV 합성
                # controlnet 모드: 전 카테고리 generate_product() 경유
                _cv_only_set = _CV_ONLY_CATS if gen_mode == "cv_composite" else set()
                if cat in _cv_only_set:
                    current = composite_product_simple(current, prod_alpha, (x1, y1, x2, y2), category=cat)
                    current = _add_contact_shadow(current, (x1, y1, x2, y2), cat)
                    num_placed += 1
                    print(f"  [_run_cn CV] {cat} 합성 완료 (num_placed={num_placed})")
                    return

                # 밝기 체크: 어두운 이미지 → IP-Adapter 색감 인식 불가 → 비활성화
                _prod_rgb    = prod_img.convert("RGB")
                brightness   = float(np.array(_prod_rgb).mean())
                print(f"  [_run_cn] {cat} brightness={brightness:.1f}")

                if brightness < 40:
                    ip_scale = 0.0
                    print(f"  [_run_cn] {cat} 이미지 너무 어두움 → IP-Adapter 비활성화")
                elif cat == "DESK_SHELF":
                    ip_scale = 0.0
                elif cat == "MONITOR":
                    ip_scale = 0.10  # black screen 모드: 화면 콘텐츠 최소화
                elif cat == "SPEAKER":
                    ip_scale = 0.40
                elif cat == "DESK_LAMP":
                    ip_scale = 0.20
                elif cat in ("MOUSE", "MOUSEPAD"):
                    ip_scale = 0.65
                else:
                    ip_scale = 0.50

                # prod_alpha(RGBA) 전달 — generate_product 내부에서 letterbox/silhouette 처리
                mask           = _make_rect_mask(img_w, img_h, x1, y1, x2, y2)
                context_region = (0, img_h // 2, img_w, img_h) if cat in _FRONT_CATS else None
                print(f"  [_run_cn] {cat} ({x1},{y1},{x2},{y2}) ip_scale={ip_scale}")
                _debug_meta = {
                    "image_id":      p.image_id,
                    "width_mm":      getattr(p, "width_mm", None),
                    "depth_mm":      getattr(p, "depth_mm", None),
                    "height_mm":     getattr(p, "height_mm", None),
                    "product_raw_w": _raw_w,
                    "product_raw_h": _raw_h,
                }
                current = cn_proc.generate_product(
                    image=current, mask=mask, product_image=prod_alpha,
                    category=p.category, style=req.style.value,
                    context_region=context_region,
                    ip_adapter_scale=ip_scale,
                    debug_dir=_debug_dir / "products",
                    debug_meta=_debug_meta,
                )
                current = _add_contact_shadow(current, (x1, y1, x2, y2), cat)
                num_placed += 1
                print(f"  [_run_cn] {cat} 완료 (num_placed={num_placed})")
            except Exception as _e:
                msg = f"{p.category}: ControlNet generation failed: {_e}"
                run_errors.append(msg)
                print(f"  [_run_cn ERROR] {msg}")
                log.error("_run_cn ERROR %s\n%s", msg, traceback.format_exc())

        # ── 배치 위치 결정: top-view available space 우선, 없으면 _calc_regions ──
        placement_items:    list[dict] = []
        _unplaced_for_json: list[dict] = []

        if req.top_view_image_base64:
            print(f"[Generate] desk_width_mm={req.desk_width_mm} desk_depth_mm={req.desk_depth_mm}")
            top_image_for_place = b64_to_image(req.top_view_image_base64)
            _all_space_results = calc_placements_from_available_space(
                front_image=current,
                top_view_image=top_image_for_place,
                products=products,
                desk_width_mm=req.desk_width_mm,
                desk_depth_mm=req.desk_depth_mm,
                mode=mode,
                remover=remover,
                debug_dir=_debug_dir,
            )

            # scored (region not None) vs failed 분리
            _scored_items = [i for i in _all_space_results if i.get("region") is not None]
            _failed_items = [i for i in _all_space_results if i.get("region") is None]
            _failed_meta  = {id(i["product"]): i for i in _failed_items}

            # scored 내부 overlap 안전 제거 (grid 단계에서 이미 체크했으나 보험)
            _deduped: list[dict] = []
            for _item in _scored_items:
                if all(bbox_iou(_item["region"], _prev["region"]) < 0.25 for _prev in _deduped):
                    _deduped.append(_item)
                else:
                    print(f"[Placement SKIP] internal overlap: {normalize_category(_item['product'].category)} {_item['region']}")
            placement_items = _deduped

            # scoring 실패 제품 → _calc_regions fallback
            _failed_products = [i["product"] for i in _failed_items]
            if _failed_products:
                _desk_bbox_fb = _detect_desk_bbox(current)
                _raw_fallback = _calc_regions(img_w, img_h, _failed_products, _desk_bbox_fb, req.desk_width_mm)
                _added = 0
                for _fi in _raw_fallback:
                    _fi_cat  = normalize_category(_fi["product"].category)
                    _fi_meta = _failed_meta.get(id(_fi["product"]), {})
                    _fi.update({
                        "placement_source":    "fallback",
                        "fallback_reason":     _fi_meta.get("fallback_reason"),
                        "candidate_count":     _fi_meta.get("candidate_count", 0),
                        "score":               None,
                        "available_region_id": None,
                        "anchor_rx":           None,
                        "anchor_ry":           None,
                    })
                    if all(bbox_iou(_fi["region"], _prev["region"]) < 0.25 for _prev in placement_items):
                        placement_items.append(_fi)
                        _added += 1
                    else:
                        print(f"[Placement SKIP] overlap: {_fi_cat} {_fi['region']}")
                print(f"[Generate] space-fail {len(_failed_products)}개 → fallback {_added}개 추가")

            # scoring+fallback 모두 실패한 제품 → JSON에만 기록
            _placed_ids = {id(i["product"]) for i in placement_items}
            _unplaced_for_json = [i for i in _failed_items if id(i["product"]) not in _placed_ids]

            print(f"[Generate] available_space 배치: {len(placement_items)}개")

        if not placement_items:
            # fallback: 기존 _calc_regions + DINO 매칭
            desk_bbox = _detect_desk_bbox(current)
            if desk_bbox:
                print(f"[Generate] fallback desk_bbox: {desk_bbox}")
            else:
                print("[Generate] desk 감지 실패 → 이미지 비율 fallback")

            matched, unmatched = _match_products_to_detections(products, front_detections)
            print(f"[Generate] fallback 매칭: matched={len(matched)}개, unmatched={len(unmatched)}개")

            matched_back          = [i for i in matched   if normalize_category(i["product"].category) in _BACK_CATS]
            unmatched_back        = [p for p in unmatched if normalize_category(p.category)            in _BACK_CATS]
            matched_front_prods   = [i["product"] for i in matched   if normalize_category(i["product"].category) in _FRONT_CATS]
            unmatched_front_prods = [p            for p in unmatched if normalize_category(p.category)            in _FRONT_CATS]
            all_front_prods       = matched_front_prods + unmatched_front_prods

            for item in matched_back:
                p = item["product"]
                region = _region_from_detection_center(
                    img_w, img_h, item["region"], p, desk_bbox, req.desk_width_mm,
                )
                placement_items.append({"product": p, "region": region})

            for item in _calc_regions(img_w, img_h, unmatched_back, desk_bbox, req.desk_width_mm):
                placement_items.append(item)

            for item in _calc_regions(img_w, img_h, all_front_prods, desk_bbox, req.desk_width_mm):
                placement_items.append(item)

        # ── 배치 가능 제품 없으면 즉시 failed ──────────────────────
        print(f"[Generate] placement_items={len(placement_items)}")

        if not placement_items:
            job_store[job_id].status = JobStatus.failed
            job_store[job_id].error = (
                "배치 가능한 제품 영역이 없습니다. "
                "products.category, image_id, top_view_image_base64, "
                "desk_mask/available_space 분석 결과를 확인하세요."
                + ((" | " + " | ".join(run_errors[:5])) if run_errors else "")
            )
            return

        # ── 생성 순서 정렬: 큰/뒤쪽 제품 먼저 ─────────────────────────
        placement_items.sort(
            key=lambda item: _PLACEMENT_ORDER.get(normalize_category(item["product"].category), 999)
        )

        # ── 배치 위치 시각화 저장 ────────────────────────────────────
        from PIL import ImageDraw as _ID
        _dbg = current.copy()
        _draw = _ID.Draw(_dbg)
        for _item in placement_items:
            _x1, _y1, _x2, _y2 = _item["region"]
            _cat    = normalize_category(_item["product"].category)
            _score  = _item.get("score")
            _rid    = _item.get("available_region_id")
            _src    = "av" if _item.get("placement_source") == "available_space_scoring" else "fb"
            _label  = _cat
            if _score is not None:
                _label += f" s={_score:.2f}"
            if _rid is not None:
                _label += f" r{_rid}"
            _label += f" [{_src}]"
            _draw.rectangle([_x1, _y1, _x2, _y2], outline=(255, 0, 0), width=4)
            _draw.text((_x1 + 4, _y1 + 4), _label, fill=(255, 0, 0))
        _dbg.save(_debug_dir / "placement_debug.png")
        print(f"[Generate] 배치 시각화 저장: {_debug_dir}/placement_debug.png")

        # ── 항상 저장: placement_only 결과 ──────────────────────────
        _dbg.save(_debug_dir / "placement_only_result.png")

        # ── 항상 저장: cv_composite 결과 (GPU 불필요, 빠름) ──────────
        _cv_base = current.copy()
        for _cv_item in placement_items:
            _cv_p   = _cv_item["product"]
            _cv_cat = normalize_category(_cv_p.category)
            if _cv_p.image_id is None:
                continue
            _cv_path = find_product_image(_cv_p.image_id)
            if _cv_path is None:
                continue
            try:
                _cv_img  = Image.open(_cv_path)
                _cv_base = composite_product_simple(_cv_base, _cv_img, _cv_item["region"], category=_cv_cat)
            except Exception:
                pass
        _cv_base.save(_debug_dir / "cv_composite_result.png")
        print(f"[Generate] cv_composite 디버그 저장: {_debug_dir}/cv_composite_result.png")

        # ── 제품 목록 JSON 저장 ──────────────────────────────────────
        import json as _json
        _products_info = [
            {
                "category":               normalize_category(_it["product"].category),
                "image_id":               _it["product"].image_id,
                "region":                 list(_it["region"]),
                "width_px":               _it["region"][2] - _it["region"][0],
                "height_px":              _it["region"][3] - _it["region"][1],
                "placement_source":       _it.get("placement_source", "fallback"),
                "selected_reason":        _it.get("selected_reason"),
                "fallback_reason":        _it.get("fallback_reason"),
                "candidate_count":        _it.get("candidate_count"),
                "overlap_reject_count":   _it.get("overlap_reject_count"),
                "low_score_reject_count": _it.get("low_score_reject_count"),
                "score":                  _it.get("score"),
                "available_region_id":    _it.get("available_region_id"),
                "anchor_rx":              _it.get("anchor_rx"),
                "anchor_ry":              _it.get("anchor_ry"),
                "category_anchor_type":   "base_contact",
                "front_contact_x":        (_it["region"][0] + _it["region"][2]) // 2,
                "front_contact_y":        _it["region"][3],
                "applied_front_offset_x": 0,
                "applied_front_offset_y": _CONTACT_Y_OFFSET.get(
                    normalize_category(_it["product"].category), 0
                ),
            }
            for _it in placement_items
        ] + [
            {
                "category":               normalize_category(_it["product"].category),
                "image_id":               _it["product"].image_id,
                "region":                 None,
                "width_px":               None,
                "height_px":              None,
                "placement_source":       "unplaced",
                "selected_reason":        None,
                "fallback_reason":        _it.get("fallback_reason"),
                "candidate_count":        _it.get("candidate_count"),
                "overlap_reject_count":   _it.get("overlap_reject_count"),
                "low_score_reject_count": _it.get("low_score_reject_count"),
                "score":                  None,
                "available_region_id":    None,
                "anchor_rx":              None,
                "anchor_ry":              None,
            }
            for _it in _unplaced_for_json
        ]
        (_debug_dir / "products_list.json").write_text(
            _json.dumps(_products_info, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        gen_mode = getattr(req, "generation_mode", "controlnet")

        # ── placement_only 모드: bbox만 그려서 반환 ──────────────────
        if gen_mode == "placement_only":
            job_store[job_id].num_placed  = 0
            job_store[job_id].result_image = image_to_b64(_dbg)
            job_store[job_id].status       = JobStatus.done
            print("[Generate] placement_only 완료")
            return

        # ── cv_composite 모드: 제품 이미지 단순 합성 ────────────────
        if gen_mode == "cv_composite":
            for item in placement_items:
                p   = item["product"]
                cat = normalize_category(p.category)
                if p.image_id is None:
                    run_errors.append(f"{cat}: image_id=None")
                    continue
                prod_path = find_product_image(p.image_id)
                if prod_path is None:
                    run_errors.append(f"{cat}: image not found id={p.image_id}")
                    continue
                prod_img = Image.open(prod_path)
                current  = composite_product_simple(current, prod_img, item["region"], category=cat)
                num_placed += 1
                print(f"  [cv_composite] {cat} 합성 완료 (num_placed={num_placed})")

            current.save(_debug_dir / "cv_composite_result.png")
            if num_placed == 0:
                job_store[job_id].status = JobStatus.failed
                job_store[job_id].error  = "cv_composite 0개: " + " | ".join(run_errors[:5])
                return
            job_store[job_id].num_placed   = num_placed
            job_store[job_id].result_image = image_to_b64(current)
            job_store[job_id].status       = JobStatus.done
            print("[Generate] cv_composite 완료")
            return

        # ── controlnet 모드 ──────────────────────────────────────────
        for item in placement_items:
            p   = item["product"]
            cat = normalize_category(p.category)
            # E: DECO score <= 0 → 배치 기록은 유지하되 생성 skip
            if cat == "DECO" and (item.get("score") is None or item.get("score", 0) <= 0):
                print(f"  [Generate SKIP] DECO score={item.get('score')} ≤ 0 → 생성 제외")
                continue
            print(f"  [Generate] 처리: {p.category} image_id={p.image_id} region={item['region']}")
            _run_cn(p, *item["region"])

        print(f"[Generate] num_placed={num_placed}")

        # ── 실제 생성 0개면 failed ──────────────────────────────────
        if num_placed == 0:
            job_store[job_id].status = JobStatus.failed
            job_store[job_id].error = (
                "제품 생성이 0개 수행되었습니다: "
                + (" | ".join(run_errors[:5]) if run_errors else "알 수 없는 오류")
            )
            return

        job_store[job_id].num_placed   = num_placed
        job_store[job_id].result_image = image_to_b64(current)
        job_store[job_id].status       = JobStatus.done

    except Exception as e:
        job_store[job_id].status = JobStatus.failed
        job_store[job_id].error  = str(e)
        log.error("job_id=%s\n%s", job_id, traceback.format_exc())


# ── Segment (SAM-2 포인트 세그멘테이션) ─────────────────────────
@app.post("/segment", response_model=SegmentResult)
async def run_segment(req: SegmentRequest):
    job_id = str(uuid.uuid4())
    job_store[job_id] = SegmentResult(job_id=job_id, status=JobStatus.pending)
    asyncio.get_event_loop().run_in_executor(executor, _run_segment, job_id, req)
    return job_store[job_id]


def _run_segment(job_id: str, req: SegmentRequest):
    job_store[job_id].status = JobStatus.running
    try:
        processor = get_sam2_processor()
        image = b64_to_image(req.image_base64)
        w, h = image.size
        px = int(req.point_x * w)
        py = int(req.point_y * h)
        result = processor.segment(image, points=[[px, py]], point_labels=[req.label])
        job_store[job_id].mask_base64 = result["mask"]
        job_store[job_id].overlay_base64 = result["overlay"]
        job_store[job_id].status = JobStatus.done
    except Exception as e:
        job_store[job_id].status = JobStatus.failed
        job_store[job_id].error = str(e)
        log.error("job_id=%s\n%s", job_id, traceback.format_exc())
