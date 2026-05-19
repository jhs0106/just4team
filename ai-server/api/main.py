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

Path("logs").mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.ERROR,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("logs/server_errors.log", encoding="utf-8"),
    ],
)
log = logging.getLogger("ai-server")

from .models import (
    StyleName, JobStatus,
    ObjectRemovalRequest, ObjectRemovalResult,
    ProductPlaceRequest, ProductPlaceResult,
    SegmentRequest, SegmentResult,
    GenerateRequest, GenerateResult,
)
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

# 45도 앵글 뷰에서 제품의 높이/너비 비율
# 수직 제품(모니터·스탠드·램프)은 크고, 수평 제품(키보드·마우스패드)은 작음
_FRONT_HEIGHT_RATIO = {
    "MONITOR":      0.70,
    "KEYBOARD":     0.22,   # 45° 뷰에서 깊이 짧음
    "MOUSE":        0.90,
    "MOUSEPAD":     0.30,
    "SPEAKER":      1.20,
    "DESK_LAMP":    2.00,
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
    "MONITOR":      0.42,
    "SPEAKER":      0.09,
    "DESK_LAMP":    0.06,
    "DESK_SHELF":   0.42,
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
        cat = p.category.upper()
        found = None
        for i, det in enumerate(available):
            det_cat = _DINO_LABEL_TO_CATEGORY.get(det.label.lower(), "").upper()
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


_CATEGORY_PROMPT = {
    "KEYBOARD":     "mechanical keyboard on desk mat, top view",
    "MOUSE":        "wireless mouse on desk, top view",
    "MONITOR":      "monitor on desk, front view",
    "SPEAKER":      "desktop speaker on desk",
    "DESK_LAMP":    "modern desk lamp on desk",
    "DESK_SHELF":   "monitor riser shelf on desk",
    "LAPTOP_STAND": "laptop stand on desk",
    "DECO":         "small desk decoration",
    "CLOCK":        "minimalist desk clock",
}


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


def _order_corners(pts: np.ndarray) -> np.ndarray:
    # 4개 코너를 TL → TR → BR → BL 순으로 정렬
    rect = np.zeros((4, 2), dtype=np.float32)
    s    = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).flatten()
    rect[0] = pts[np.argmin(s)]     # TL: x+y 최소
    rect[2] = pts[np.argmax(s)]     # BR: x+y 최대
    rect[1] = pts[np.argmin(diff)]  # TR: y-x 최소
    rect[3] = pts[np.argmax(diff)]  # BL: y-x 최대
    return rect


def _detect_desk_corners(image: Image.Image) -> np.ndarray | None:
    # DINO bbox → 4코너(TL,TR,BR,BL) 반환. bbox 코너를 그대로 사용 (프로토타입)
    bbox = _detect_desk_bbox(image)
    if bbox is None:
        return None
    x1, y1, x2, y2 = bbox
    pts = np.array([[x1,y1],[x2,y1],[x2,y2],[x1,y2]], dtype=np.float32)
    return _order_corners(pts)


def _calc_top_view_regions(top_corners: np.ndarray, products) -> list:
    # 탑뷰 코너 기준 카테고리별 배치 사각형 계산 → [{"product", "top_rect_corners": (4×2)}]
    import cv2
    tl, tr, br, bl = top_corners

    desk_w = float(np.linalg.norm(tr - tl) + np.linalg.norm(br - bl)) / 2
    desk_h = float(np.linalg.norm(bl - tl) + np.linalg.norm(br - tr)) / 2

    # 카테고리별 배치 (cx_비율, cy_비율, w_비율, h_비율) — 탑뷰 기준
    _TOP_POS = {
        "MONITOR":      (0.50, 0.12, 0.48, 0.14),
        "KEYBOARD":     (0.38, 0.52, 0.38, 0.14),
        "MOUSE":        (0.66, 0.53, 0.08, 0.11),
        "SPEAKER":      (0.15, 0.18, 0.10, 0.15),
        "DESK_LAMP":    (0.82, 0.12, 0.07, 0.18),
        "DESK_SHELF":   (0.50, 0.28, 0.50, 0.10),
        "LAPTOP_STAND": (0.35, 0.38, 0.22, 0.16),
        "DECO":         (0.18, 0.50, 0.07, 0.09),
        "CLOCK":        (0.85, 0.48, 0.08, 0.10),
    }

    def _interp(r_x, r_y):
        # 탑뷰 정규화 좌표(0~1) → 실제 픽셀 (4코너 보간)
        top  = tl + r_x * (tr - tl)
        bot  = bl + r_x * (br - bl)
        return top + r_y * (bot - top)

    regions = []
    for p in products:
        cat = p.category.upper()
        pos = _TOP_POS.get(cat)
        if pos is None:
            continue
        cx_r, cy_r, w_r, h_r = pos
        hw = w_r / 2
        hh = h_r / 2
        # 4코너를 정규화 좌표로 정의
        corners = np.array([
            _interp(cx_r - hw, cy_r - hh),
            _interp(cx_r + hw, cy_r - hh),
            _interp(cx_r + hw, cy_r + hh),
            _interp(cx_r - hw, cy_r + hh),
        ], dtype=np.float32)
        regions.append({"product": p, "top_rect_corners": corners})

    return regions


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
        dx1, dy1 = 0, int(img_h * 0.45)
        dx2, dy2 = img_w, img_h

    DW = dx2 - dx1
    DH = dy2 - dy1
    cx = (dx1 + dx2) // 2

    # 전면(카메라 가까운 쪽): 책상 표면 78% 지점
    front_y = dy1 + int(DH * 0.78)
    # 후면(벽 쪽): 책상 상단 경계 — 후면 제품은 위쪽(벽)으로 솟아오름
    back_top = dy1

    def clip(x1, y1, x2, y2):
        return max(0, x1), max(0, y1), min(img_w - 1, x2), min(img_h - 1, y2)

    def perspective_scale(center_y: int) -> float:
        ratio = (center_y - dy1) / max(DH, 1)
        return 0.60 + 0.40 * ratio

    def product_pixel_size(p) -> tuple[int, int]:
        cat     = p.category.upper()
        w_mm    = getattr(p, "width_mm", None) or _CATEGORY_DIMS_MM.get(cat, (100, 100))[0]
        h_ratio = _FRONT_HEIGHT_RATIO.get(cat, 0.80)

        if desk_width_mm and desk_width_mm > 0:
            pw = int(w_mm * DW / desk_width_mm)
        else:
            # 책상 너비 대비 카테고리별 비율로 fallback
            pw = int(DW * _DESK_W_RATIO.get(cat, 0.15))

        ph = int(pw * h_ratio)
        return pw, ph

    regions = []
    for p in products:
        cat = p.category.upper()
        if cat not in _CATEGORY_DIMS_MM:
            continue

        base_pw, base_ph = product_pixel_size(p)

        if cat in ("KEYBOARD", "MOUSE", "MOUSEPAD"):
            # 전면 제품: 책상 앞쪽, 원근 적용
            ps = perspective_scale(front_y)
            pw = int(base_pw * ps)
            ph = base_ph

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
            ph = base_ph

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


def _run_generate(job_id: str, req: GenerateRequest):
    job_store[job_id].status = JobStatus.running
    try:
        image = b64_to_image(req.image_base64)
        img_w, img_h = image.size

        # ── Step 1: 기존 물체 제거 ──────────────────────────────
        _REMOVAL_PROMPT = (
            "laptop. laptop computer. notebook computer. monitor. keyboard. mouse. "
            "mouse pad. mousepad. headset. cup. mug. book. notebook. speaker. "
            "desk lamp. lamp. cable. pen. pencil. phone. tablet. controller. box. bottle."
        )
        remover = get_object_removal_processor()
        detection = remover.detect_with_prompt(
            image=image,
            prompt=_REMOVAL_PROMPT,
            max_area_ratio=0.40,
        )
        job_store[job_id].num_removed = detection["num_objects"]

        if detection["num_objects"] > 0:
            lama = get_lama_processor()
            current = image
            for mask_pil in detection["individual_masks"]:
                current = lama.inpaint(image=current, mask=mask_pil)
            torch.cuda.empty_cache()
        else:
            current = image

        job_store[job_id].cleaned_image = image_to_b64(current)

        # ── Step 2+3: ControlNet Depth/Canny + IP-Adapter-Plus로 제품 생성 ──
        cn_proc = get_controlnet_inpaint_processor()

        step1_detections = detection.get("detections", [])
        matched, unmatched = _match_products_to_detections(req.products, step1_detections)
        print(f"[Generate] 매칭: {len(matched)}개 감지위치, {len(unmatched)}개 fallback")

        num_placed = 0

        def _run_cn(p, x1, y1, x2, y2):
            nonlocal current, num_placed
            if p.image_id is None:
                return
            prod_path = PRODUCT_IMAGE_DIR / f"{p.image_id}.png"
            if not prod_path.exists():
                return
            prod_img = Image.open(prod_path).convert("RGB")
            mask = _make_rect_mask(img_w, img_h, x1, y1, x2, y2)

            # 전면 제품: 이미지 하단 절반을 context로 사용 → 제품 영역이 512에서 더 크게 표현됨
            # 후면 제품: 전체 이미지 사용 → LoRA가 장면 전체 보고 스타일 적용
            cat = p.category.upper()
            if cat in ("KEYBOARD", "MOUSE", "MOUSEPAD"):
                context_region = (0, img_h // 2, img_w, img_h)
            else:
                context_region = None

            current = cn_proc.generate_product(
                image=current,
                mask=mask,
                product_image=prod_img,
                category=p.category,
                style=req.style.value,
                context_region=context_region,
            )
            num_placed += 1

        desk_bbox = _detect_desk_bbox(current)
        if desk_bbox:
            print(f"[Generate] 책상 감지 성공: {desk_bbox}")
        else:
            print("[Generate] 책상 감지 실패 → 이미지 비율 fallback")

        # 전면/후면 카테고리 분류
        _FRONT_CATS = {"KEYBOARD", "MOUSE", "MOUSEPAD"}
        _BACK_CATS  = {"MONITOR", "SPEAKER", "DESK_LAMP", "DESK_SHELF", "LAPTOP_STAND", "DECO", "CLOCK"}

        # 후면 제품: DINO 매칭 위치 사용 (있으면)
        matched_back   = [i for i in matched   if i["product"].category.upper() in _BACK_CATS]
        unmatched_back = [p for p in unmatched if p.category.upper()            in _BACK_CATS]

        # 전면 제품: DINO 위치 무시 → 항상 _calc_regions() front_y 사용
        matched_front_prods  = [i["product"] for i in matched   if i["product"].category.upper() in _FRONT_CATS]
        unmatched_front_prods= [p            for p in unmatched if p.category.upper()            in _FRONT_CATS]
        all_front_prods = matched_front_prods + unmatched_front_prods

        # ── 후면 먼저 생성 ─────────────────────────────────────────
        for item in matched_back:
            p = item["product"]
            region = _region_from_detection_center(
                img_w, img_h, item["region"], p, desk_bbox, req.desk_width_mm,
            )
            print(f"  [Resize] {p.category} DINO={item['region']} → {region}")
            _run_cn(p, *region)

        if unmatched_back:
            for item in _calc_regions(img_w, img_h, unmatched_back, desk_bbox, req.desk_width_mm):
                _run_cn(item["product"], *item["region"])

        # ── 전면 나중에 생성 (책상 앞줄 고정 위치) ────────────────
        if all_front_prods:
            for item in _calc_regions(img_w, img_h, all_front_prods, desk_bbox, req.desk_width_mm):
                _run_cn(item["product"], *item["region"])

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
