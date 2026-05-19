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
    import numpy as np
    from PIL import ImageDraw
    mask = Image.new("L", (img_w, img_h), 0)
    ImageDraw.Draw(mask).rectangle([x1, y1, x2, y2], fill=255)
    return mask


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
    # 실제 치수(mm) 기반 배치 계산: px_per_mm = DW / desk_width_mm → 제품 픽셀 크기
    # desk_width_mm 없으면 카테고리 기본 비율로 fallback
    if desk_bbox:
        dx1, dy1, dx2, dy2 = desk_bbox
    else:
        dx1, dy1 = 0, int(img_h * 0.45)
        dx2, dy2 = img_w, img_h

    DW = dx2 - dx1
    DH = dy2 - dy1
    cx = (dx1 + dx2) // 2

    front_y = dy1 + int(DH * 0.72)
    back_y  = dy1 + int(DH * 0.08)

    def clip(x1, y1, x2, y2):
        return max(dx1, x1), max(dy1, y1), min(dx2 - 1, x2), min(dy2 - 1, y2)

    def perspective_scale(center_y: int) -> float:
        ratio = (center_y - dy1) / max(DH, 1)
        return 0.60 + 0.40 * ratio

    def product_pixel_size(p) -> tuple[int, int]:
        # CSV metadata 치수 우선, 없으면 카테고리 기본값
        cat = p.category.upper()
        w_mm = getattr(p, "width_mm", None)
        d_mm = getattr(p, "depth_mm", None)

        if w_mm is None or d_mm is None:
            dims = _CATEGORY_DIMS_MM.get(cat, (100, 100))
            w_mm, d_mm = dims

        if desk_width_mm and desk_width_mm > 0:
            # 실제 치수 → 픽셀 변환
            px_per_mm = DW / desk_width_mm
            pw = int(w_mm * px_per_mm)
            # 깊이 방향은 원근 보정: depth_mm / desk_depth 비율을 DH에 매핑
            # desk depth는 통상 desk_width * 0.5 (1200x600mm 등)
            desk_depth_mm = desk_width_mm * 0.5
            ph = int(d_mm / desk_depth_mm * DH)
        else:
            # fallback: 책상 픽셀 면적 대비 비율로 추정
            ref_w = _CATEGORY_DIMS_MM.get(cat, (100, 100))[0]
            ref_d = _CATEGORY_DIMS_MM.get(cat, (100, 100))[1]
            pw = int(DW * w_mm / max(ref_w, 1) * 0.42)
            ph = int(DH * d_mm / max(ref_d, 1) * 0.22)

        return pw, ph

    regions = []
    for p in products:
        cat = p.category.upper()
        if cat not in _CATEGORY_DIMS_MM:
            continue

        if cat in ("KEYBOARD", "MOUSE", "LAPTOP_STAND", "MOUSEPAD"):
            center_y = front_y
        else:
            center_y = back_y + int(DH * 0.25)

        ps = perspective_scale(center_y)
        base_pw, base_ph = product_pixel_size(p)
        pw = int(base_pw * ps)
        ph = int(base_ph * ps)

        if cat == "KEYBOARD":
            x1 = cx - pw // 2;              x2 = x1 + pw
            y2 = front_y;                    y1 = y2 - ph
        elif cat == "MOUSEPAD":
            x1 = cx - pw // 2;              x2 = x1 + pw
            y2 = front_y + int(DH * 0.05);  y1 = y2 - ph
        elif cat == "MOUSE":
            x1 = cx + int(DW * 0.22);       x2 = x1 + pw
            y2 = front_y + int(DH * 0.03);  y1 = y2 - ph
        elif cat == "MONITOR":
            x1 = cx - pw // 2;              x2 = x1 + pw
            y1 = back_y;                     y2 = y1 + ph
        elif cat == "SPEAKER":
            x2 = cx - int(DW * 0.28);       x1 = x2 - pw
            y1 = back_y + int(DH * 0.05);   y2 = y1 + ph
        elif cat == "DESK_LAMP":
            x1 = cx + int(DW * 0.32);       x2 = x1 + pw
            y1 = back_y;                     y2 = y1 + ph
        elif cat == "DESK_SHELF":
            x1 = cx - pw // 2;              x2 = x1 + pw
            y1 = back_y + int(DH * 0.18);   y2 = y1 + ph
        elif cat == "LAPTOP_STAND":
            x2 = cx - int(DW * 0.14);       x1 = x2 - pw
            y2 = front_y - int(DH * 0.05);  y1 = y2 - ph
        else:
            x1 = cx + int(DW * 0.36);       x2 = x1 + pw
            y1 = back_y + int(DH * 0.18);   y2 = y1 + ph

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
            "monitor. keyboard. mouse. mouse pad. mousepad. headset. "
            "cup. mug. book. notebook. speaker. desk lamp. lamp. cable. "
            "pen. pencil. phone. tablet. controller. box. bottle."
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

        # ── Step 2: 제품 합성 ─────────────────────────────────────────
        inpainter = get_product_inpaint_processor()
        inpainter.pipe.to("cuda")

        if req.top_view_image_base64:
            # ── 탑뷰 Homography 기반 배치 ──────────────────────────────
            top_img       = b64_to_image(req.top_view_image_base64)
            top_corners   = _detect_desk_corners(top_img)
            front_corners = _detect_desk_corners(current)

            if top_corners is not None and front_corners is not None:
                print(f"[Generate] 탑뷰 코너: {top_corners.tolist()}")
                print(f"[Generate] 프론트뷰 코너: {front_corners.tolist()}")

                H, _  = cv2.findHomography(top_corners, front_corners)
                top_regions = _calc_top_view_regions(top_corners, req.products)

                composite_items = []
                for item in top_regions:
                    p = item["product"]
                    if p.image_id is None:
                        continue
                    path = PRODUCT_IMAGE_DIR / f"{p.image_id}.png"
                    if not path.exists():
                        continue
                    top_c   = item["top_rect_corners"].reshape(1, -1, 2)
                    front_c = cv2.perspectiveTransform(top_c, H).reshape(-1, 2)
                    composite_items.append({
                        "image_path":    str(path),
                        "front_corners": front_c,
                        "category":      p.category,
                    })

                composited = inpainter.composite_products_homography(current, composite_items)
                print(f"[Generate] Homography 합성 완료 — 제품 {len(composite_items)}개")
            else:
                print("[Generate] 코너 감지 실패 → 비율 기반 배치 fallback")
                regions = _calc_regions(img_w, img_h, req.products, None, req.desk_width_mm)
                composite_items = []
                for item in regions:
                    p = item["product"]
                    if p.image_id is not None:
                        path = PRODUCT_IMAGE_DIR / f"{p.image_id}.png"
                        if path.exists():
                            composite_items.append({
                                "image_path": str(path),
                                "region":     item["region"],
                                "category":   p.category,
                            })
                composited = inpainter.composite_products(current, composite_items)
        else:
            # ── bbox 기반 배치 fallback ────────────────────────────────
            desk_bbox = _detect_desk_bbox(current)
            if desk_bbox:
                print(f"[Generate] 책상 감지 성공: {desk_bbox}")
            else:
                desk_bbox = None
                print("[Generate] 책상 감지 실패 → fallback 사용")

            regions = _calc_regions(img_w, img_h, req.products, desk_bbox, req.desk_width_mm)
            composite_items = []
            for item in regions:
                p = item["product"]
                if p.image_id is not None:
                    path = PRODUCT_IMAGE_DIR / f"{p.image_id}.png"
                    if path.exists():
                        composite_items.append({
                            "image_path": str(path),
                            "region":     item["region"],
                            "category":   p.category,
                        })
            composited = inpainter.composite_products(current, composite_items)

        job_store[job_id].composited_image = image_to_b64(composited)

        # ── Step 3: 제품 영역별 SD img2img + LoRA 자연화 ──────────────
        # composite_items에서 영역 정보 추출 (region 또는 front_corners)
        refine_regions = []
        for item in composite_items:
            cat = item.get("category", "")
            if "region" in item:
                refine_regions.append({"region": item["region"], "category": cat})
            elif "front_corners" in item:
                fc = np.array(item["front_corners"])
                bbox = (
                    int(fc[:, 0].min()), int(fc[:, 1].min()),
                    int(fc[:, 0].max()), int(fc[:, 1].max()),
                )
                refine_regions.append({"region": bbox, "category": cat})

        if refine_regions:
            print(f"[Generate] Step 3 — 영역별 SD refinement {len(refine_regions)}개 시작")
            current = inpainter.refine_per_region(
                image=composited,
                regions=refine_regions,
                style=req.style.value,
            )
        else:
            current = composited

        inpainter.pipe.to("cpu")
        torch.cuda.empty_cache()

        job_store[job_id].num_placed   = len(composite_items)
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
