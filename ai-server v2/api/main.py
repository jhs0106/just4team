import uuid
import asyncio
import logging
import traceback
import torch
import numpy as np
import cv2
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor
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
    GenerateRequest, GenerateResult, RecommendAndGenerateRequest,
)
from .object_removal_processor import get_object_removal_processor
from .lama_processor import get_lama_processor
from .product_inpaint_processor import get_product_inpaint_processor
from .controlnet_inpaint_processor import get_controlnet_inpaint_processor
# harmonization_processor: 2026-05-23 실험 → 환경 hallucination 문제로 폐기.
# 파일은 보존하지만 import하지 않음. 자세한 내용: docs/ARCHITECTURE.md §4.3
from .sam2_processor import get_sam2_processor
from .config import (
    _CV_ONLY_CATS, _CAT_ASPECT_VALID, _PLACEMENT_ORDER,
    _CONTACT_Y_OFFSET, _FRONT_CATS, _BACK_CATS, _REMOVAL_PROMPT,
)
from .utils import (
    b64_to_image, image_to_b64,
    normalize_category, find_product_image, enrich_products_from_db,
)
from .composite import (
    prepare_product_image_for_composite, composite_product_simple,
    composite_one_with_silhouette,
    _add_shadows, _detect_desk_bbox, _calc_regions,
)
from .placement import (
    calc_placements_from_available_space, bbox_iou,
    _match_products_to_detections, _region_from_detection_center,
    _make_rect_mask, _overlap_threshold,
)

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
            top_view  = b64_to_image(req.top_view_image_base64)
            detection = remover.detect_from_top_view(front_view=image, top_view=top_view)
        else:
            detection = remover.detect_and_mask(image=image, max_area_ratio=req.max_area_ratio)

        job_store[job_id].mask_image        = detection["mask"]
        job_store[job_id].detection_overlay = detection["overlay"]
        job_store[job_id].num_objects       = detection["num_objects"]

        if detection["num_objects"] == 0:
            job_store[job_id].cleaned_image = image_to_b64(image)
            job_store[job_id].status = JobStatus.done
            return

        lama    = get_lama_processor()
        current = image
        for mask_pil in detection["individual_masks"]:
            current = lama.inpaint(image=current, mask=mask_pil)
        torch.cuda.empty_cache()

        job_store[job_id].cleaned_image = image_to_b64(current)
        job_store[job_id].status = JobStatus.done
    except Exception as e:
        job_store[job_id].status = JobStatus.failed
        job_store[job_id].error  = str(e)
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
        processor = get_product_inpaint_processor()
        image     = b64_to_image(req.image_base64)
        mask_np   = np.array(b64_to_image(req.mask_base64).convert("L"))
        ys, xs    = np.where(mask_np > 128)
        if len(xs) == 0:
            job_store[job_id].status       = JobStatus.done
            job_store[job_id].result_image = image_to_b64(image)
            return
        x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())

        if req.product_image_base64:
            import tempfile, base64, os
            raw = base64.b64decode(req.product_image_base64)
            tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            tmp.write(raw); tmp.close()
            try:
                composited = processor.composite_products(
                    image=image,
                    products=[{"image_path": tmp.name, "region": (x1, y1, x2, y2), "category": ""}],
                )
            finally:
                os.unlink(tmp.name)
        else:
            composited = image

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

        job_store[job_id].status       = JobStatus.done
        job_store[job_id].result_image = image_to_b64(result)
    except Exception as e:
        job_store[job_id].status = JobStatus.failed
        job_store[job_id].error  = str(e)
        log.error("job_id=%s\n%s", job_id, traceback.format_exc())


# ── Generate (전체 파이프라인 단일 호출) ─────────────────────────────
@app.post("/generate", response_model=GenerateResult)
async def run_generate(req: GenerateRequest):
    job_id = str(uuid.uuid4())
    job_store[job_id] = GenerateResult(job_id=job_id, status=JobStatus.pending)
    asyncio.get_event_loop().run_in_executor(executor, _run_generate, job_id, req)
    return job_store[job_id]


# ── Recommend + Generate (추천 서버 호출 → 생성까지 한 번에) ───────────
# 사용자는 theme + budget + 책상 사진만 보내면 됨. 내부에서:
#   1. recommendation 서버(:8001) 호출해 setup(5종 제품) 받음
#   2. recommendation_bridge로 GenerateRequest 변환
#   3. 추천 제품 + persona prompt로 NanoBanana API 이미지 생성
import os
import requests as _requests

_RECOMMENDATION_API_URL = os.getenv("RECOMMENDATION_API_URL", "http://127.0.0.1:8001")


@app.post("/recommend-and-generate", response_model=GenerateResult)
async def recommend_and_generate(req: RecommendAndGenerateRequest):
    from .adapters.recommendation_bridge import setup_to_generate_request

    # 1. top-view 분석으로 카테고리별 max size 사전 계산 — 가용 공간에 맞는 제품만 추천되게
    space_constraints: dict = {}
    if req.top_view_image_base64 and req.desk_width_mm and req.desk_depth_mm:
        try:
            from .placement import analyze_top_view_only, compute_size_constraints_from_space
            tv_img = b64_to_image(req.top_view_image_base64)
            _space_info = analyze_top_view_only(
                tv_img, req.desk_width_mm, req.desk_depth_mm, mode=req.mode,
            )
            if _space_info:
                space_constraints = compute_size_constraints_from_space(_space_info)
                print(f"[Recommend] space_constraints={space_constraints}")
        except Exception as e:
            print(f"[Recommend] space_constraints 계산 실패: {e} — 제약 없이 추천 진행")

    # 2. 추천 서버 호출 — 정면 사진 + 가용 공간 제약 전달
    try:
        rec_resp = _requests.post(
            f"{_RECOMMENDATION_API_URL}/recommend",
            json={
                "theme":             req.theme,
                "budget":            req.budget,
                "image_base64":      req.image_base64,
                "space_constraints": space_constraints,
            },
            timeout=180,
        )
    except _requests.RequestException as e:
        raise HTTPException(503, f"추천 서버 연결 실패: {e}")

    if rec_resp.status_code >= 400:
        # 추천 서버 detail(JSON.detail) 또는 raw body를 그대로 노출 — 디버깅 가능하도록
        try:
            _detail = rec_resp.json().get("detail") or rec_resp.text
        except Exception:
            _detail = rec_resp.text
        raise HTTPException(
            502,
            f"추천 서버 {rec_resp.status_code}: {_detail}",
        )

    setup = rec_resp.json().get("setup")
    if not setup:
        raise HTTPException(502, "추천 서버 응답에 setup 없음")

    # 2a. top_view 처리: 모드 무관 항상 필수.
    # 사용자가 안 주면 즉시 400 에러 (default 폴백 X — 다른 책상 분석으로 인한 잘못된 결과 방지)
    _top_view_b64 = req.top_view_image_base64
    if not _top_view_b64:
        raise HTTPException(
            400,
            "top_view 사진이 필요합니다. 모든 모드에서 사용자 책상의 top-view 사진이 필수입니다.",
        )
    _top_view_source = "user_provided"

    # desk_width_mm/depth_mm도 필수 (사진만으론 실측 mm 불가)
    if req.desk_width_mm is None or req.desk_depth_mm is None:
        raise HTTPException(
            400,
            "desk_width_mm과 desk_depth_mm은 필수입니다. (사용자가 입력한 cm × 10)",
        )

    # 2b. setup → GenerateRequest 변환 (style_mapper가 theme 자동 인식)
    try:
        gen_req = setup_to_generate_request(
            setup                = setup,
            color_text           = req.theme,
            theme_text           = req.theme,
            desk_image_b64       = req.image_base64,
            desk_width_mm        = req.desk_width_mm,
            desk_depth_mm        = req.desk_depth_mm,
            top_view_image_b64   = _top_view_b64,
            mode                 = req.mode,
            generation_mode      = req.generation_mode,
            removal_strategy     = req.removal_strategy,
            verify_images        = True,
            on_missing           = "skip",
        )
        gen_req.top_view_source = _top_view_source
        # 빈 책상 모드용 클릭 좌표 전달 (SAM2 prompt로 사용)
        gen_req.desk_click_x = req.desk_click_x
        gen_req.desk_click_y = req.desk_click_y
    except (ValueError, FileNotFoundError) as e:
        raise HTTPException(400, f"setup → GenerateRequest 변환 실패: {e}")

    # 2c. setup → JSP 표시용 제품 정보 추출 (ai-server는 image_id만 쓰지만 JSP는 가격/링크/이미지 필요)
    from .models import RecommendedProduct
    from .adapters.recommendation_bridge import _CATEGORY_MAP
    _products_meta: list[RecommendedProduct] = []
    for _rec_cat, _item in (setup.get("items") or {}).items():
        _ai_cat = _CATEGORY_MAP.get(_rec_cat)
        if _ai_cat is None:
            continue
        _products_meta.append(RecommendedProduct(
            category    = _ai_cat,
            name        = str(_item.get("title") or _ai_cat),
            image_id    = int(_item["id"]) if _item.get("id") else None,
            price       = int(_item["lprice"]) if _item.get("lprice") else None,
            image_url   = _item.get("image_url"),
            product_url = _item.get("product_url") or _item.get("link"),
        ))

    # 3. NanoBanana 생성 호출 + job_store에 products 정보 attach
    job_id = str(uuid.uuid4())
    job_store[job_id] = GenerateResult(
        job_id=job_id,
        status=JobStatus.pending,
    )

    asyncio.get_event_loop().run_in_executor(
        executor,
        _run_nanobanana_generate,
        job_id,
        req,
        setup,
        _products_meta,
        space_constraints,
    )

    return job_store[job_id]
def _run_nanobanana_generate(
        job_id: str,
        req: RecommendAndGenerateRequest,
        setup: dict,
        products_meta: list,
        space_constraints: dict,
):
    job_store[job_id].status = JobStatus.running

    try:
        from .nanobanana_processor import get_nanobanana_processor
        from .adapters.recommendation_bridge import _CATEGORY_MAP

        processor = get_nanobanana_processor()

        products_for_generation = []

        for rec_cat, item in (setup.get("items") or {}).items():
            ai_cat = _CATEGORY_MAP.get(rec_cat)

            if ai_cat is None:
                continue

            product_id = item.get("id")

            products_for_generation.append({
                "id": int(product_id) if product_id else None,
                "image_id": int(product_id) if product_id else None,
                "category": ai_cat,
                "recommendation_category": rec_cat,
                "name": str(item.get("title") or ai_cat),
                "price": int(item["lprice"]) if item.get("lprice") else None,
                "image_url": item.get("image_url"),
                "product_url": item.get("product_url") or item.get("link"),
                "brand": item.get("brand"),
                "mall_name": item.get("mallName"),
            })

        mode_value = req.mode.value if hasattr(req.mode, "value") else str(req.mode)

        generation = processor.generate(
            image_base64=req.image_base64,
            theme=req.theme,
            mode=mode_value,
            products=products_for_generation,
            space_constraints=space_constraints,
        )

        job_store[job_id].result_image = generation["result_image_base64"]
        job_store[job_id].products = products_meta
        job_store[job_id].num_placed = len(products_for_generation)
        job_store[job_id].status = JobStatus.done

        job_store[job_id].debug = {
            "generator": "nanobanana",
            "prompt": generation.get("prompt"),
            "space_constraints": space_constraints,
        }

    except Exception as e:
        job_store[job_id].status = JobStatus.failed
        job_store[job_id].error = str(e)
        log.error("job_id=%s\n%s", job_id, traceback.format_exc())

def extract_desk_top_band(mask_np: np.ndarray) -> tuple | None:
    # SAM2 mask에서 책상 윗면 후보 영역 추출.
    # 최종 sanity 검증(validate_tabletop_bbox)은 호출자에서 수행 — 여기선 후보만 생성.
    #
    # 알고리즘:
    #   1. derivative-based: row 폭 가장 큰 감소 = 다리 시작 직전 row → 윗면 후보
    #   2. 후보 band가 mask 전체 bbox 대비 너무 작으면(<25% 또는 <30px) Otsu fallback
    #   3. Otsu도 작으면 mask full bbox 그대로 사용 (validate_tabletop_bbox가 최종 판단)
    #
    # 회귀 fix (2026-05-26): 옛 derivative-only는 사다리꼴 옆면 점진 감소 구간을 다리로
    #   판단해 윗면 좁은 edge band(67px)만 잡는 경우가 있었음. full bbox fallback으로
    #   "너무 좁음"을 후보 단계에서 1차 보정 + 호출자 validate가 "너무 두꺼움" 2차 보정.
    bin_mask   = (mask_np > 127).astype(np.uint8)
    row_widths = bin_mask.sum(axis=1).astype(np.float32)
    if row_widths.max() == 0:
        return None
    valid = np.where(row_widths > 0)[0]
    if len(valid) < 5:
        return None
    full_y_top    = int(valid[0])
    full_y_bottom = int(valid[-1])
    full_h        = full_y_bottom - full_y_top + 1

    section          = row_widths[full_y_top:full_y_bottom + 1]
    derivatives      = np.diff(section)
    biggest_drop_idx = int(np.argmin(derivatives)) if len(derivatives) > 0 else 0
    biggest_drop_val = float(derivatives[biggest_drop_idx]) if len(derivatives) > 0 else 0.0
    avg_width        = float(row_widths.mean())
    min_band_h       = max(30, int(full_h * 0.25))

    band_y_top, band_y_bottom = full_y_top, full_y_top + biggest_drop_idx
    band_source = "derivative"

    deriv_band_h = band_y_bottom - band_y_top + 1
    if -biggest_drop_val < avg_width * 0.2 or deriv_band_h < min_band_h:
        _ot = _otsu_band_fallback(mask_np, full_y_top, full_y_bottom)
        if _ot is not None:
            _oy1, _oy2 = _ot[1], _ot[3]
            if (_oy2 - _oy1 + 1) >= min_band_h:
                band_y_top, band_y_bottom = _oy1, _oy2
                band_source = "otsu_fallback"
        if (band_y_bottom - band_y_top + 1) < min_band_h:
            band_y_top, band_y_bottom = full_y_top, full_y_bottom
            band_source = "full_bbox"

    band       = mask_np[band_y_top:band_y_bottom + 1]
    xs_in_band = np.where((band > 127).any(axis=0))[0]
    if len(xs_in_band) == 0:
        return None
    x_min, x_max = int(xs_in_band.min()), int(xs_in_band.max())

    print(f"  [DeskBand] source={band_source} y_top={band_y_top} y_bottom={band_y_bottom} "
          f"dh={band_y_bottom - band_y_top + 1} full_dh={full_h} "
          f"drop={biggest_drop_val:.0f}px avg_width={avg_width:.0f}")
    return (x_min, band_y_top, x_max, band_y_bottom)


def _otsu_band_fallback(mask_np: np.ndarray, y_start: int, y_end: int) -> tuple | None:
    # derivative 방식 실패 시 fallback — Otsu로 wide row 그룹의 가장 긴 연속 run 추출.
    bin_mask   = (mask_np > 127).astype(np.uint8)
    row_widths = bin_mask.sum(axis=1).astype(np.float32)
    section    = row_widths[y_start:y_end + 1]
    if section.max() == 0:
        return None
    section_norm = (section / section.max() * 255).astype(np.uint8)
    thresh_val, _ = cv2.threshold(
        section_norm.reshape(-1, 1), 0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )
    actual_threshold = (thresh_val / 255.0) * section.max()
    wide_rows = section >= actual_threshold

    runs: list[tuple[int, int]] = []
    start_idx = None
    for i, v in enumerate(wide_rows):
        if v and start_idx is None:
            start_idx = i
        elif (not v) and start_idx is not None:
            runs.append((start_idx, i - 1))
            start_idx = None
    if start_idx is not None:
        runs.append((start_idx, len(wide_rows) - 1))
    if not runs:
        return None
    rel_top, rel_bottom = max(runs, key=lambda r: r[1] - r[0])
    y_top    = y_start + rel_top
    y_bottom = y_start + rel_bottom
    band     = mask_np[y_top:y_bottom + 1]
    xs       = np.where((band > 127).any(axis=0))[0]
    if len(xs) == 0:
        return None
    print(f"  [DeskBand] Otsu fallback: y_top={y_top} y_bottom={y_bottom} dh={y_bottom - y_top}")
    return (int(xs.min()), y_top, int(xs.max()), y_bottom)


def _run_generate(job_id: str, req: GenerateRequest):
    job_store[job_id].status = JobStatus.running
    try:
        from datetime import datetime
        _debug_dir = Path("outputs/debug") / datetime.now().strftime("%Y%m%d_%H%M%S")
        _debug_dir.mkdir(parents=True, exist_ok=True)
        print(f"[Generate] mode={req.mode}")

        image       = b64_to_image(req.image_base64)
        img_w, img_h = image.size
        mode        = req.mode

        # result.jsp가 mode 따라 진행 표시 분기 (빈 책상이면 '제거' 안 보임)
        job_store[job_id].mode = mode.value if hasattr(mode, "value") else str(mode)

        remover  = get_object_removal_processor()
        # LaMa 잔여물 방지 — DINO threshold 낮추고 mask dilation 키워 over-removal 유도.
        # 잘못 제거된 영역은 LaMa가 책상 표면 텍스처로 잘 채움. under-removal보다 안전.
        detection = remover.detect_with_prompt(
            image=image, prompt=_REMOVAL_PROMPT, max_area_ratio=0.40,
            box_threshold=0.20, dilation_size=35,
        )
        front_instances  = detection.get("individual_masks", [])
        front_detections = detection.get("detections", [])
        job_store[job_id].num_removed = detection["num_objects"]

        print(f"[Removal] num_objects={detection.get('num_objects')}")
        for _det in front_detections:
            print(f"[Removal] label={_det.label}, score={_det.score:.3f}, bbox={_det.box_xyxy}")
        print(f"[Removal] front_instances={len(front_instances)}")

        from PIL import ImageDraw as _IDraw
        _ov      = image.copy()
        _ov_draw = _IDraw.Draw(_ov)
        for _det in front_detections:
            _bx1, _by1, _bx2, _by2 = _det.box_xyxy
            _ov_draw.rectangle([_bx1, _by1, _bx2, _by2], outline=(255, 0, 0), width=3)
            _ov_draw.text((_bx1 + 4, _by1 + 4), f"{_det.label}:{_det.score:.2f}", fill=(255, 0, 0))
        _ov.save(_debug_dir / "front_detection_overlay.png")

        _combined_np = np.zeros((img_h, img_w), dtype=np.uint8)
        for _mp in front_instances:
            _arr = np.array(_mp.convert("L"))
            _combined_np = cv2.bitwise_or(_combined_np, (_arr > 127).astype(np.uint8) * 255)
        Image.fromarray(_combined_np).save(_debug_dir / "front_remove_mask.png")

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
            lama    = get_lama_processor()
            current = image
            for _mask_pil in front_instances:
                current = lama.inpaint(image=current, mask=_mask_pil)
            current.save(_debug_dir / "cleaned_front_sequential.png")
            torch.cuda.empty_cache()
        else:
            lama              = get_lama_processor()
            _cleaned_combined = lama.inpaint(image=image, mask=_combined_mask_pil)
            _cleaned_combined.save(_debug_dir / "cleaned_front_combined.png")
            current = _cleaned_combined
            torch.cuda.empty_cache()

        current.save(_debug_dir / "cleaned_front.png")
        job_store[job_id].cleaned_image = image_to_b64(current)

        # pipeline_mode: controlnet (기본, per-product SD generation) |
        #                cv_composite (SD 미사용) | placement_only (배치만) |
        #                harmonize (실험, 폐기 — 환경에 hallucinated 객체 생성 문제)
        _pipeline_mode = getattr(req, "generation_mode", "controlnet")
        # 'harmonize' alias는 사용자가 명시 요청 시에만, 기본은 controlnet으로
        print(f"[Generate] pipeline_mode={_pipeline_mode}")

        cn_proc = get_controlnet_inpaint_processor()

        import json as _jmeta
        _lora_ext_dir = Path(__file__).parent.parent / "outputs" / "models" / "lora_external"
        (_debug_dir / "generation_meta.json").write_text(
            _jmeta.dumps({
                "backbone":         "sd1.5-inpaint+depth-controlnet+ip-adapter-plus+lora",
                "base_model":       "runwayml/stable-diffusion-inpainting",
                "controlnets":      ["sd-controlnet-depth"],
                "ip_adapter":       "ip-adapter-plus_sd15",
                "image_encoder":    "default (CLIP-ViT-L from h94/IP-Adapter models)",
                "lora_adapter":     "outputs/models/lora_external (PEFT)",
                "pipeline_mode":    _pipeline_mode,
                "has_lora":         cn_proc._has_lora,
                "lora_path_exists": _lora_ext_dir.exists(),
                "lora_reason":      None if cn_proc._has_lora else "LoRA load failed (check path/format)",
                "lora_status":      "active" if cn_proc._has_lora else "skipped",
                "top_view_source":  getattr(req, "top_view_source", None) or (
                    "user_provided" if req.top_view_image_base64 else "none"
                ),
            }, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        products = enrich_products_from_db(req.products)
        print(f"[Generate] products={len(products)}")
        for _p in products:
            print(f"  product: category={_p.category}, image_id={_p.image_id}, "
                  f"size={_p.width_mm}x{_p.depth_mm}")

        num_placed    = 0
        run_errors:   list[str]   = []
        gen_mode      = _pipeline_mode
        _gen_results: dict[str, dict] = {}

        def _run_cn(p, x1, y1, x2, y2):
            # per-product SD 생성 (IP-Adapter + ControlNet + LoRA). 직전 안정 버전 복원.
            nonlocal current, num_placed, run_errors
            cat       = normalize_category(p.category)
            _ar_range = _CAT_ASPECT_VALID.get(cat)

            def _record(route, status, error=None, ar=None, ar_valid=None, debug_json=False):
                _gen_results[cat] = {
                    "generation_route":   route,
                    "generation_status":  status,
                    "generation_error":   error,
                    "aspect_ratio":       round(ar, 3) if ar is not None else None,
                    "aspect_ratio_valid": ar_valid,
                    "aspect_ratio_range": list(_ar_range) if _ar_range else None,
                    "debug_json_created": debug_json,
                }

            if p.image_id is None:
                msg = f"{cat}: image_id=None"
                run_errors.append(msg); print(f"  [_run_cn SKIP] {msg}")
                _record("skipped_no_image", "skipped", error=msg); return
            prod_path = find_product_image(p.image_id)
            if prod_path is None:
                msg = f"{cat}: product image not found for image_id={p.image_id}"
                run_errors.append(msg); print(f"  [_run_cn SKIP] {msg}")
                _record("skipped_no_file", "skipped", error=msg); return

            try:
                prod_img        = Image.open(prod_path)
                _raw_w, _raw_h  = prod_img.size
                _prod_debug_dir = _debug_dir / "products"
                _prod_debug_dir.mkdir(exist_ok=True)
                prod_img.save(_prod_debug_dir / f"{cat}_{p.image_id}_raw.png")
                prod_alpha = prepare_product_image_for_composite(prod_img)
                prod_alpha.save(_prod_debug_dir / f"{cat}_{p.image_id}_alpha.png")

                _ar         = prod_alpha.width / max(prod_alpha.height, 1)
                _ar_valid   = True
                _ar_invalid = False
                if _ar_range is not None:
                    _ar_min, _ar_max = _ar_range
                    if not (_ar_min <= _ar <= _ar_max):
                        _ar_invalid = True; _ar_valid = False
                        msg = (f"{cat} id={p.image_id}: aspect_ratio={_ar:.2f} "
                               f"out of [{_ar_min}, {_ar_max}]")
                        run_errors.append(msg)
                        print(f"  [WARNING] {msg} — CV fallback")

                if cat in _CV_ONLY_CATS or _ar_invalid:
                    route = "cv_composite" if cat in _CV_ONLY_CATS else "cv_fallback_aspect_invalid"
                    _record(route, "done", ar=_ar, ar_valid=_ar_valid)
                    current = composite_product_simple(current, prod_alpha, (x1, y1, x2, y2), category=cat)
                    current = _add_shadows(current, (x1, y1, x2, y2), cat,
                                           prod_alpha=prod_alpha, debug_dir=_prod_debug_dir)
                    num_placed += 1
                    print(f"  [_run_cn CV] {cat} 완료 (num_placed={num_placed})")
                    return

                # === IP-Adapter scale 정책 ===
                # 제품 reference 이미지의 영향 강도. 너무 약하면 hallucination(QR/JG79 등),
                # 너무 강하면 SD가 원본 이미지를 거의 그대로 복사하여 어색.
                #
                # 카테고리 | scale | 근거
                # --------|-------|-----
                # MONITOR  | 0.60  | 화면 hallucination 방지 위해 강하게 (2026-05-24 QR 사태 후 0.35→0.60)
                # MOUSE    | 0.60  | 작은 디테일 보존 필요 (버튼/스크롤휠)
                # MOUSEPAD | 0.60  | 표면 패턴/로고 보존
                # SPEAKER  | 0.55  | 형태 다양성 큼, 중간 강도
                # DEFAULT  | 0.55  | 일반 카테고리 안전 기본값
                # DESK_LAMP| 0.40  | 어두운 부분 많아 너무 강하면 음영 어색
                # DESK_SHELF | 0.35 | 책상 위에 얹는 단순 구조, 강한 참조 불필요
                # brightness<40 | 0.0 | 너무 어두운 이미지(lifestyle 컷)는 IP-Adapter가 망침
                _prod_rgb  = prod_img.convert("RGB")
                brightness = float(np.array(_prod_rgb).mean())
                # upright 제품(MONITOR/SPEAKER)은 IP-Adapter scale 낮춤 — 원본 시점 강제 복사 방지.
                # CV composite로 제품 강하게 보존 + SDXL은 경계/그림자만 자연화하는 전략.
                if brightness < 40:
                    ip_scale = 0.0
                elif cat == "DESK_SHELF":
                    ip_scale = 0.30
                elif cat == "MONITOR":
                    ip_scale = 0.40
                elif cat == "SPEAKER":
                    ip_scale = 0.40
                elif cat == "DESK_LAMP":
                    ip_scale = 0.40
                elif cat in ("MOUSE", "MOUSEPAD"):
                    ip_scale = 0.55
                else:
                    ip_scale = 0.50

                # 이전: view_type=='top_view'면 ip_scale 0.5배. 제거됨.
                # 이유: controlnet_inpaint_processor.analyze_product_image_risk()가
                #       runtime에서 입력 이미지를 분석해 ip_scale을 adaptive로 결정.
                #       view_type 메타데이터에 의존하지 않고 형태(aspect ratio, contact base)
                #       기준으로 위험도 판단 → 더 견고함.
                # 여기서 전달하는 ip_scale은 main.py의 카테고리별 default일 뿐,
                # generate_product 내부에서 risk analysis에 의해 override됨.

                mask           = _make_rect_mask(img_w, img_h, x1, y1, x2, y2)
                context_region = (0, img_h // 2, img_w, img_h) if cat in _FRONT_CATS else None
                _debug_meta = {
                    "image_id":           p.image_id,
                    "width_mm":           getattr(p, "width_mm", None),
                    "depth_mm":           getattr(p, "depth_mm", None),
                    "product_raw_w":      _raw_w, "product_raw_h": _raw_h,
                    "aspect_ratio":       round(_ar, 3),
                    "aspect_ratio_valid": _ar_valid,
                }
                _record("controlnet", "done", ar=_ar, ar_valid=_ar_valid)
                current = cn_proc.generate_product(
                    image=current, mask=mask, product_image=prod_alpha,
                    category=p.category, style=req.style.value,
                    context_region=context_region,
                    ip_adapter_scale=ip_scale,
                    debug_dir=_prod_debug_dir,
                    debug_meta=_debug_meta,
                )
                # contact shadow는 generate_product 내부에서 composite_sd에 미리 그려넣음.
                # upright(MONITOR/SPEAKER 등)는 후처리 _add_shadows 비활성 — 그림자 중복 방지.
                # KEYBOARD/MOUSE 등은 _add_shadows에 cast shadow 의존성 있어 유지.
                from .config import _PRODUCT_FORM_TIER as _PFT
                if _PFT.get(cat) != "upright":
                    current = _add_shadows(current, (x1, y1, x2, y2), cat,
                                           prod_alpha=prod_alpha, debug_dir=_prod_debug_dir)
                else:
                    print(f"  [_add_shadows] {cat} upright — skip (contact shadow는 generate_product 내부 처리)")
                num_placed += 1
                print(f"  [_run_cn] {cat} 완료 (num_placed={num_placed})")
            except Exception as _e:
                msg = f"{p.category}: ControlNet generation failed: {_e}"
                run_errors.append(msg)
                print(f"  [_run_cn ERROR] {msg}")
                log.error("_run_cn ERROR %s\n%s", msg, traceback.format_exc())
                if cat not in _gen_results:
                    _record("skipped_error", "failed", error=str(_e))
                else:
                    _gen_results[cat]["generation_status"] = "failed"
                    _gen_results[cat]["generation_error"]  = str(_e)

        # ── 배치 위치 결정: top-view available space 우선, 없으면 _calc_regions ──
        placement_items:    list[dict] = []
        _unplaced_for_json: list[dict] = []

        # === SAM2 책상 윗면 mask 추출 (mode=add + 클릭 좌표 있을 때) ===
        # top-view 유무와 무관하게 항상 시도. 결과는 두 placement 경로에서 공통 사용.
        _front_desk_bbox_override = None
        _click_x = getattr(req, "desk_click_x", None)
        _click_y = getattr(req, "desk_click_y", None)
        if mode == RemoveMode.add and _click_x is not None and _click_y is not None:
            try:
                from .sam2_processor import get_sam2_processor, b64_to_image as _sam_b2i
                _sam = get_sam2_processor()
                _px = int(_click_x * current.size[0])
                _py = int(_click_y * current.size[1])
                print(f"[DeskClick] add mode + click=({_click_x:.3f},{_click_y:.3f}) "
                      f"→ pixel=({_px},{_py}) → SAM2 segment")
                _sam_result = _sam.segment(
                    image=current, points=[[_px, _py]], point_labels=[1],
                )
                _desk_mask_pil = _sam_b2i(_sam_result["mask"]).convert("L")
                _desk_mask_np  = np.array(_desk_mask_pil)
                _desk_mask_pil.save(_debug_dir / "front_desk_click_mask.png")

                _ys, _xs = np.where(_desk_mask_np > 127)
                if len(_xs) == 0:
                    print("[DeskClick] SAM2 mask empty — fallback to default desk detection")
                else:
                    _full_bbox = (int(_xs.min()), int(_ys.min()),
                                  int(_xs.max()), int(_ys.max()))
                    # SAM이 책상 다리까지 한 mask로 잡는 케이스 대응 — Otsu로 책상 윗면 띠만 추출
                    _band = extract_desk_top_band(_desk_mask_np)
                    if _band is not None:
                        _front_desk_bbox_override = _band
                        _band_img = np.zeros_like(_desk_mask_np)
                        _band_img[_band[1]:_band[3] + 1, _band[0]:_band[2] + 1] = 255
                        Image.fromarray(_band_img).save(
                            _debug_dir / "front_desk_top_band.png"
                        )
                        print(f"[DeskClick] SAM2 full bbox={_full_bbox} → "
                              f"Otsu band={_band} "
                              f"(dh: {_full_bbox[3]-_full_bbox[1]} → {_band[3]-_band[1]})")
                    else:
                        _front_desk_bbox_override = _full_bbox
                        print(f"[DeskClick] Otsu band extraction failed → "
                              f"fallback to SAM full bbox = {_full_bbox}")
            except Exception as _e:
                print(f"[DeskClick] SAM2 호출 실패: {_e} — fallback to default")

        # === Placement: top-view 항상 받으므로 단일 경로 ===
        # top-view는 main.py 진입 단계에서 필수 검증됨 → 여기 도달 시 항상 존재.
        # SAM2 클릭 mask가 있으면 front_desk_bbox_override로 활용 (검은 책상 등 DINO 약한 케이스 보정).
        print(f"[Generate] desk_width_mm={req.desk_width_mm} desk_depth_mm={req.desk_depth_mm}")
        top_image_for_place = b64_to_image(req.top_view_image_base64)
        _all_space_results, _tabletop_meta = calc_placements_from_available_space(
            front_image=current,
            top_view_image=top_image_for_place,
            products=products,
            desk_width_mm=req.desk_width_mm,
            desk_depth_mm=req.desk_depth_mm,
            mode=mode,
            remover=remover,
            debug_dir=_debug_dir,
            front_desk_bbox_override=_front_desk_bbox_override,
        )

        # tabletop bbox invalid 처리는 빈 책상 모드(RemoveMode.add) 한정.
        # 일반 모드(remove/preserve)는 옛 fallback 경로(_calc_regions)로 진행 — 1차 fix는
        # 빈 책상 모드에서 SAM2/DINO가 책상 상판을 잘못 잡는 케이스만 다룬 것이라,
        # 일반 모드에 strict invalid 적용 시 정상 사진도 fail되는 회귀 발생.
        if not _tabletop_meta.get("tabletop_valid", True):
            _reason   = _tabletop_meta.get("tabletop_invalid_reason", "unknown")
            _attempts = _tabletop_meta.get("tabletop_candidates", [])
            _summary  = " | ".join(f"{a['source']}:{a['reason']}" for a in _attempts)
            if mode == RemoveMode.add:
                job_store[job_id].status = JobStatus.failed
                job_store[job_id].error  = (
                    f"tabletop bbox invalid: {_reason}. "
                    f"책상 상판 영역을 인식하지 못했습니다. 클릭 위치를 책상 상판 중앙으로 다시 시도해주세요. "
                    f"후보 시도: {_summary}"
                )
                print(f"[Generate] tabletop invalid (add mode) → job failed. {_summary}")
                return
            else:
                print(f"[Generate] tabletop invalid (mode={mode.value}) — 일반 모드는 fallback 진행. {_summary}")

        # === dedup + space-fail fallback (분기 공통) ===
        _scored_items = [i for i in _all_space_results if i.get("region") is not None]
        _failed_items = [i for i in _all_space_results if i.get("region") is None]
        _failed_meta  = {id(i["product"]): i for i in _failed_items}

        _deduped: list[dict] = []
        for _item in _scored_items:
            _cat = normalize_category(_item["product"].category)
            # dedup도 _OVERLAP_TOLERANCE 따름 (MOUSE↔MOUSEPAD 같은 의도된 겹침 허용)
            if all(
                bbox_iou(_item["region"], _prev["region"])
                < _overlap_threshold(_cat, normalize_category(_prev["product"].category))
                for _prev in _deduped
            ):
                _deduped.append(_item)
            else:
                print(f"[Placement SKIP] internal overlap: {_cat} {_item['region']}")
        placement_items = _deduped

        _failed_products = [i["product"] for i in _failed_items]
        if _failed_products:
            _desk_bbox_fb  = _detect_desk_bbox(current)
            _raw_fallback  = _calc_regions(img_w, img_h, _failed_products, _desk_bbox_fb, req.desk_width_mm)
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
                if all(
                    bbox_iou(_fi["region"], _prev["region"])
                    < _overlap_threshold(_fi_cat, normalize_category(_prev["product"].category))
                    for _prev in placement_items
                ):
                    placement_items.append(_fi)
                    _added += 1
                else:
                    print(f"[Placement SKIP] overlap: {_fi_cat} {_fi['region']}")
            print(f"[Generate] space-fail {len(_failed_products)}개 → fallback {_added}개 추가")

        _placed_ids        = {id(i["product"]) for i in placement_items}
        _unplaced_for_json = [i for i in _failed_items if id(i["product"]) not in _placed_ids]
        print(f"[Generate] available_space 배치: {len(placement_items)}개")

        if not placement_items:
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

        print(f"[Generate] placement_items={len(placement_items)}")

        if not placement_items:
            job_store[job_id].status = JobStatus.failed
            job_store[job_id].error  = (
                "배치 가능한 제품 영역이 없습니다. "
                "products.category, image_id, top_view_image_base64, "
                "desk_mask/available_space 분석 결과를 확인하세요."
                + ((" | " + " | ".join(run_errors[:5])) if run_errors else "")
            )
            return

        placement_items.sort(
            key=lambda item: _PLACEMENT_ORDER.get(normalize_category(item["product"].category), 999)
        )

        from PIL import ImageDraw as _ID
        _dbg  = current.copy()
        _draw = _ID.Draw(_dbg)
        for _item in placement_items:
            _x1, _y1, _x2, _y2 = _item["region"]
            _cat   = normalize_category(_item["product"].category)
            _score = _item.get("score")
            _rid   = _item.get("available_region_id")
            _src   = "av" if _item.get("placement_source") == "available_space_scoring" else "fb"
            _label = _cat
            if _score is not None:
                _label += f" s={_score:.2f}"
            if _rid is not None:
                _label += f" r{_rid}"
            _label += f" [{_src}]"
            _draw.rectangle([_x1, _y1, _x2, _y2], outline=(255, 0, 0), width=4)
            _draw.text((_x1 + 4, _y1 + 4), _label, fill=(255, 0, 0))
        _dbg.save(_debug_dir / "placement_debug.png")
        print(f"[Generate] 배치 시각화 저장: {_debug_dir}/placement_debug.png")

        _dbg.save(_debug_dir / "placement_only_result.png")

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
                "rule_score":             (_it.get("score_meta") or {}).get("rule_score"),
                "learned_layout_score":   (_it.get("score_meta") or {}).get("learned_layout_score"),
                "final_score":            (_it.get("score_meta") or {}).get("final_score"),
                "ranker_used":            (_it.get("score_meta") or {}).get("ranker_used"),
                "ranker_skipped_reason":  (_it.get("score_meta") or {}).get("ranker_skipped_reason"),
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
                "anchor_ry":             None,
            }
            for _it in _unplaced_for_json
        ]
        _products_list_meta = {
            "generation_mode": req.generation_mode,
            "products":        _products_info,
        }
        (_debug_dir / "products_list.json").write_text(
            _json.dumps(_products_list_meta, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        if gen_mode == "placement_only":
            job_store[job_id].num_placed   = 0
            job_store[job_id].result_image = image_to_b64(_dbg)
            job_store[job_id].status       = JobStatus.done
            print("[Generate] placement_only 완료")
            return

        if gen_mode == "cv_composite":
            # SD 미사용 모드 — 그림자도 SD가 안 만들어주므로 _add_shadows 직접 호출
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
                prod_img   = Image.open(prod_path)
                prod_alpha = prepare_product_image_for_composite(prod_img)
                current    = composite_product_simple(current, prod_img, item["region"], category=cat)
                current    = _add_shadows(current, item["region"], cat,
                                          prod_alpha=prod_alpha, debug_dir=_debug_dir / "products")
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

        # ── SPEAKER 전처리: 스테레오=동일 이미지 양쪽, 바형=1개만 ────
        _sp_items = [(i, it) for i, it in enumerate(placement_items)
                     if normalize_category(it["product"].category) == "SPEAKER"]
        if len(_sp_items) >= 2:
            _sp0_prod = _sp_items[0][1]["product"]
            _sp0_path = find_product_image(_sp0_prod.image_id) if _sp0_prod.image_id else None
            if _sp0_path:
                _sp0_alpha = prepare_product_image_for_composite(Image.open(_sp0_path))
                _sp0_ar    = _sp0_alpha.width / max(_sp0_alpha.height, 1)
                if _sp0_ar > 2.5:
                    _keep_idxs = {_sp_items[0][0]}
                    placement_items = [it for i, it in enumerate(placement_items)
                                       if normalize_category(it["product"].category) != "SPEAKER"
                                       or i in _keep_idxs]
                    print(f"[Speaker] 바형(AR={_sp0_ar:.2f}) → 1개 배치")
                else:
                    for _si, _ in _sp_items[1:]:
                        placement_items[_si]["product"] = _sp0_prod
                    print(f"[Speaker] 스테레오형(AR={_sp0_ar:.2f}) → 동일 이미지 {len(_sp_items)}개 배치")

        # ── per-product SD generation 루프 (controlnet 모드) ──
        # IP-Adapter + ControlNet(depth+canny) + LoRA로 제품마다 SD 호출.
        # harmonize 방식(2026-05-23 실험)은 환경에 hallucinated 객체 추가 문제로 폐기.
        for item in placement_items:
            p   = item["product"]
            cat = normalize_category(p.category)
            if cat == "DECO" and (item.get("score") is None or item.get("score", 0) <= 0):
                print(f"  [Generate SKIP] DECO score={item.get('score')} ≤ 0")
                continue
            print(f"  [Generate] 처리: {p.category} image_id={p.image_id} region={item['region']}")
            _run_cn(p, *item["region"])

        # SD pipeline은 GPU에 상주시킴 (init 주석 참조).
        # 이전 코드: cn_proc.pipe.to("cpu") + empty_cache → 2번째 호출 시 pipe가 CPU에
        # 남아 SD inference가 CPU에서 hang. 메모리 절약 의도였으나 12GB에 여유 있음.
        torch.cuda.empty_cache()

        print(f"[Generate] num_placed={num_placed}")

        for _entry in _products_info:
            _gr = _gen_results.get(_entry.get("category"), {})
            _entry["generation_route"]   = _gr.get("generation_route")
            _entry["generation_status"]  = _gr.get("generation_status")
            _entry["generation_error"]   = _gr.get("generation_error")
            _entry["aspect_ratio"]       = _gr.get("aspect_ratio")
            _entry["aspect_ratio_valid"] = _gr.get("aspect_ratio_valid")
            _entry["aspect_ratio_range"] = _gr.get("aspect_ratio_range")
        (_debug_dir / "products_list.json").write_text(
            _json.dumps(_products_list_meta, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        if num_placed == 0:
            job_store[job_id].status = JobStatus.failed
            job_store[job_id].error  = (
                "제품 생성이 0개 수행되었습니다: "
                + (" | ".join(run_errors[:5]) if run_errors else "알 수 없는 오류")
            )
            return

        job_store[job_id].num_placed   = num_placed
        job_store[job_id].result_image = image_to_b64(current)
        job_store[job_id].status       = JobStatus.done
        print(f"[Generate] 완료. num_placed={num_placed}")

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
        image     = b64_to_image(req.image_base64)
        w, h      = image.size
        px = int(req.point_x * w)
        py = int(req.point_y * h)
        result = processor.segment(image, points=[[px, py]], point_labels=[req.label])
        job_store[job_id].mask_base64    = result["mask"]
        job_store[job_id].overlay_base64 = result["overlay"]
        job_store[job_id].status         = JobStatus.done
    except Exception as e:
        job_store[job_id].status = JobStatus.failed
        job_store[job_id].error  = str(e)
        log.error("job_id=%s\n%s", job_id, traceback.format_exc())
