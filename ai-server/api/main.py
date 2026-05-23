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
    GenerateRequest, GenerateResult,
)
from .object_removal_processor import get_object_removal_processor
from .lama_processor import get_lama_processor
from .product_inpaint_processor import get_product_inpaint_processor
from .harmonization_processor import get_harmonization_processor
from .sam2_processor import get_sam2_processor
from .config import (
    _PLACEMENT_ORDER, _CONTACT_Y_OFFSET,
    _FRONT_CATS, _BACK_CATS, _REMOVAL_PROMPT,
)
from .utils import (
    b64_to_image, image_to_b64,
    normalize_category, find_product_image, enrich_products_from_csv,
)
from .composite import (
    prepare_product_image_for_composite, composite_product_simple,
    composite_one_with_silhouette,
    _add_shadows, _detect_desk_bbox, _calc_regions,
)
from .placement import (
    calc_placements_from_available_space, bbox_iou,
    _match_products_to_detections, _region_from_detection_center,
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

        remover  = get_object_removal_processor()
        detection = remover.detect_with_prompt(
            image=image, prompt=_REMOVAL_PROMPT, max_area_ratio=0.40,
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

        # pipeline_mode: harmonize (기본, 새 4-stage) | cv_composite (SD 미사용) | placement_only (배치만)
        # 'controlnet'은 backward-compat: harmonize로 alias
        _pipeline_mode = getattr(req, "generation_mode", "harmonize")
        if _pipeline_mode == "controlnet":
            _pipeline_mode = "harmonize"
        print(f"[Generate] pipeline_mode={_pipeline_mode}")

        import json as _jmeta
        _lora_ext_dir = Path(__file__).parent.parent / "outputs" / "models" / "lora_external"
        (_debug_dir / "generation_meta.json").write_text(
            _jmeta.dumps({
                "backbone":         "sd1.5-controlnet (harmonization)",
                "pipeline_mode":    _pipeline_mode,
                "lora_path_exists": _lora_ext_dir.exists(),
            }, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        products = enrich_products_from_csv(req.products)
        print(f"[Generate] products={len(products)}")
        for _p in products:
            print(f"  product: category={_p.category}, image_id={_p.image_id}, "
                  f"size={_p.width_mm}x{_p.depth_mm}")

        num_placed   = 0
        run_errors:  list[str]  = []
        gen_mode     = _pipeline_mode

        # ── 배치 위치 결정: top-view available space 우선, 없으면 _calc_regions ──
        placement_items:    list[dict] = []
        _unplaced_for_json: list[dict] = []

        if req.top_view_image_base64:
            print(f"[Generate] desk_width_mm={req.desk_width_mm} desk_depth_mm={req.desk_depth_mm}")
            top_image_for_place = b64_to_image(req.top_view_image_base64)
            _all_space_results  = calc_placements_from_available_space(
                front_image=current,
                top_view_image=top_image_for_place,
                products=products,
                desk_width_mm=req.desk_width_mm,
                desk_depth_mm=req.desk_depth_mm,
                mode=mode,
                remover=remover,
                debug_dir=_debug_dir,
            )

            _scored_items = [i for i in _all_space_results if i.get("region") is not None]
            _failed_items = [i for i in _all_space_results if i.get("region") is None]
            _failed_meta  = {id(i["product"]): i for i in _failed_items}

            _deduped: list[dict] = []
            for _item in _scored_items:
                if all(bbox_iou(_item["region"], _prev["region"]) < 0.25 for _prev in _deduped):
                    _deduped.append(_item)
                else:
                    print(f"[Placement SKIP] internal overlap: {normalize_category(_item['product'].category)} {_item['region']}")
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
                    if all(bbox_iou(_fi["region"], _prev["region"]) < 0.25 for _prev in placement_items):
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

        # ── Stage 2 [A]: Retrieval Injection (Augmentation) ──
        # Spring Boot에서 retrieve된 제품 PNG를 cleaned_desk에 한 번에 CV composite으로 inject.
        # 각 제품 silhouette → Stage 3의 faithfulness mask (strength=0 영역) 계산 입력.
        cv_base = current.copy()
        silhouettes:    list[Image.Image] = []
        cats_for_prompt: list[str]        = []
        placed_meta:    list[dict]        = []

        for item in placement_items:
            p   = item["product"]
            cat = normalize_category(p.category)
            x1, y1, x2, y2 = item["region"]

            if cat == "DECO" and (item.get("score") is None or item.get("score", 0) <= 0):
                print(f"  [Composite SKIP] DECO score={item.get('score')} ≤ 0")
                continue
            if p.image_id is None:
                msg = f"{cat}: image_id=None"
                run_errors.append(msg)
                print(f"  [Composite SKIP] {msg}")
                continue
            prod_path = find_product_image(p.image_id)
            if prod_path is None:
                msg = f"{cat}: image not found id={p.image_id}"
                run_errors.append(msg)
                print(f"  [Composite SKIP] {msg}")
                continue

            try:
                prod_img        = Image.open(prod_path)
                _prod_debug_dir = _debug_dir / "products"
                _prod_debug_dir.mkdir(exist_ok=True)
                prod_img.save(_prod_debug_dir / f"{cat}_{p.image_id}_raw.png")

                cv_base, sil = composite_one_with_silhouette(
                    cv_base, prod_img, (x1, y1, x2, y2), category=cat,
                )
                # B: SD 보조 그림자 — CV로 contact/cast shadow 미리 깔기.
                # Stage 3 SD는 그림자 위에서 color/lighting harmonization만 담당
                prod_alpha = prepare_product_image_for_composite(prod_img)
                cv_base    = _add_shadows(cv_base, (x1, y1, x2, y2), cat,
                                          prod_alpha=prod_alpha, debug_dir=_prod_debug_dir)
                silhouettes.append(sil)
                cats_for_prompt.append(cat)
                placed_meta.append({"category": cat, "image_id": p.image_id, "region": [x1, y1, x2, y2]})
                num_placed += 1
                print(f"  [Composite+Shadow] {cat} 완료 (n={num_placed})")
            except Exception as _e:
                msg = f"{cat}: composite failed: {_e}"
                run_errors.append(msg)
                print(f"  [Composite ERROR] {msg}")
                log.error("Composite ERROR %s\n%s", msg, traceback.format_exc())

        cv_base.save(_debug_dir / "composite_full.png")
        print(f"[Generate] Stage 2 완료: {num_placed}개 제품 합성")

        if num_placed == 0:
            job_store[job_id].status = JobStatus.failed
            job_store[job_id].error  = (
                "composite 0개: "
                + (" | ".join(run_errors[:5]) if run_errors else "알 수 없는 오류")
            )
            return

        # ── Stage 3 [G]: Conditional Generation (단일 SD 호출) ──
        # Augmented context(cv_base) 위에서 retrieved fact 주변(seam/그림자/조명)만 SD가 생성.
        # 제품 픽셀은 differential blend로 100% 보존 (Visual-RAG faithfulness guarantee).
        print(f"[Generate] Stage 3 시작: conditional generation (Visual-RAG G)")
        proc = get_harmonization_processor()
        try:
            result = proc.harmonize(
                composite_full=cv_base,
                silhouettes=silhouettes,
                categories=cats_for_prompt,
                style=req.style.value,
                debug_dir=_debug_dir,
            )
            print("[Generate] Stage 3 완료")
        except Exception as _e:
            msg = f"harmonization failed: {_e}"
            run_errors.append(msg)
            print(f"  [Harmonize ERROR] {msg}")
            log.error("Harmonize ERROR %s\n%s", msg, traceback.format_exc())
            # SD 실패시 CV composite 결과만으로 fallback
            result = cv_base
            print("[Generate] fallback: CV composite 결과 그대로 사용")

        for _entry in _products_info:
            _cat_entry = _entry.get("category")
            _entry["generation_route"]  = "harmonize" if any(m["category"] == _cat_entry for m in placed_meta) else "skipped"
            _entry["generation_status"] = "done"      if any(m["category"] == _cat_entry for m in placed_meta) else "skipped"
        (_debug_dir / "products_list.json").write_text(
            _json.dumps(_products_list_meta, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        job_store[job_id].num_placed   = num_placed
        job_store[job_id].result_image = image_to_b64(result)
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
