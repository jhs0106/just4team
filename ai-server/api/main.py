import uuid
import asyncio
import torch
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .models import (
    StyleName, JobStatus,
    ObjectRemovalRequest, ObjectRemovalResult,
    Img2ImgRequest, Img2ImgResult,
    InpaintRequest, InpaintResult,
    PlaceRequest, PlaceResult,
    AnalyzeRequest, AnalyzeResult,
)
from .img2img_processor import get_img2img_processor, b64_to_image, image_to_b64
from .inpaint_processor import get_inpaint_processor
from .place_processor import get_place_processor
from .object_removal_processor import get_object_removal_processor
from .lama_processor import get_lama_processor


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
    version="4.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


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


# ── Step 2: Style Transfer ──────────────────────────────
@app.post("/img2img", response_model=Img2ImgResult)
async def run_img2img(req: Img2ImgRequest):
    job_id = str(uuid.uuid4())
    job_store[job_id] = Img2ImgResult(job_id=job_id, status=JobStatus.pending)
    asyncio.get_event_loop().run_in_executor(executor, _run_img2img, job_id, req)
    return job_store[job_id]


def _run_img2img(job_id: str, req: Img2ImgRequest):
    job_store[job_id].status = JobStatus.running
    try:
        processor = get_img2img_processor()
        processor.pipe.to("cuda")

        image = b64_to_image(req.image_base64)
        result = processor.process(
            image=image,
            prompt=req.prompt,
            style=req.style.value if req.style else None,
            strength=req.strength,
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


# ── Step 3: Product Placement (임시 — 향후 /place로 교체) ──
@app.post("/inpaint", response_model=InpaintResult)
async def run_inpaint(req: InpaintRequest):
    job_id = str(uuid.uuid4())
    job_store[job_id] = InpaintResult(job_id=job_id, status=JobStatus.pending)
    asyncio.get_event_loop().run_in_executor(executor, _run_inpaint, job_id, req)
    return job_store[job_id]


def _run_inpaint(job_id: str, req: InpaintRequest):
    job_store[job_id].status = JobStatus.running
    try:
        processor = get_inpaint_processor()
        processor.pipe.to("cuda")

        image = b64_to_image(req.image_base64)
        mask = b64_to_image(req.mask_base64)
        product = b64_to_image(req.product_image_base64)
        result = processor.inpaint(
            image=image,
            mask=mask,
            product_image=product,
            prompt=req.prompt,
            style=req.style.value if req.style else None,
            num_inference_steps=req.num_inference_steps,
            guidance_scale=req.guidance_scale,
            ip_adapter_scale=req.ip_adapter_scale,
        )

        processor.pipe.to("cpu")
        torch.cuda.empty_cache()

        job_store[job_id].status = JobStatus.done
        job_store[job_id].result_image = image_to_b64(result)
    except Exception as e:
        job_store[job_id].status = JobStatus.failed
        job_store[job_id].error = str(e)


# ── Step 3 (확정): Product Placement ───────────────────────
@app.post("/place", response_model=PlaceResult)
async def run_place(req: PlaceRequest):
    job_id = str(uuid.uuid4())
    job_store[job_id] = PlaceResult(job_id=job_id, status=JobStatus.pending)
    asyncio.get_event_loop().run_in_executor(executor, _run_place, job_id, req)
    return job_store[job_id]


def _run_place(job_id: str, req: PlaceRequest):
    job_store[job_id].status = JobStatus.running
    try:
        processor = get_place_processor()

        background = b64_to_image(req.image_base64)
        mask = b64_to_image(req.mask_base64)
        product = b64_to_image(req.product_image_base64)
        result = processor.place(background=background, mask=mask, product=product)

        job_store[job_id].status = JobStatus.done
        job_store[job_id].result_image = image_to_b64(result)
    except Exception as e:
        job_store[job_id].status = JobStatus.failed
        job_store[job_id].error = str(e)


# ── Space Analysis ──────────────────────────────────────────
@app.post("/analyze", response_model=AnalyzeResult)
async def run_analyze(req: AnalyzeRequest):
    job_id = str(uuid.uuid4())
    job_store[job_id] = AnalyzeResult(job_id=job_id, status=JobStatus.pending)
    asyncio.get_event_loop().run_in_executor(executor, _run_analyze, job_id, req)
    return job_store[job_id]


def _run_analyze(job_id: str, req: AnalyzeRequest):
    job_store[job_id].status = JobStatus.running
    try:
        from .space_processor import get_space_processor
        processor = get_space_processor()
        image = b64_to_image(req.image_base64)
        result = processor.analyze(image, req.desk_width_cm, req.desk_depth_cm)

        job_store[job_id].status = JobStatus.done
        job_store[job_id].metrics = result["metrics"]
        job_store[job_id].available_mask = result["available_mask"]
        job_store[job_id].occupied_mask = result["occupied_mask"]
        job_store[job_id].placement_center = result["placement_center"]
        job_store[job_id].num_objects = result["num_objects"]
    except Exception as e:
        job_store[job_id].status = JobStatus.failed
        job_store[job_id].error = str(e)
