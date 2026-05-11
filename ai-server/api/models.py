from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum


class StyleName(str, Enum):
    white = "white"
    black = "black"
    modern = "modern"
    gaming = "gaming"
    cozy = "cozy"
    nordic = "nordic"
    retro = "retro"
    industrial = "industrial"
    general = "general"


class JobStatus(str, Enum):
    pending = "pending"
    running = "running"
    done = "done"
    failed = "failed"


# ── Step 1: Object Removal ──────────────────────────────
class ObjectRemovalRequest(BaseModel):
    image_base64: str = Field(..., description="원본 정면 이미지 (base64)")
    prompt: Optional[str] = Field(
        None, description="제거할 물체 텍스트 (예: 'keyboard. mouse. headset.'). 제공 시 Grounding DINO + SAM-2 사용"
    )
    top_view_image_base64: Optional[str] = Field(
        None, description="책상 탑뷰 이미지 (base64) — 있으면 탑뷰 기반 물체 감지 사용"
    )
    max_area_ratio: float = Field(
        0.20, ge=0.05, le=0.60,
        description="이 비율 초과 크기의 마스크는 책상/배경으로 간주하고 제외 (기본 20%)",
    )


class ObjectRemovalResult(BaseModel):
    job_id: str
    status: JobStatus
    cleaned_image: Optional[str] = None
    mask_image: Optional[str] = None
    detection_overlay: Optional[str] = None
    num_objects: int = 0
    error: Optional[str] = None


# ── Step 2: Style Transfer ──────────────────────────────
class Img2ImgRequest(BaseModel):
    image_base64: str = Field(..., description="입력 이미지 (base64) — 빈 책상 권장")
    prompt: str = Field(..., description="스타일 프롬프트")
    style: Optional[StyleName] = None
    strength: float = Field(0.7, ge=0.0, le=1.0, description="0=원본 유지, 1=완전 재생성. 0.6~0.8 권장")
    num_inference_steps: int = Field(30, ge=10, le=100)
    guidance_scale: float = Field(7.5, ge=1.0, le=20.0)


class Img2ImgResult(BaseModel):
    job_id: str
    status: JobStatus
    result_image: Optional[str] = None
    error: Optional[str] = None


# ── Step 3: Product Placement (임시 — 향후 /place로 교체) ──
class InpaintRequest(BaseModel):
    image_base64: str = Field(..., description="원본 이미지 (base64)")
    mask_base64: str = Field(..., description="마스크 이미지 (base64, 흰색=교체영역)")
    product_image_base64: str = Field(..., description="삽입할 상품 이미지 (base64)")
    prompt: str = Field(..., description="inpainting 프롬프트")
    style: Optional[StyleName] = None
    num_inference_steps: int = Field(30, ge=10, le=100)
    guidance_scale: float = Field(7.5, ge=1.0, le=20.0)
    ip_adapter_scale: float = Field(0.8, ge=0.0, le=1.0)


class InpaintResult(BaseModel):
    job_id: str
    status: JobStatus
    result_image: Optional[str] = None
    error: Optional[str] = None


# ── Step 3 (확정): Product Placement ───────────────────────
class PlaceRequest(BaseModel):
    image_base64: str = Field(..., description="배경 이미지 (base64)")
    mask_base64: str = Field(..., description="배치 영역 마스크 (base64, 흰색=배치 영역)")
    product_image_base64: str = Field(..., description="삽입할 상품 이미지 (base64)")


class PlaceResult(BaseModel):
    job_id: str
    status: JobStatus
    result_image: Optional[str] = None
    error: Optional[str] = None


# ── Space Analysis ──────────────────────────────────────────
class AnalyzeRequest(BaseModel):
    image_base64: str = Field(..., description="탑뷰 이미지 (base64)")
    desk_width_cm: float = Field(..., gt=0, description="책상 가로 길이 (cm)")
    desk_depth_cm: float = Field(..., gt=0, description="책상 세로 길이 (cm)")


class AnalyzeResult(BaseModel):
    job_id: str
    status: JobStatus
    metrics: Optional[dict] = None
    available_mask: Optional[str] = None
    occupied_mask: Optional[str] = None
    placement_center: Optional[list] = None
    num_objects: int = 0
    error: Optional[str] = None
