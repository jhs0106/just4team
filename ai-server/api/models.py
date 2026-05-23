from pydantic import BaseModel, Field
from typing import Optional, List
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


class RemoveMode(str, Enum):
    add        = "add"        # 기존 물체 유지, 제품 추가
    own_desk   = "own_desk"   # 기존 물체 모두 제거 후 추가 (기본)
    replace    = "replace"    # 선택 물체만 제거 후 교체
    empty_desk = "empty_desk" # 전체 제거 후 빈 책상에 배치


class JobStatus(str, Enum):
    pending = "pending"
    running = "running"
    done = "done"
    failed = "failed"


# ── Step 1: Object Removal ──────────────────────────────
class ObjectRemovalRequest(BaseModel):
    image_base64: str = Field(..., description="원본 정면 이미지 (base64)")
    prompt: Optional[str] = Field(None, description="제거할 물체 텍스트. 제공 시 DINO+SAM2 사용")
    top_view_image_base64: Optional[str] = Field(None, description="탑뷰 이미지 (base64)")
    max_area_ratio: float = Field(0.20, ge=0.05, le=0.60, description="이 비율 초과 마스크는 배경으로 간주해 제외")


class ObjectRemovalResult(BaseModel):
    job_id: str
    status: JobStatus
    cleaned_image: Optional[str] = None
    mask_image: Optional[str] = None
    detection_overlay: Optional[str] = None
    num_objects: int = 0
    error: Optional[str] = None


# ── Step 2: Product Placement ───────────────────────────
class ProductPlaceRequest(BaseModel):
    image_base64: str = Field(..., description="배경 이미지 (base64)")
    mask_base64: str = Field(..., description="제품 배치 영역 마스크 (base64, 흰색=배치 영역)")
    prompt: str = Field(..., description="제품 설명 프롬프트")
    product_image_base64: Optional[str] = Field(None, description="실제 제품 이미지 (base64) — 제공 시 CV 합성")
    style: Optional[str] = Field("general", description="스타일 이름")
    num_inference_steps: int = Field(30, ge=10, le=100)
    guidance_scale: float = Field(8.0, ge=1.0, le=20.0)


class ProductPlaceResult(BaseModel):
    job_id: str
    status: JobStatus
    result_image: Optional[str] = None
    error: Optional[str] = None


# ── Segment ──────────────────────────────────────────────
class SegmentRequest(BaseModel):
    image_base64: str = Field(..., description="입력 이미지 (base64)")
    point_x: float = Field(..., ge=0.0, le=1.0, description="포인트 x (0~1 정규화)")
    point_y: float = Field(..., ge=0.0, le=1.0, description="포인트 y (0~1 정규화)")
    label: int = Field(1, description="1=포그라운드, 0=백그라운드")


class SegmentResult(BaseModel):
    job_id: str
    status: JobStatus
    mask_base64: Optional[str] = None
    overlay_base64: Optional[str] = None
    error: Optional[str] = None


# ── Generate ─────────────────────────────────────────────
class ProductItem(BaseModel):
    category: str           = Field(..., description="제품 카테고리 (KEYBOARD, MOUSE, MONITOR 등)")
    name: str               = Field(..., description="제품명")
    image_id: Optional[int] = Field(None, description="processed_images/{id}.png 파일 번호")
    width_mm: Optional[int] = Field(None, description="제품 실제 가로 치수 (mm)")
    depth_mm: Optional[int] = Field(None, description="제품 실제 세로 치수 (mm)")


class GenerateRequest(BaseModel):
    image_base64:          str               = Field(..., description="원본 책상 front-view 이미지 (base64)")
    style:                 StyleName         = Field(..., description="사용자 선택 스타일")
    products:              List[ProductItem] = Field(..., description="배치할 제품 목록")
    desk_width_mm:         Optional[int]     = Field(None, description="책상 실제 가로 치수 (mm) — 제품 픽셀 크기 계산 기준")
    desk_depth_mm:         Optional[int]     = Field(None, description="책상 실제 세로 치수 (mm) — top-view 공간 분석 기준")
    max_area_ratio:        float             = Field(0.20, ge=0.05, le=0.60, description="이 비율 초과 마스크는 배경으로 간주해 제외")
    top_view_image_base64: Optional[str]     = Field(None, description="탑뷰 이미지 (base64) — 제공 시 공간 분석 기반 배치")
    mode:                  RemoveMode        = Field(RemoveMode.own_desk, description="물체 제거 정책")
    generation_mode:       str               = Field("controlnet", description="생성 모드: controlnet (기본, per-product SD+IP-Adapter+ControlNet+LoRA) | cv_composite (SD 미사용) | placement_only (배치 시각화) | harmonize (실험, 폐기됨)")
    removal_strategy:      str               = Field("combined",   description="제거 방식: none | sequential | combined")


class GenerateResult(BaseModel):
    job_id:           str
    status:           JobStatus
    cleaned_image:    Optional[str] = None  # Step 1: 물체 제거 후
    composited_image: Optional[str] = None  # Step 2: CV 합성 후
    result_image:     Optional[str] = None  # Step 3: SD refinement 후
    num_removed:      int           = 0
    num_placed:       int           = 0
    error:            Optional[str] = None
