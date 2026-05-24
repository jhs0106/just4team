import json
import math
import torch
import numpy as np
import cv2
from pathlib import Path
from PIL import Image

from .config import (
    _PRODUCT_IS_FLAT_ON_DESK, _DEFAULT_DESK_TILT_DEG,
    _PRODUCT_FORM_TIER, _UPRIGHT_PROMPT_TOKEN,
    _CAT_TILT_DEG, _CAT_DEPTH_SHADING,
)


def _detect_contact_base(alpha: np.ndarray, cat: str) -> tuple[int, int, int, int] | None:
    # 제품 alpha mask에서 책상과 접지되는 base 영역 bbox 추출.
    # MONITOR: 하단 30%에서 가장 넓은 connected component (스탠드 베이스).
    # SPEAKER: 하단 20%에서 occupied width (스피커 하단).
    # 그 외: alpha 전체 bottom edge.
    h, w = alpha.shape
    if (alpha > 127).sum() == 0:
        return None
    if cat == "MONITOR":
        bot_y1 = int(h * 0.70)
        binary = ((alpha[bot_y1:h] > 127).astype(np.uint8)) * 255
        num_labels, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        if num_labels <= 1:
            return None
        _areas = [stats[i, cv2.CC_STAT_AREA] for i in range(1, num_labels)]
        largest = 1 + int(np.argmax(_areas))
        x  = int(stats[largest, cv2.CC_STAT_LEFT])
        y  = int(stats[largest, cv2.CC_STAT_TOP]) + bot_y1
        cw = int(stats[largest, cv2.CC_STAT_WIDTH])
        ch = int(stats[largest, cv2.CC_STAT_HEIGHT])
        return (x, y, x + cw, y + ch)
    elif cat == "SPEAKER":
        bot_y1 = int(h * 0.80)
        bot_region = alpha[bot_y1:h]
        rows_with = np.where((bot_region > 127).any(axis=1))[0]
        if len(rows_with) == 0:
            return None
        last_row = int(rows_with[-1])
        cols = np.where(bot_region[last_row] > 127)[0]
        if len(cols) == 0:
            return None
        return (int(cols[0]), int(bot_y1 + rows_with[0]),
                int(cols[-1]) + 1, int(bot_y1 + last_row) + 1)
    else:
        rows = np.where((alpha > 127).any(axis=1))[0]
        if len(rows) == 0:
            return None
        bot_y = int(rows[-1])
        cols  = np.where(alpha[bot_y] > 127)[0]
        if len(cols) == 0:
            return None
        return (int(cols[0]), max(0, bot_y - 5), int(cols[-1]) + 1, bot_y + 1)


def analyze_product_image_risk(product_rgba: Image.Image, category: str) -> dict:
    # 입력 product 이미지 품질/시점 위험도 분석 → adaptive generation strategy 산출.
    # 시점 수동 분류 대신 runtime 위험도 판단으로 다양한 입력 이미지에 견고하게 대응.
    # risk가 높을수록 composite 보존을 강하게(SDXL 영향력 최소화), IP-Adapter 약화.
    arr = np.array(product_rgba.convert("RGBA"))
    h_full, w_full = arr.shape[:2]
    alpha = arr[:, :, 3]
    total = h_full * w_full
    cat = category.upper()

    alpha_pixels = int((alpha > 127).sum())
    alpha_fill_ratio = alpha_pixels / max(total, 1)

    # tight bbox of alpha (canvas padding 무시한 실제 제품 비율)
    rows = np.where((alpha > 127).any(axis=1))[0]
    cols = np.where((alpha > 127).any(axis=0))[0]

    # 빈 alpha → high_risk fallback
    if len(rows) == 0 or len(cols) == 0:
        return {
            "shape_risk":                 "high_risk",
            "aspect_ratio":               0.0,
            "alpha_fill_ratio":           round(float(alpha_fill_ratio), 4),
            "bottom_contact_width_ratio": 0.0,
            "has_clear_contact_base":     False,
            "is_flat_like":               False,
            "adaptive_ip_scale":          0.25,
            "adaptive_inner_blend":       0.95,
            "adaptive_edge_blend":        0.25,
            "adaptive_strategy":          "empty_alpha_max_preserve",
            "risk_reasons":               ["empty_alpha"],
        }

    tight_w = int(cols[-1] - cols[0] + 1)
    tight_h = int(rows[-1] - rows[0] + 1)
    tight_ar = tight_w / max(tight_h, 1)

    base_bbox = _detect_contact_base(alpha, cat)
    has_clear_contact_base = base_bbox is not None
    if base_bbox is not None:
        bw = base_bbox[2] - base_bbox[0]
        bottom_contact_width_ratio = bw / max(tight_w, 1)
    else:
        bottom_contact_width_ratio = 0.0

    is_flat_like = tight_ar > 3.0

    # 카테고리별 위험도 규칙
    shape_risk    = "safe"
    risk_reasons: list[str] = []

    if cat == "MONITOR":
        if tight_ar < 1.0 or tight_ar > 3.5:
            shape_risk = "high_risk"
            risk_reasons.append(f"aspect_ratio={tight_ar:.2f} out_of[1.0,3.5]")
        elif not has_clear_contact_base:
            shape_risk = "high_risk"
            risk_reasons.append("no_contact_base")
        elif bottom_contact_width_ratio > 0.70:
            shape_risk = "risky"
            risk_reasons.append(f"contact_w_ratio={bottom_contact_width_ratio:.2f}>0.70(no_stand?)")
        elif bottom_contact_width_ratio < 0.05:
            shape_risk = "risky"
            risk_reasons.append(f"contact_w_ratio={bottom_contact_width_ratio:.2f}<0.05")
    elif cat == "SPEAKER":
        if not has_clear_contact_base:
            shape_risk = "risky"
            risk_reasons.append("no_contact_base")
        elif bottom_contact_width_ratio < 0.30:
            shape_risk = "risky"
            risk_reasons.append(f"contact_w_ratio={bottom_contact_width_ratio:.2f}<0.30")
        elif tight_ar > 2.5:
            shape_risk = "risky"
            risk_reasons.append(f"aspect_ratio={tight_ar:.2f}>2.5(wide_for_speaker)")
    elif cat == "KEYBOARD":
        if tight_ar < 2.0:
            shape_risk = "risky"
            risk_reasons.append(f"aspect_ratio={tight_ar:.2f}<2.0(too_tall)")
        elif tight_ar > 8.0:
            shape_risk = "risky"
            risk_reasons.append(f"aspect_ratio={tight_ar:.2f}>8.0(too_thin)")
    elif cat == "MOUSEPAD":
        if tight_ar < 1.2:
            shape_risk = "risky"
            risk_reasons.append(f"aspect_ratio={tight_ar:.2f}<1.2(vertical_mousepad)")
    elif cat == "MOUSE":
        if tight_ar > 2.5 or tight_ar < 0.4:
            shape_risk = "risky"
            risk_reasons.append(f"aspect_ratio={tight_ar:.2f}unusual")
    elif cat == "DESK_LAMP":
        if tight_ar > 1.5:
            shape_risk = "risky"
            risk_reasons.append(f"aspect_ratio={tight_ar:.2f}>1.5(horizontal_lamp?)")

    # form_tier × shape_risk → adaptive parameters
    form_tier = _PRODUCT_FORM_TIER.get(cat, "semi_flat")
    if form_tier == "upright":
        _table = {
            "safe":      (0.88, 0.40, 0.42),
            "risky":     (0.93, 0.30, 0.30),
            "high_risk": (0.95, 0.25, 0.25),
        }
    elif form_tier == "flat":
        _table = {
            "safe":      (0.65, 0.25, 0.55),
            "risky":     (0.80, 0.25, 0.45),
            "high_risk": (0.90, 0.20, 0.35),
        }
    else:  # semi_flat
        _table = {
            "safe":      (0.75, 0.30, 0.50),
            "risky":     (0.85, 0.25, 0.40),
            "high_risk": (0.92, 0.22, 0.30),
        }
    adaptive_inner_blend, adaptive_edge_blend, adaptive_ip_scale = _table[shape_risk]

    strategy = f"{form_tier}+{shape_risk}"
    return {
        "shape_risk":                 shape_risk,
        "aspect_ratio":               round(float(tight_ar), 3),
        "alpha_fill_ratio":           round(float(alpha_fill_ratio), 4),
        "bottom_contact_width_ratio": round(float(bottom_contact_width_ratio), 3),
        "has_clear_contact_base":     bool(has_clear_contact_base),
        "is_flat_like":               bool(is_flat_like),
        "adaptive_ip_scale":          float(adaptive_ip_scale),
        "adaptive_inner_blend":       float(adaptive_inner_blend),
        "adaptive_edge_blend":        float(adaptive_edge_blend),
        "adaptive_strategy":          strategy,
        "risk_reasons":               risk_reasons,
    }


def _warp_product_to_desk_perspective(
    product_rgba: Image.Image, depression_deg: float = _DEFAULT_DESK_TILT_DEG,
) -> Image.Image:
    # top-down 제품 이미지를 책상 perspective(3/4 view)에 맞게 변형.
    # depression_deg = 카메라 광축이 수평선 아래로 기운 각도.
    #   90°: 완전 top-down (변형 없음), 30°(기본): h를 sin(30°)=0.5로 압축.
    # taper: 위쪽이 약간 좁아짐 (perspective 원근감, 5%).
    if product_rgba.mode != "RGBA":
        product_rgba = product_rgba.convert("RGBA")
    w, h = product_rgba.size
    fore = max(0.3, min(1.0, math.sin(math.radians(depression_deg))))
    new_h = max(8, int(h * fore))
    taper = 0.05
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = np.float32([
        [w * taper, 0], [w * (1 - taper), 0],
        [w, new_h],     [0, new_h],
    ])
    M = cv2.getPerspectiveTransform(src, dst)
    arr = np.array(product_rgba)
    # RGB와 alpha를 분리해 warp — RGB는 부드러운 LANCZOS4, alpha는 NEAREST + 임계값.
    # 이유: alpha에 LANCZOS 보간 적용 시 가장자리에 partial alpha(50~200) 생성됨.
    #   → 3-zone blend에서 부분 가중치 → 제품이 투명해 보이는 현상 발생.
    rgb   = arr[:, :, :3]
    alpha = arr[:, :, 3]
    warped_rgb = cv2.warpPerspective(
        rgb, M, (w, new_h),
        flags=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    warped_alpha = cv2.warpPerspective(
        alpha, M, (w, new_h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    # NEAREST 사용에도 발생 가능한 가장자리 sub-pixel artifact 제거 (임계값 binary).
    warped_alpha = np.where(warped_alpha > 128, 255, 0).astype(np.uint8)
    warped = np.dstack([warped_rgb, warped_alpha])
    return Image.fromarray(warped, mode="RGBA")


def _letterbox_512(img: Image.Image) -> Image.Image:
    # IP-Adapter SDXL은 CLIP-H image encoder 사용 → 224×224로 자동 리사이즈됨.
    # 여기선 비율 보존만 중요하므로 512x512 letterbox 유지 (속도/품질 trade-off).
    if img.mode == "RGBA":
        bg = Image.new("RGB", img.size, (210, 210, 210))
        bg.paste(img.convert("RGB"), mask=img.getchannel("A"))
        img_rgb = bg
    else:
        img_rgb = img.convert("RGB")
    w, h = img_rgb.size
    scale = min(512 / max(w, 1), 512 / max(h, 1))
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    resized = img_rgb.resize((nw, nh), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (512, 512), (210, 210, 210))
    canvas.paste(resized, ((512 - nw) // 2, (512 - nh) // 2))
    return canvas

# === Adaptive Generation-based 2.5D Compositing (2026-05-25) ===
# 본 파이프라인은 SD를 3D 렌더러로 쓰지 않음. 단일 2D 제품 사진에서 임의 각도의
# 3D 재생성은 불가능 (3D mesh/NeRF 필요). 대신:
#   - 제품 원형은 CV composite로 강하게 보존 (입력 사진 그대로 paste)
#   - SD의 역할: edge, contact shadow, 주변 책상면, color harmonization만 담당
#   - 결과 표현: "product-preserving generative compositing" (2.5D)
#
# Adaptive 정책 (view_type 수동 분류 대신 runtime 위험도 판단):
#   - analyze_product_image_risk()가 입력 이미지의 aspect ratio, alpha fill,
#     contact base 폭 등을 분석 → safe/risky/high_risk 3-level 분류
#   - risk에 따라 IP scale, blend weights를 dynamic 조정:
#     * safe:      moderate (제품과 SDXL 균형)
#     * risky:     composite 강하게 보존, IP 약화
#     * high_risk: composite 최대 보존(0.95+), IP 최소(0.25)
#   - high_risk라도 skip 안 함 — fallback strategy로 망가지지 않게 처리
#   - 시점이 섞여 들어와도(top-down 제품 이미지 포함) 견고하게 동작
#
# Tunable constants — 비상 reference. 실제 값은 risk analysis가 산출.
_UPRIGHT_INNER_WEIGHT = 0.90   # 0.85~0.95 권장. 0.95에 가까울수록 paste 효과
_UPRIGHT_EDGE_WEIGHT  = 0.40   # 경계 자연화 강도
_SEMI_FLAT_INNER_WEIGHT = 0.75
_SEMI_FLAT_EDGE_WEIGHT  = 0.30
_FLAT_INNER_WEIGHT = 0.65
_FLAT_EDGE_WEIGHT  = 0.25

# SDXL Inpaint strength: composite_sd 보존도. 낮을수록 init image(=composite) 더 유지.
#   0.5~0.7 권장. 너무 낮으면 자연화 부족, 너무 높으면 SDXL이 새로 그려서 product identity 손실.
_INPAINT_STRENGTH = 0.65

# IP-Adapter scale — main.py에서 카테고리별 override. 여기는 기본값.
_DEFAULT_IP_SCALE = 0.40


_CAT_PROMPT = {
    "KEYBOARD":     "keyboard on desk, front view, natural lighting, sharp detail",
    "MOUSE":        "wireless mouse on desk, top-front view, natural lighting, sharp detail",
    "MOUSEPAD":     "desk mat on desk surface, thin flat rectangle, natural lighting",
    "MONITOR":      "computer monitor with thin bezel and stand, front view, natural lighting, sharp detail",
    "SPEAKER":      "desktop speaker on desk, front view, natural lighting, sharp detail",
    "DESK_LAMP":    "desk lamp with base and arm, standing on desk, natural lighting, sharp detail",
    "DESK_SHELF":   "monitor riser shelf on desk, open storage below, natural lighting",
    "LAPTOP_STAND": "laptop stand on desk, angled metal, natural lighting",
    "DECO":         "small desk decoration on desk surface, natural lighting",
    "CLOCK":        "digital desk clock, visible face, natural lighting",
}

# === Negative prompt 정책 ===
# 모든 카테고리에 공통으로 차단할 것 (퀄리티 + 명백한 실패 케이스).
# QR/barcode는 2026-05-24 MONITOR 사태 직후 추가 — IP-Adapter conditioning이
# 약할 때 SD가 학습 데이터의 모니터 화면을 따라 QR/barcode를 그리는 경향 관찰됨.
# "screen content"류 단어는 의도적으로 제외 — 모니터의 자연 화면 표시 허용.
_NEGATIVE_PROMPT = (
    "blurry, low quality, distorted, watermark, qr code, barcode, text overlay, "
    "person, face, floating, deformed, ugly, empty desk, bare desk, no product, "
    "missing object, invisible, transparent, same as background, "
    "extra objects, duplicate, multiple monitors, multiple keyboards"
)

# === 카테고리별 추가 Negative prompt ===
# 각 카테고리에서 자주 발생하는 hallucination 패턴 차단.
# 추가 시 영향 범위 명시 필수.
_CAT_NEGATIVE = {
    # MONITOR: QR/barcode/scrambled text 패턴 차단. 화면 콘텐츠는 허용 (negative에 screen 키워드 X)
    "MONITOR": (
        "keyboard, laptop, shelf, qr code, barcode, scrambled text, "
        "garbled letters, gibberish, distorted screen, broken display"
    ),
    # DESK_LAMP: 케이블만 그려지는 케이스, 기반 부재 차단
    "DESK_LAMP": (
        "cable only, wire only, floating line, no base, broken lamp, "
        "thin random curve, snake, cord"
    ),
    # MOUSE: 키보드/모니터로 그려지는 케이스 차단
    "MOUSE": (
        "large object, keyboard, monitor, floating, deformed mouse"
    ),
    # KEYBOARD: 모니터/세로 물체로 그려지는 케이스 차단
    "KEYBOARD": (
        "monitor, laptop screen, vertical object, floating keys"
    ),
    # MOUSEPAD: 두꺼운 물체로 그려지는 케이스 차단
    "MOUSEPAD": (
        "thick object, monitor, keyboard, floating mat"
    ),
    # DESK_SHELF: 벽 선반/액자로 그려지는 케이스 차단
    "DESK_SHELF": (
        "monitor, laptop, items on shelf, picture frame, wall shelf"
    ),
    # SPEAKER: 손잡이/액자/문 같은 잘못된 형태 차단
    "SPEAKER": (
        "handle, picture frame, door, arch shape, bracket"
    ),
}


_CAT_MAX_SCALE = {
    "KEYBOARD":  4.0,
    "MOUSE":     3.0,
    "SPEAKER":   3.0,
    "DESK_LAMP": 3.0,
    "MONITOR":   2.5,
    "DECO":      3.0,
    "CLOCK":     3.0,
}


class ControlNetInpaintProcessor:
    # === Adaptive Generation-based 2.5D Compositing (product-preserving) ===
    # SD1.5를 "3D 렌더러"가 아닌 "compositing naturalizer"로 사용.
    # 파이프라인:
    #   1. analyze_product_image_risk()로 입력 이미지 위험도 판단 (safe/risky/high_risk)
    #   2. risk × form_tier 매트릭스로 adaptive IP scale, blend weights 결정
    #   3. CV composite + contact shadow 합성 → composite_sd
    #   4. SD가 composite_sd를 init image로 받아 strength만큼 다시 그림
    #      (image=composite_sd. ★ image=img_sd 아님)
    #   5. 3-zone blend로 inner는 composite 강하게(risk 따라 0.65~0.95)
    #   6. Post-blend depth shading (vertical gradient AO)으로 입체감 illusion
    #
    # 입체감은 "2.5D" — 진짜 3D perspective 재해석은 불가능
    # (단일 2D 사진으로 임의 시점 렌더링은 3D mesh / NeRF / multi-view 필요).
    # 표현: "product-preserving generative compositing".
    #
    # 시점이 섞인 이미지(top-down/front/3-quarter 등)가 들어와도 risk analysis가
    # 자동으로 strategy 선택 → 망가지지 않고 적응적으로 합성.
    #
    # === 모델 구성 (2026-05-25 SD1.5 롤백) ===
    #   - SD1.5 Inpaint (runwayml/stable-diffusion-inpainting)
    #   - sd-controlnet-depth 단일 (canny 제거 — perspective 재해석 방해)
    #   - ip-adapter-plus_sd15 (CLIP-ViT-L 기본 image encoder)
    #   - LoRA JU_DeskStyle (PEFT, outputs/models/lora_external) — SD1.5 호환
    #   - 해상도 512 (SD1.5 native), GPU 전체 상주 (12GB 여유)
    #
    # 카테고리 form tier × shape risk → adaptive params:
    #   flat × safe:           inner 0.65 / edge 0.25 / ip 0.55
    #   flat × risky:          inner 0.80 / edge 0.25 / ip 0.45
    #   flat × high_risk:      inner 0.90 / edge 0.20 / ip 0.35
    #   semi_flat × safe:      inner 0.75 / edge 0.30 / ip 0.50
    #   semi_flat × risky:     inner 0.85 / edge 0.25 / ip 0.40
    #   semi_flat × high_risk: inner 0.92 / edge 0.22 / ip 0.30
    #   upright × safe:        inner 0.88 / edge 0.40 / ip 0.42
    #   upright × risky:       inner 0.93 / edge 0.30 / ip 0.30
    #   upright × high_risk:   inner 0.95 / edge 0.25 / ip 0.25

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype  = torch.float16 if self.device == "cuda" else torch.float32
        print("[ControlNetInpaint/SD1.5] 모델 로드 중...")
        self._load_depth_model()
        self._load_pipeline()
        print("[ControlNetInpaint/SD1.5] 로드 완료.")

    def _load_depth_model(self):
        from transformers import DPTImageProcessor, DPTForDepthEstimation
        self._depth_proc  = DPTImageProcessor.from_pretrained("Intel/dpt-large")
        self._depth_model = DPTForDepthEstimation.from_pretrained(
            "Intel/dpt-large", low_cpu_mem_usage=False
        ).to(self.device)
        self._depth_model.eval()

    def _load_pipeline(self):
        from diffusers import (
            ControlNetModel,
            StableDiffusionControlNetInpaintPipeline,
            DPMSolverMultistepScheduler,
        )

        # ControlNet은 depth만 사용 (canny 제거).
        # 이유: canny는 평면 product silhouette을 강제해 perspective 재해석 방해.
        # SD1.5에서 VRAM은 여유 있지만 동일 정책 유지 (compositing 안정성).
        cn_depth = ControlNetModel.from_pretrained(
            "lllyasviel/sd-controlnet-depth",
            torch_dtype=self.dtype,
        )

        # SD1.5 Inpaint + ControlNet pipeline.
        # variant="fp16" 미지정 — runwayml/stable-diffusion-inpainting은 fp16 variant 없음.
        pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
            "runwayml/stable-diffusion-inpainting",
            controlnet=cn_depth,
            torch_dtype=self.dtype,
            safety_checker=None,
        )
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            pipe.scheduler.config,
            algorithm_type="dpmsolver++",
            use_karras_sigmas=True,
        )
        pipe.vae.enable_slicing()
        # SD1.5 + ControlNet + IP-Adapter @ 512는 12GB GPU에 여유 있음 → CPU offload 불필요.
        # 전체 GPU 상주가 속도 빠름.
        pipe.to(self.device)

        # IP-Adapter Plus SD1.5 변형 (CLIP-L 기본 image encoder 사용, 별도 명시 불필요)
        pipe.load_ip_adapter(
            "h94/IP-Adapter",
            subfolder="models",
            weight_name="ip-adapter-plus_sd15.bin",
        )
        pipe.set_ip_adapter_scale(0.7)

        # LoRA: PEFT format adapter (outputs/models/lora_external).
        # SD1.5 base와 호환 — 학습된 책상 스타일 톤/조명 복원.
        self._has_lora = False
        _lora_dir = Path(__file__).parent.parent / "outputs" / "models" / "lora_external"
        if _lora_dir.exists():
            try:
                from peft import PeftModel
                pipe.unet = PeftModel.from_pretrained(pipe.unet, str(_lora_dir))
                self._has_lora = True
                print(f"[ControlNetInpaint/SD1.5] LoRA 로드 완료: {_lora_dir.name}")
            except Exception as _le:
                print(f"[ControlNetInpaint/SD1.5] LoRA 로드 실패: {_le}")

        self.pipe = pipe

    def _sd_size(self, w: int, h: int) -> tuple[int, int]:
        # SD1.5 native 해상도 512. 긴 변=512, 짧은 변은 비율 보존, 8의 배수.
        base = 512
        if w >= h:
            sw = base
            sh = max(256, round(h * base / w / 8) * 8)
        else:
            sh = base
            sw = max(256, round(w * base / h / 8) * 8)
        return sw, sh

    def _get_depth(self, image: Image.Image) -> Image.Image:
        # DPT-Large로 depth map 생성 → 3채널 RGB 반환
        inputs = self._depth_proc(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            depth = self._depth_model(**inputs).predicted_depth
        depth = torch.nn.functional.interpolate(
            depth.unsqueeze(1),
            size=image.size[::-1],
            mode="bicubic",
            align_corners=False,
        ).squeeze().cpu().numpy()
        depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8) * 255
        return Image.fromarray(depth.astype(np.uint8)).convert("RGB")

    def _get_canny(self, image: Image.Image, low: int = 80, high: int = 180) -> Image.Image:
        arr = cv2.Canny(np.array(image.convert("L")), low, high)
        return Image.fromarray(np.stack([arr] * 3, axis=-1))

    def generate_product(
        self,
        image: Image.Image,
        mask: Image.Image,
        product_image: Image.Image,
        category: str,
        style: str,
        num_inference_steps: int = 28,
        guidance_scale: float = 9.0,
        ip_adapter_scale: float = 0.4,
        controlnet_scale: float = 0.6,
        lora_scale: float = 0.65,
        context_region: tuple | None = None,
        debug_dir: Path | None = None,
        debug_meta: dict | None = None,
        variant_prefix: str | None = None,
        cn_scales_override: list | None = None,
        lora_scale_override: float | None = None,
    ) -> Image.Image:
        iw, ih = image.size
        cat = category.upper()

        # 1. placement bbox from mask
        mask_np = np.array(mask.convert("L"))
        ys, xs  = np.where(mask_np > 127)
        if len(xs) == 0:
            print(f"  [generate_product] {cat} mask 비어있음 — skip")
            return image

        x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
        pw, ph = max(1, x2 - x1), max(1, y2 - y1)

        # 2. context crop with padding
        pad    = min(int(max(pw, ph) * 0.5), 160)
        cx1    = max(0, x1 - pad); cy1 = max(0, y1 - pad)
        cx2    = min(iw, x2 + pad); cy2 = min(ih, y2 + pad)
        cw, ch = cx2 - cx1, cy2 - cy1

        crop_img  = image.crop((cx1, cy1, cx2, cy2))
        sd_w, sd_h = self._sd_size(cw, ch)

        img_sd   = crop_img.resize((sd_w, sd_h), Image.Resampling.LANCZOS).convert("RGB")
        mask_sd  = mask.crop((cx1, cy1, cx2, cy2)).resize((sd_w, sd_h), Image.Resampling.NEAREST).convert("L")

        # product bbox in SD coords (relative to crop origin)
        sx = sd_w / max(cw, 1); sy = sd_h / max(ch, 1)
        bx1 = int((x1 - cx1) * sx); bx2 = int((x2 - cx1) * sx)
        by1 = int((y1 - cy1) * sy); by2 = int((y2 - cy1) * sy)
        bw  = max(1, bx2 - bx1);   bh  = max(1, by2 - by1)

        _white_px = int((np.array(mask_sd) > 127).sum())
        print(f"  [generate_product] {cat} white_px={_white_px} crop={cw}x{ch} sd={sd_w}x{sd_h}")

        # 3. product RGBA — main.py에서 이미 tight-crop 처리된 상태
        prod_rgba = product_image if product_image.mode == "RGBA" else product_image.convert("RGBA")

        # === Adaptive risk analysis (warp 전 원본 기준) ===
        # 시점 수동 분류 대신 입력 이미지 품질/형태 기반 runtime 위험도 판단.
        # 결과에 따라 IP scale, blend weights를 동적으로 조정 → 다양한 입력 견고 대응.
        _risk = analyze_product_image_risk(prod_rgba, cat)
        _orig_ip_scale = ip_adapter_scale
        ip_adapter_scale = _risk["adaptive_ip_scale"]
        print(f"  [risk] {cat} {_risk['adaptive_strategy']} "
              f"ar={_risk['aspect_ratio']} contact_w_ratio={_risk['bottom_contact_width_ratio']} "
              f"ip:{_orig_ip_scale:.2f}->{ip_adapter_scale:.2f}")
        if _risk["risk_reasons"]:
            print(f"  [risk] {cat} reasons: {_risk['risk_reasons']}")

        # 3-tier 카테고리 분류에 따른 처리:
        #   flat (KEYBOARD/MOUSEPAD): perspective warp 적용
        #   semi_flat (MOUSE/LAPTOP_STAND): warp 미적용, upright prompt 미적용
        #   upright (MONITOR/SPEAKER/DESK_LAMP 등): warp 절대 금지, upright prompt 추가
        _form_tier   = _PRODUCT_FORM_TIER.get(cat, "semi_flat")
        _is_flat     = (_form_tier == "flat")
        _is_upright  = (_form_tier == "upright")
        _warp_applied = False
        _tilt_deg = _CAT_TILT_DEG.get(cat, _DEFAULT_DESK_TILT_DEG)
        if _is_flat:
            prod_rgba = _warp_product_to_desk_perspective(prod_rgba, _tilt_deg)
            _warp_applied = True
            print(f"  [perspective_warp] {cat} depression={_tilt_deg}° → {prod_rgba.size}")
        else:
            print(f"  [form_tier] {cat} = {_form_tier} (warp skip)")

        _prod_orig_w, _prod_orig_h = prod_rgba.width, prod_rgba.height

        # fit product into SD placement bbox (category max scale 적용, 1.0 cap 제거)
        _cat_max_scale = _CAT_MAX_SCALE.get(cat, 2.0)
        _sc  = min(bw / max(prod_rgba.width, 1), bh / max(prod_rgba.height, 1), _cat_max_scale)
        _fpw = max(8, int(prod_rgba.width * _sc))
        _fph = max(8, int(prod_rgba.height * _sc))

        # KEYBOARD horizontal stretch 비활성화 (2026-05-24)
        # 이유: stretch 발동 시 세로로 짜부러진 왜곡 키보드 + SD가 정상 비율로 재생성 →
        #      "키보드가 2개로 나뉘어 보임" 현상 발생.
        # 자연 비율 유지가 시각적으로 더 자연스러움 (bbox에 여백 있어도 OK).
        # if cat == "KEYBOARD" and _fpw < bw * 0.80:
        #     _fpw = max(8, int(bw * 0.90))

        # MONITOR 최소 fill 0.82: aspect ratio 유지하며 확장, height cap 초과 시 height 기준 재계산
        _monitor_min_fill_applied = False
        if cat == "MONITOR" and _fpw < bw * 0.82:
            _mon_w = max(8, int(bw * 0.88))
            _mon_h = int(prod_rgba.height * _mon_w / max(prod_rgba.width, 1))
            if _mon_h <= bh:
                _fpw, _fph = _mon_w, _mon_h
                _monitor_min_fill_applied = True
            else:
                # height가 bbox를 넘으면 height 기준으로 최대 너비 재계산
                _mon_w2 = int(bh * prod_rgba.width / max(prod_rgba.height, 1))
                if _mon_w2 > _fpw:
                    _fpw, _fph = _mon_w2, bh
                    _monitor_min_fill_applied = True
            if _monitor_min_fill_applied:
                print(f"  [MONITOR min fill] width → {_fpw} (fill={_fpw/max(bw,1):.2f})")

        _prod_fit = prod_rgba.resize((_fpw, _fph), Image.Resampling.LANCZOS)

        _ppx = bx1 + (bw - _fpw) // 2
        # contact alignment: alpha object bottom → bbox bottom (not image bottom)
        _alpha_arr_raw = np.array(prod_rgba.getchannel("A"))
        _rows_with_alpha = np.where((_alpha_arr_raw > 127).any(axis=1))[0]
        _alpha_bottom_norm = (_rows_with_alpha[-1] / max(prod_rgba.height - 1, 1)) if len(_rows_with_alpha) > 0 else 1.0
        _obj_bottom_in_fit = max(1, int(_fph * _alpha_bottom_norm))
        _ppy = by2 - _obj_bottom_in_fit
        _applied_contact_shift = _fph - _obj_bottom_in_fit  # image bottom보다 위로 올라간 px
        print(f"  [contact_align] {cat} alpha_bottom={_alpha_bottom_norm:.3f} obj_bottom_fit={_obj_bottom_in_fit} shift={_applied_contact_shift}")

        _pc1x = max(0, _ppx);             _pc2x = min(sd_w, _ppx + _fpw)
        _pc1y = max(0, _ppy);             _pc2y = min(sd_h, _ppy + _fph)
        _src1x = max(0, -_ppx);           _src1y = max(0, -_ppy)
        _src2x = _src1x + (_pc2x - _pc1x); _src2y = _src1y + (_pc2y - _pc1y)

        print(f"  [generate_product] {cat} tight={_prod_orig_w}x{_prod_orig_h} "
              f"scale={_sc:.3f}(max={_cat_max_scale}) fit={_fpw}x{_fph} bbox_sd={bw}x{bh}")

        # B. CV composite 선행: img_sd에 제품 합성
        composite_sd = img_sd.copy().convert("RGBA")
        if _pc2x > _pc1x and _pc2y > _pc1y:
            _slice = _prod_fit.crop((_src1x, _src1y, _src2x, _src2y))
            composite_sd.alpha_composite(_slice, (_pc1x, _pc1y))
        composite_sd = composite_sd.convert("RGB")

        # === Contact base 검출 + shadow on composite ===
        # 목적: 그림자를 SDXL 입력 전에 composite에 미리 그려넣어 SDXL이 "그림자 있는 제품"을
        #   보고 주변 책상면을 자연화. 이전 후처리(_add_shadows)가 별도로 그리면 분리되어 보임.
        # inpaint mask도 shadow + floor 영역 포함하도록 확장 → SDXL이 그림자 영역도 자연 생성.
        _fit_alpha = np.array(_prod_fit.getchannel("A"))
        _contact_base_local = _detect_contact_base(_fit_alpha, cat)
        _contact_base_sd: tuple[int, int, int, int] | None = None
        _shadow_layer_arr: np.ndarray | None = None

        if _contact_base_local is not None and _pc2x > _pc1x and _pc2y > _pc1y:
            # _prod_fit 로컬 좌표 → SD canvas 좌표 변환 (paste 위치 + crop offset 보정)
            _cb_x1 = _pc1x + max(0, _contact_base_local[0] - _src1x)
            _cb_y1 = _pc1y + max(0, _contact_base_local[1] - _src1y)
            _cb_x2 = _pc1x + min(_pc2x - _pc1x, _contact_base_local[2] - _src1x)
            _cb_y2 = _pc1y + min(_pc2y - _pc1y, _contact_base_local[3] - _src1y)
            _contact_base_sd = (_cb_x1, _cb_y1, _cb_x2, _cb_y2)
            _cb_cx = (_cb_x1 + _cb_x2) // 2
            _cb_w  = max(1, _cb_x2 - _cb_x1)

            # 카테고리별 contact shadow 파라미터
            if cat == "MONITOR":
                _sh_hw, _sh_h, _sh_blur, _sh_str = max(_cb_w // 2, 8), 4, 9, 0.55
            elif cat == "SPEAKER":
                _sh_hw, _sh_h, _sh_blur, _sh_str = max(_cb_w // 2, 6), 4, 7, 0.55
            else:
                _sh_hw, _sh_h, _sh_blur, _sh_str = max(_cb_w // 3, 6), 3, 5, 0.40

            _shadow_layer_arr = np.zeros((sd_h, sd_w), dtype=np.float32)
            cv2.ellipse(_shadow_layer_arr, (_cb_cx, _cb_y2), (_sh_hw, _sh_h), 0, 0, 360, 1.0, -1)
            _shadow_layer_arr = cv2.GaussianBlur(_shadow_layer_arr, (_sh_blur, _sh_blur), 0)
            _shadow_layer_arr = np.clip(_shadow_layer_arr * _sh_str, 0.0, 0.55)

            # composite_sd에 shadow darkening 적용 (SDXL 입력에 미리 포함)
            _comp_arr = np.array(composite_sd).astype(np.float32)
            _comp_arr = _comp_arr * (1.0 - _shadow_layer_arr[:, :, np.newaxis])
            composite_sd = Image.fromarray(np.clip(_comp_arr, 0, 255).astype(np.uint8))
            print(f"  [contact_base] {cat} bbox_sd=({_cb_x1},{_cb_y1},{_cb_x2},{_cb_y2}) "
                  f"shadow=hw{_sh_hw}/blur{_sh_blur}/str{_sh_str}")
        else:
            print(f"  [contact_base] {cat} 검출 실패 — shadow skip")

        # color matching: composite_sd는 SDXL 입력용 — 적용 안 함
        _color_matched_composite_sd = composite_sd.copy()

        # C. silhouette mask: alpha 있으면 전 카테고리 적용, 없으면 bbox fallback.
        # 추가: contact base 검출됐으면 shadow + floor 영역도 inpaint mask에 포함.
        # → SDXL이 그림자 자체와 그림자 주변 책상면도 자연 생성.
        _has_alpha = np.array(prod_rgba.getchannel("A")).min() < 250
        mask_type = "bbox"
        if _has_alpha and _pc2x > _pc1x and _pc2y > _pc1y:
            _sil = np.zeros((sd_h, sd_w), dtype=np.uint8)
            _alpha_slice = np.array(_prod_fit.getchannel("A"))[_src1y:_src2y, _src1x:_src2x]
            _sil[_pc1y:_pc2y, _pc1x:_pc2x] = _alpha_slice
            _k   = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
            _sil = cv2.dilate(_sil, _k)
            _sil = cv2.GaussianBlur(_sil, (7, 7), 0)
            mask_type = "silhouette"
            # contact base 있으면 shadow region + floor strip 추가
            if _contact_base_sd is not None and _shadow_layer_arr is not None:
                _cb_x1, _cb_y1, _cb_x2, _cb_y2 = _contact_base_sd
                _floor_y1 = _cb_y2
                _floor_y2 = min(sd_h, _cb_y2 + max(8, (_cb_y2 - _cb_y1) // 2 + 8))
                _floor_x1 = max(0, _cb_x1 - 12)
                _floor_x2 = min(sd_w, _cb_x2 + 12)
                _floor_mask = np.zeros((sd_h, sd_w), dtype=np.uint8)
                _floor_mask[_floor_y1:_floor_y2, _floor_x1:_floor_x2] = 255
                _shadow_mask_u8 = (_shadow_layer_arr * 4 * 255 / 0.55).clip(0, 255).astype(np.uint8)
                _expansion = np.maximum(_floor_mask, _shadow_mask_u8)
                _expansion = cv2.GaussianBlur(_expansion, (7, 7), 0)
                _sil = np.maximum(_sil, _expansion)
                mask_type = "silhouette+contact_shadow+floor"
            mask_sd = Image.fromarray(_sil).convert("L")
            print(f"  [generate_product] {cat} mask_type={mask_type}")
        else:
            print(f"  [generate_product] {cat} bbox mask 사용 (has_alpha={_has_alpha})")

        # 4. ControlNet: depth(빈 배경, 책상 구조)만 사용. canny 제거됨.
        # 이유: canny는 평면 product silhouette을 강제해 perspective 재해석 방해.
        depth_sd = self._get_depth(img_sd)

        # 5. IP-Adapter: letterbox 비율 유지 512×512.
        # warp이 적용됐다면 warped prod_rgba 사용해 SD가 perspective 일관된 reference 받음.
        prod_ip = _letterbox_512(prod_rgba if _warp_applied else product_image)

        # 카테고리별 depth 가중치 (canny 제거 후 단일 ControlNet).
        # 평면 카테고리는 책상 plane geometry가 perspective 결정에 중요 → 더 강하게.
        if cn_scales_override is not None:
            cn_scales = cn_scales_override[0] if isinstance(cn_scales_override, list) else cn_scales_override
        elif cat in ("KEYBOARD", "MOUSE", "MOUSEPAD"):
            cn_scales = 0.45
        elif cat == "MONITOR":
            cn_scales = 0.30
        else:
            cn_scales = 0.35

        _effective_lora_scale = lora_scale if lora_scale_override is None else lora_scale_override
        cat_desc = _CAT_PROMPT.get(cat, "product on desk, natural lighting")
        lora_token = "JU_Style, " if (self._has_lora and _effective_lora_scale > 0) else ""
        # 평면 카테고리: SD에 perspective 명시. depth ControlNet + warp과 함께 정렬 보강.
        # upright 카테고리: 수직으로 서 있음 명시 (모니터/스피커/스탠드 등).
        if _is_flat:
            _form_token = "viewed from above at angle, foreshortened, matching desk perspective, "
        elif _is_upright:
            _form_token = _UPRIGHT_PROMPT_TOKEN
        else:
            _form_token = ""
        prompt   = (
            f"{lora_token}{_form_token}{cat_desc}, {style} style, "
            "on desk surface, drop shadow, photorealistic, sharp focus"
        )
        negative_prompt = _NEGATIVE_PROMPT + (
            ", " + _CAT_NEGATIVE[cat] if cat in _CAT_NEGATIVE else ""
        )

        self.pipe.set_ip_adapter_scale(ip_adapter_scale)
        # SD1.5에선 CPU offload 안 씀 — 파이프라인은 init 시 GPU 상주됨.

        # === [핵심] composite_sd를 SD init image로 전달 ===
        # 이전 버그: image=img_sd(빈 책상)였음 → SD가 composite 못 보고 새로 그림.
        # 수정: image=composite_sd(제품+그림자 합성됨) → SD가 init에서 시작해
        #   strength 만큼만 다시 그림 → product identity 보존하며 edge/floor만 자연화.
        # generation-based 2.5D compositing 철학의 핵심 구현.
        pipe_kwargs = dict(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=composite_sd,
            mask_image=mask_sd,
            control_image=depth_sd,
            ip_adapter_image=prod_ip,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            controlnet_conditioning_scale=cn_scales,
            strength=_INPAINT_STRENGTH,
            width=sd_w,
            height=sd_h,
        )
        if self._has_lora and _effective_lora_scale > 0:
            pipe_kwargs["cross_attention_kwargs"] = {"scale": _effective_lora_scale}

        # Pass 1: single-pass (pass2는 hallucination 유발 — 비활성화)
        result_sd_pass1 = self.pipe(**pipe_kwargs).images[0]
        result_sd = result_sd_pass1

        # === Tier-specific 3-zone blend 정책 ===
        # upright (MONITOR/SPEAKER/DESK_LAMP): CV composite 강하게 보존 (SDXL이 모니터 화면/스피커 그릴 거 기대 X).
        # flat   (KEYBOARD/MOUSEPAD): 중간 보존 (warp으로 perspective 이미 적용).
        # semi_flat (MOUSE/LAPTOP_STAND): 중간.
        #
        # tier        | inner | edge  | 의도
        # ------------|-------|-------|--------
        # upright     | 0.90  | 0.40  | 제품 원형 강하게 유지, 경계만 SDXL이 책상 톤과 융합
        # semi_flat   | 0.75  | 0.30  | 제품 70% 유지, edge 자연화
        # flat        | 0.65  | 0.25  | 기존 (warped product에 SDXL이 좀 더 자유롭게)
        # blend weights: risk analysis가 산출한 adaptive 값 사용 (static constant fallback 아님).
        # _UPRIGHT/_FLAT_*_WEIGHT 상수는 비상시 reference로만 보존.
        _blend_inner_w = _risk["adaptive_inner_blend"]
        _blend_edge_w  = _risk["adaptive_edge_blend"]
        print(f"  [blend] {cat} tier={_form_tier} risk={_risk['shape_risk']} "
              f"inner={_blend_inner_w} edge={_blend_edge_w}")

        _blend_inner_arr = _blend_edge_arr = _blend_result_arr = None
        if _has_alpha and _pc2x > _pc1x and _pc2y > _pc1y:
            _min_dim = max(1, min(_fpw, _fph))
            _ek_inner = max(5, min(15, _min_dim // 5))
            if _ek_inner % 2 == 0:
                _ek_inner += 1
            _ek_edge = max(3, min(9, _min_dim // 10))
            if _ek_edge % 2 == 0:
                _ek_edge += 1
            _ia = np.array(_prod_fit.getchannel("A")).astype(np.float32) / 255.0
            _sil_w = np.zeros((sd_h, sd_w), dtype=np.float32)
            _sil_w[_pc1y:_pc2y, _pc1x:_pc2x] = _ia[_src1y:_src2y, _src1x:_src2x]
            _sil_uint8 = (_sil_w * 255).astype(np.uint8)
            _k_inner = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (_ek_inner, _ek_inner))
            _inner_raw = cv2.erode(_sil_uint8, _k_inner).astype(np.float32) / 255.0
            _blur_inner = max(3, _ek_inner * 2 + 1)
            if _blur_inner % 2 == 0:
                _blur_inner += 1
            _inner_smooth = cv2.GaussianBlur(_inner_raw, (_blur_inner, _blur_inner), 0)
            _k_edge = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (_ek_edge, _ek_edge))
            _sil_light = cv2.erode(_sil_uint8, _k_edge).astype(np.float32) / 255.0
            _blur_edge = max(3, _ek_edge * 2 + 1)
            if _blur_edge % 2 == 0:
                _blur_edge += 1
            _sil_light_smooth = cv2.GaussianBlur(_sil_light, (_blur_edge, _blur_edge), 0)
            _edge_ring_smooth = np.clip(_sil_light_smooth - _inner_smooth, 0.0, 1.0)
            _blend_w = np.clip(_inner_smooth * _blend_inner_w + _edge_ring_smooth * _blend_edge_w, 0.0, 1.0)
            # dark rim occlusion: KEYBOARD/MOUSE — contact 하단 2~4px 어둡게
            if cat in ("KEYBOARD", "MOUSE"):
                _rim_h = max(2, min(4, max(_fph, 1) // 20))
                _rim_y1 = max(0, _pc2y - _rim_h)
                _rim_mask = np.zeros((sd_h, sd_w), dtype=np.float32)
                _rim_mask[_rim_y1:_pc2y, _pc1x:_pc2x] = _sil_w[_rim_y1:_pc2y, _pc1x:_pc2x]
                _rim_mask = cv2.GaussianBlur(_rim_mask, (5, 5), 0)
                _ra_arr = np.array(result_sd).astype(np.float32)
                result_sd = Image.fromarray(
                    np.clip(_ra_arr * (1.0 - _rim_mask[:, :, np.newaxis] * 0.30), 0, 255).astype(np.uint8)
                )
            _ca = np.array(composite_sd).astype(np.float32)
            _ra = np.array(result_sd).astype(np.float32)
            _blended = np.clip(
                _ca * _blend_w[:, :, np.newaxis] + _ra * (1 - _blend_w[:, :, np.newaxis]), 0, 255
            ).astype(np.uint8)
            result_sd = Image.fromarray(_blended)
            _blend_inner_arr  = (_inner_smooth * 255).astype(np.uint8)
            _blend_edge_arr   = (_edge_ring_smooth * 255).astype(np.uint8)
            _blend_result_arr = _blended

            # === Depth shading (post-blend AO 효과) ===
            # 제품 silhouette 내부에 vertical gradient darkening 적용 → 2.5D 입체감 illusion.
            # 위쪽=1.0 (조명 받는 면), 아래쪽=1.0 - shading_max (그림자 면).
            # SDXL output에 직접 적용하는 게 아니라 blended 결과에 — SDXL confuse 방지.
            _shading_max = _CAT_DEPTH_SHADING.get(cat, 0.10)
            if _shading_max > 0:
                _sil_bin = (_sil_w > 0.5).astype(np.float32)
                _sil_rows = np.where(_sil_bin.any(axis=1))[0]
                if len(_sil_rows) > 0:
                    _pmin, _pmax = int(_sil_rows[0]), int(_sil_rows[-1])
                    _phgt = max(1, _pmax - _pmin + 1)
                    _y_rel = (np.arange(sd_h) - _pmin) / _phgt
                    _y_rel = np.clip(_y_rel, 0.0, 1.0)
                    _vert_factor = 1.0 - _y_rel * _shading_max
                    _depth_map = np.ones((sd_h, sd_w), dtype=np.float32)
                    _depth_map[:, :] = _vert_factor[:, np.newaxis]
                    _depth_factor = 1.0 - _sil_bin * (1.0 - _depth_map)
                    _rsd_arr = np.array(result_sd).astype(np.float32)
                    _rsd_arr = _rsd_arr * _depth_factor[:, :, np.newaxis]
                    result_sd = Image.fromarray(np.clip(_rsd_arr, 0, 255).astype(np.uint8))
                    print(f"  [depth_shading] {cat} max_darken={_shading_max} prod_y=[{_pmin},{_pmax}]")

        # refine mask: pass2 비활성화 상태에서도 debug/paste용으로 silhouette 확장만
        _refine_mask_arr = np.array(mask_sd)
        _dil_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        _refine_mask_arr = cv2.dilate(_refine_mask_arr, _dil_k)
        _refine_mask_arr = cv2.GaussianBlur(_refine_mask_arr, (5, 5), 0)
        _refine_mask = Image.fromarray(_refine_mask_arr).convert("L")

        # SD1.5 + ControlNet + IP-Adapter @ 512는 12GB GPU에 여유 — offload 불필요.
        # 다음 inference에 메모리 안정 위해 empty_cache만 호출.
        torch.cuda.empty_cache()

        # 6. 원본 크기 복원: paste mask는 SD inpaint mask(부드러운)와 분리.
        # 원본 제품 alpha를 직접 사용 → sharp edge로 "floating/ghost" 효과 제거.
        # 작은 anti-aliasing(3x3 blur)만 적용.
        # 기존엔 dilated+blurred silhouette를 또 blur해서 4겹 페이드 발생 → 제품이 떠 있어 보임.
        result_crop = result_sd.resize((cw, ch), Image.Resampling.LANCZOS)
        output      = image.copy()

        if _has_alpha and _pc2x > _pc1x and _pc2y > _pc1y:
            _sharp_paste_sd = np.zeros((sd_h, sd_w), dtype=np.uint8)
            _sharp_paste_sd[_pc1y:_pc2y, _pc1x:_pc2x] = np.array(
                _prod_fit.getchannel("A")
            )[_src1y:_src2y, _src1x:_src2x]
            # shadow 영역도 paste mask에 포함 — SDXL이 그린 shadow가 output에 반영되도록.
            if _shadow_layer_arr is not None:
                _sh_paste_u8 = (_shadow_layer_arr * 4 * 255 / 0.55).clip(0, 255).astype(np.uint8)
                _sharp_paste_sd = np.maximum(_sharp_paste_sd, _sh_paste_u8)
            _sharp_paste_sd = cv2.GaussianBlur(_sharp_paste_sd, (3, 3), 0)
            final_paste_mask = Image.fromarray(_sharp_paste_sd).resize(
                (cw, ch), Image.Resampling.LANCZOS,
            )
        else:
            final_paste_mask = mask_sd.resize((cw, ch), Image.Resampling.LANCZOS).filter(
                __import__("PIL.ImageFilter", fromlist=["GaussianBlur"]).GaussianBlur(radius=1)
            )
        output.paste(result_crop, (cx1, cy1), mask=final_paste_mask)

        # debug 파일 저장
        if debug_dir is not None:
            try:
                _ddir = Path(debug_dir)
                _ddir.mkdir(parents=True, exist_ok=True)
                _prefix = variant_prefix or cat
                img_sd.save(_ddir / f"{_prefix}_crop_img.png")
                composite_sd.save(_ddir / f"{_prefix}_composite_with_shadow_sd.png")
                mask_sd.save(_ddir / f"{_prefix}_mask_sd.png")
                _refine_mask.save(_ddir / f"{_prefix}_refine_mask_sd.png")
                final_paste_mask.save(_ddir / f"{_prefix}_final_paste_mask.png")
                result_sd_pass1.save(_ddir / f"{_prefix}_pass1_result_sd.png")
                prod_ip.save(_ddir / f"{_prefix}_prod_ip.png")
                # contact base + shadow debug
                if _contact_base_sd is not None:
                    _cb_mask = np.zeros((sd_h, sd_w), dtype=np.uint8)
                    _cb_x1, _cb_y1, _cb_x2, _cb_y2 = _contact_base_sd
                    _cb_mask[_cb_y1:_cb_y2, _cb_x1:_cb_x2] = 255
                    Image.fromarray(_cb_mask).save(_ddir / f"{_prefix}_contact_base_mask.png")
                if _shadow_layer_arr is not None:
                    Image.fromarray(
                        (_shadow_layer_arr * 255 / 0.55).clip(0, 255).astype(np.uint8)
                    ).save(_ddir / f"{_prefix}_contact_shadow_mask.png")
                if _blend_inner_arr is not None:
                    Image.fromarray(_blend_inner_arr).save(_ddir / f"{_prefix}_inner_mask.png")
                    Image.fromarray(_blend_edge_arr).save(_ddir / f"{_prefix}_edge_ring_mask.png")
                    Image.fromarray(_blend_result_arr).save(_ddir / f"{_prefix}_blend_result.png")
                _m = debug_meta or {}
                _actual_img_w = int(_fpw * cw / max(sd_w, 1))
                _actual_img_h = int(_fph * ch / max(sd_h, 1))
                _dbg_json = {
                    "category":                     cat,
                    "image_id":                     _m.get("image_id"),
                    "width_mm":                     _m.get("width_mm"),
                    "depth_mm":                     _m.get("depth_mm"),
                    "height_mm":                    _m.get("height_mm"),
                    "target_bbox_width_px":         pw,
                    "target_bbox_height_px":        ph,
                    "product_original_width_px":    _m.get("product_raw_w"),
                    "product_original_height_px":   _m.get("product_raw_h"),
                    "product_tight_crop_width_px":  _prod_orig_w,
                    "product_tight_crop_height_px": _prod_orig_h,
                    "actual_composite_width_px":    _actual_img_w,
                    "actual_composite_height_px":   _actual_img_h,
                    "actual_sd_width_px":           _fpw,
                    "actual_sd_height_px":          _fph,
                    "sd_input_size":                f"{sd_w}x{sd_h}",
                    "context_crop_size":            f"{cw}x{ch}",
                    "scale_to_bbox":                round(_sc, 4),
                    "category_max_scale":           _cat_max_scale,
                    "mask_type":                    mask_type,
                    "final_paste_mask_type":        "silhouette_blur_r3",
                    "pass2_enabled":                False,
                    "monitor_min_fill_applied":     _monitor_min_fill_applied,
                    "has_lora":                     self._has_lora,
                    "lora_scale_applied":           _effective_lora_scale if self._has_lora else None,
                    "prompt_has_lora_token":        bool(lora_token),
                    "ip_adapter_scale":             ip_adapter_scale,
                    "cn_scales":                    cn_scales,
                    "prompt":                       prompt,
                    "negative_prompt":              negative_prompt,
                }
                (_ddir / f"{_prefix}_debug.json").write_text(
                    json.dumps(_dbg_json, indent=2, ensure_ascii=False), encoding="utf-8"
                )
                _contact_info = {
                    "form_tier":              _form_tier,
                    "warp_applied":           bool(_warp_applied),
                    "ip_adapter_scale":       float(ip_adapter_scale),
                    "ip_adapter_scale_original": float(_orig_ip_scale),
                    "inpaint_strength":       float(_INPAINT_STRENGTH),
                    "sd_init_image":          "composite_sd_with_shadow",
                    "blend_inner_weight":     float(_blend_inner_w),
                    "blend_edge_weight":      float(_blend_edge_w),
                    "risk_shape_risk":               _risk["shape_risk"],
                    "risk_aspect_ratio":             _risk["aspect_ratio"],
                    "risk_alpha_fill_ratio":         _risk["alpha_fill_ratio"],
                    "risk_bottom_contact_w_ratio":   _risk["bottom_contact_width_ratio"],
                    "risk_has_clear_contact_base":   _risk["has_clear_contact_base"],
                    "risk_is_flat_like":             _risk["is_flat_like"],
                    "risk_adaptive_strategy":        _risk["adaptive_strategy"],
                    "risk_reasons":                  _risk["risk_reasons"],
                    "warp_tilt_deg":          float(_tilt_deg) if _warp_applied else None,
                    "depth_shading_max":      float(_CAT_DEPTH_SHADING.get(cat, 0.10)),
                    "alpha_bottom_norm":      round(_alpha_bottom_norm, 4),
                    "alpha_bottom_y_in_fit":  int(_obj_bottom_in_fit),
                    "placement_contact_y_sd": int(by2),
                    "applied_contact_shift":  int(_applied_contact_shift),
                    "contact_base_bbox_sd":   list(_contact_base_sd) if _contact_base_sd else None,
                    "contact_y_sd":           int(_contact_base_sd[3]) if _contact_base_sd else None,
                    "shadow_offset":          [0, 0],
                    "shadow_blur":            int(_sh_blur) if _contact_base_sd is not None else None,
                    "shadow_opacity":         float(_sh_str) if _contact_base_sd is not None else None,
                    "mask_type":              mask_type,
                }
                (_ddir / f"{_prefix}_contact_info.json").write_text(
                    json.dumps(_contact_info, indent=2, ensure_ascii=False), encoding="utf-8"
                )
            except Exception as _de:
                print(f"  [debug save error] {cat}: {_de}")

        print(f"  [ControlNet+IP] {cat} 생성 완료 (SD {sd_w}×{sd_h})")
        return output


_instance: ControlNetInpaintProcessor | None = None


def get_controlnet_inpaint_processor() -> ControlNetInpaintProcessor:
    global _instance
    if _instance is None:
        _instance = ControlNetInpaintProcessor()
    return _instance
