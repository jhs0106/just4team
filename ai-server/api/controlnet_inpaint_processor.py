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
)


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
    # SDXL backbone (2026-05-24 upgrade, 2026-05-25 perspective warp 도입)
    #   - 기존: SD1.5 + sd-controlnet-{depth,canny} + ip-adapter-plus_sd15
    #   - 현재: SDXL Inpaint + controlnet-depth-sdxl-1.0 + ip-adapter-plus_sdxl
    #           해상도 768 (12GB GPU 균형), depth ControlNet 단일.
    # Canny ControlNet 제거 이유:
    #   - 평면 product silhouette을 강제해 perspective 재해석 방해.
    #   - VRAM 2.5GB 절약 (12GB GPU에서 spillover 해소).
    # 평면 카테고리(KEYBOARD/MOUSE 등)는 _warp_product_to_desk_perspective로
    # 책상 각도에 맞춰 pre-warp 후 SD/IP-Adapter에 전달.
    # LoRA(JU_DeskStyle)는 SD1.5용으로 학습됨 → SDXL 호환 안 됨, 일단 skip.

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype  = torch.float16 if self.device == "cuda" else torch.float32
        print("[ControlNetInpaint/SDXL] 모델 로드 중...")
        self._load_depth_model()
        self._load_pipeline()
        print("[ControlNetInpaint/SDXL] 로드 완료.")

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
            StableDiffusionXLControlNetInpaintPipeline,
            DPMSolverMultistepScheduler,
        )
        from transformers import CLIPVisionModelWithProjection

        # ControlNet은 depth만 사용 (canny 제거).
        # - VRAM 2.5GB 절약 (12GB GPU에서 spillover 해소)
        # - Canny는 평면 product silhouette을 강제해 perspective 재해석 방해.
        #   pre-warp으로 어차피 perspective-aware한 외형 됨.
        cn_depth = ControlNetModel.from_pretrained(
            "diffusers/controlnet-depth-sdxl-1.0",
            torch_dtype=self.dtype,
            low_cpu_mem_usage=False,
        )

        # ip-adapter-plus_sdxl_vit-h.bin은 CLIP ViT-H(1280-dim) image encoder 사용.
        # 명시 안 하면 diffusers가 SDXL OpenCLIP ViT-bigG(1664-dim)을 로드해 dim mismatch.
        image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            "h94/IP-Adapter",
            subfolder="models/image_encoder",
            torch_dtype=self.dtype,
            low_cpu_mem_usage=False,
        )

        pipe = StableDiffusionXLControlNetInpaintPipeline.from_pretrained(
            "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
            controlnet=cn_depth,
            image_encoder=image_encoder,
            torch_dtype=self.dtype,
            low_cpu_mem_usage=False,
            variant="fp16",
        )
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            pipe.scheduler.config,
            algorithm_type="dpmsolver++",
            use_karras_sigmas=True,
        )
        pipe.vae.enable_slicing()
        # SDXL은 VAE tiling도 권장 (큰 해상도 OOM 방지)
        try:
            pipe.vae.enable_tiling()
        except Exception:
            pass
        # VRAM 부족 시 자동 CPU↔GPU offload (8~16GB GPU 환경 대응).
        # 순수 GPU 대비 20~40% 느리지만 OOM 방지가 우선.
        try:
            pipe.enable_model_cpu_offload()
        except Exception as _e:
            print(f"[ControlNetInpaint/SDXL] CPU offload 활성화 실패: {_e}")

        # IP-Adapter Plus SDXL 변형
        pipe.load_ip_adapter(
            "h94/IP-Adapter",
            subfolder="sdxl_models",
            weight_name="ip-adapter-plus_sdxl_vit-h.bin",
        )
        pipe.set_ip_adapter_scale(0.7)

        # LoRA: SD1.5용으로 학습됨 → SDXL 호환 안 됨. 일단 skip.
        # 향후 SDXL용 재학습 필요 (별도 TODO)
        self._has_lora = False

        self.pipe = pipe

    def _sd_size(self, w: int, h: int) -> tuple[int, int]:
        # SDXL 호환 크기: 긴 변=768, 짧은 변≥512, 8의 배수.
        # SDXL native는 1024지만 GPU 부담 큼(2x ControlNet+IP-Adapter+inpaint).
        # 768은 SD1.5(512)보다 2.25x 픽셀이면서 속도/품질 균형점.
        base = 768
        if w >= h:
            sw = base
            sh = max(512, round(h * base / w / 8) * 8)
        else:
            sh = base
            sw = max(512, round(w * base / h / 8) * 8)
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

        # 3-tier 카테고리 분류에 따른 처리:
        #   flat (KEYBOARD/MOUSEPAD): perspective warp 적용
        #   semi_flat (MOUSE/LAPTOP_STAND): warp 미적용, upright prompt 미적용
        #   upright (MONITOR/SPEAKER/DESK_LAMP 등): warp 절대 금지, upright prompt 추가
        _form_tier   = _PRODUCT_FORM_TIER.get(cat, "semi_flat")
        _is_flat     = (_form_tier == "flat")
        _is_upright  = (_form_tier == "upright")
        _warp_applied = False
        if _is_flat:
            prod_rgba = _warp_product_to_desk_perspective(prod_rgba, _DEFAULT_DESK_TILT_DEG)
            _warp_applied = True
            print(f"  [perspective_warp] {cat} depression={_DEFAULT_DESK_TILT_DEG}° → {prod_rgba.size}")
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

        # B. CV composite 선행: img_sd에 제품 합성 → canny가 제품 엣지를 인식
        composite_sd = img_sd.copy().convert("RGBA")
        if _pc2x > _pc1x and _pc2y > _pc1y:
            _slice = _prod_fit.crop((_src1x, _src1y, _src2x, _src2y))
            composite_sd.alpha_composite(_slice, (_pc1x, _pc1y))
        composite_sd = composite_sd.convert("RGB")

        # color matching: composite_sd는 canny 생성용 — 적용 안 함 (canny edge 품질 간섭)
        _color_matched_composite_sd = composite_sd.copy()

        # C. silhouette mask: alpha 있으면 전 카테고리 적용, 없으면 bbox fallback
        _has_alpha = np.array(prod_rgba.getchannel("A")).min() < 250
        mask_type = "bbox"
        if _has_alpha and _pc2x > _pc1x and _pc2y > _pc1y:
            _sil = np.zeros((sd_h, sd_w), dtype=np.uint8)
            _alpha_slice = np.array(_prod_fit.getchannel("A"))[_src1y:_src2y, _src1x:_src2x]
            _sil[_pc1y:_pc2y, _pc1x:_pc2x] = _alpha_slice
            _k   = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
            _sil = cv2.dilate(_sil, _k)
            _sil = cv2.GaussianBlur(_sil, (7, 7), 0)
            mask_sd = Image.fromarray(_sil).convert("L")
            mask_type = "silhouette"
            print(f"  [generate_product] {cat} silhouette mask 적용")
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
        # enable_model_cpu_offload 활성 시 .to() 호출하면 충돌 — diffusers가 자동 관리.

        pipe_kwargs = dict(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=img_sd,
            mask_image=mask_sd,
            control_image=depth_sd,
            ip_adapter_image=prod_ip,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            controlnet_conditioning_scale=cn_scales,
            width=sd_w,
            height=sd_h,
        )
        if self._has_lora and _effective_lora_scale > 0:
            pipe_kwargs["cross_attention_kwargs"] = {"scale": _effective_lora_scale}

        # Pass 1: single-pass (pass2는 hallucination 유발 — 비활성화)
        result_sd_pass1 = self.pipe(**pipe_kwargs).images[0]
        result_sd = result_sd_pass1

        # === 3-zone blend 정책 ===
        # SD 결과(result_sd)와 CV pre-composite(composite_sd)를 silhouette 기반 zone별 가중치로 blend.
        # 목표: 제품 외형은 어느 정도 보존(SD가 완전 새로 그리지 않게) + 경계는 자연스럽게.
        #
        # zone        | composite 비중 | SD 비중 | 의도
        # ------------|---------------|--------|--------
        # inner       | 0.65          | 0.35   | 제품 내부 — DB 픽셀 65% 유지 (얼굴/로고 인식 가능)
        # edge ring   | 0.25          | 0.75   | 경계 — SD가 책상 톤과 융합하도록
        # background  | 0.00          | 1.00   | 배경 — SD가 책상 환경 자유 생성
        #
        # 카테고리별 차등 적용은 향후 실험 — 우선 단일 가중치로 통일.
        # 변경 시 docs/ARCHITECTURE.md §4.7 참조.
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
            _blend_w = np.clip(_inner_smooth * 0.65 + _edge_ring_smooth * 0.25, 0.0, 1.0)
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

        # refine mask: pass2 비활성화 상태에서도 debug/paste용으로 silhouette 확장만
        _refine_mask_arr = np.array(mask_sd)
        _dil_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        _refine_mask_arr = cv2.dilate(_refine_mask_arr, _dil_k)
        _refine_mask_arr = cv2.GaussianBlur(_refine_mask_arr, (5, 5), 0)
        _refine_mask = Image.fromarray(_refine_mask_arr).convert("L")

        # enable_model_cpu_offload가 자동으로 메모리 관리 — empty_cache만 명시.
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
                composite_sd.save(_ddir / f"{_prefix}_composite_sd.png")
                mask_sd.save(_ddir / f"{_prefix}_mask_sd.png")
                _refine_mask.save(_ddir / f"{_prefix}_refine_mask_sd.png")
                final_paste_mask.save(_ddir / f"{_prefix}_final_paste_mask.png")
                result_sd_pass1.save(_ddir / f"{_prefix}_pass1_result_sd.png")
                prod_ip.save(_ddir / f"{_prefix}_prod_ip.png")
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
                    "alpha_bottom_norm":      round(_alpha_bottom_norm, 4),
                    "alpha_bottom_y_in_fit":  int(_obj_bottom_in_fit),
                    "placement_contact_y_sd": int(by2),
                    "applied_contact_shift":  int(_applied_contact_shift),
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
