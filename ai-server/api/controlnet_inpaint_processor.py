import torch
import numpy as np
import cv2
from pathlib import Path
from PIL import Image

_CAT_PROMPT = {
    "KEYBOARD": (
        "low profile keyboard lying flat on desk, front perspective, "
        "rectangular keys visible, natural lighting"
    ),
    "MOUSE": (
        "small wireless mouse resting on desk surface, top-front view, "
        "soft contact shadow, natural lighting"
    ),
    "MOUSEPAD": (
        "flat desk mat mouse pad lying on desk surface, thin rectangular mat, "
        "natural lighting"
    ),
    "MONITOR": (
        "thin computer monitor with rectangular screen, visible narrow stand, "
        "standing upright on desk, front view, not a laptop, not a shelf, "
        "turned off dark screen, natural lighting"
    ),
    "SPEAKER": (
        "small desktop speaker standing on desk, front view, "
        "compact rectangular speaker, natural lighting"
    ),
    "DESK_LAMP": (
        "modern desk lamp with visible round base, vertical arm and lampshade, "
        "standing on desk, not a cable, natural lighting"
    ),
    "DESK_SHELF": (
        "monitor riser shelf on desk, horizontal wooden shelf, "
        "open storage space below, natural lighting"
    ),
    "LAPTOP_STAND": (
        "laptop stand on desk, angled metal stand, natural lighting"
    ),
    "DECO": (
        "small desk decoration object placed on desk surface, "
        "realistic scale, natural lighting"
    ),
    "CLOCK": (
        "small digital desk clock standing on desk, visible clock face, "
        "natural lighting"
    ),
}

_NEGATIVE_PROMPT = (
    "blurry, low quality, distorted, watermark, text, person, face, "
    "floating, deformed, ugly, empty desk, bare desk, no product, "
    "missing object, invisible, transparent, same as background, "
    "colorful screen, bright screen, screen content, display image, glowing screen"
)

_CAT_NEGATIVE = {
    "MONITOR": (
        "keyboard, laptop, shelf, bookshelf, floating screen, "
        "screen lying flat, distorted stand, extra keyboard"
    ),
    "DESK_LAMP": (
        "cable only, wire only, floating line, no base, broken lamp, "
        "thin random curve, snake, cord"
    ),
    "MOUSE": (
        "large object, keyboard, monitor, floating, deformed mouse"
    ),
    "KEYBOARD": (
        "monitor, laptop screen, vertical object, floating keys"
    ),
    "MOUSEPAD": (
        "thick object, monitor, keyboard, floating mat"
    ),
    "DESK_SHELF": (
        "monitor, laptop, items on shelf, picture frame, wall shelf"
    ),
    "SPEAKER": (
        "handle, picture frame, door, arch shape, bracket"
    ),
}


class ControlNetInpaintProcessor:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype  = torch.float16 if self.device == "cuda" else torch.float32
        print("[ControlNetInpaint] 모델 로드 중...")
        self._load_depth_model()
        self._load_pipeline()
        print("[ControlNetInpaint] 로드 완료.")

    def _load_depth_model(self):
        from transformers import DPTImageProcessor, DPTForDepthEstimation
        self._depth_proc  = DPTImageProcessor.from_pretrained("Intel/dpt-large")
        # low_cpu_mem_usage=False: thread pool 내부 충돌(Python 3.12) 방지
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

        cn_depth = ControlNetModel.from_pretrained(
            "lllyasviel/sd-controlnet-depth",
            torch_dtype=self.dtype,
            low_cpu_mem_usage=False,
        )
        cn_canny = ControlNetModel.from_pretrained(
            "lllyasviel/sd-controlnet-canny",
            torch_dtype=self.dtype,
            low_cpu_mem_usage=False,
        )

        pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            controlnet=[cn_depth, cn_canny],
            torch_dtype=self.dtype,
            low_cpu_mem_usage=False,
        )
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            pipe.scheduler.config,
            algorithm_type="dpmsolver++",
            use_karras_sigmas=True,
        )
        pipe.safety_checker = None
        pipe.vae.enable_slicing()

        # IP-Adapter-Plus: attention processor 설정 전에 먼저 로드해야 충돌 없음
        pipe.load_ip_adapter(
            "h94/IP-Adapter",
            subfolder="models",
            weight_name="ip-adapter-plus_sd15.bin",
        )
        pipe.set_ip_adapter_scale(0.7)

        # style LoRA
        lora_path = (
            Path(__file__).parent.parent
            / "outputs" / "models" / "lora_external"
        )
        self._has_lora = False
        if lora_path.exists():
            try:
                from peft import PeftModel
                pipe.unet = PeftModel.from_pretrained(pipe.unet, str(lora_path))
                self._has_lora = True
                print("[ControlNetInpaint] style LoRA 로드 완료 (PEFT).")
            except Exception as e:
                print(f"[ControlNetInpaint] LoRA 로드 실패 (스킵): {e}")
        else:
            print("[ControlNetInpaint] style LoRA 없음, 스킵.")

        self.pipe = pipe

    # 전처리

    def _sd_size(self, w: int, h: int) -> tuple[int, int]:
        """비율 유지 SD 호환 크기 (8의 배수). 긴 변 기준 512, 짧은 변 min 256."""
        if w >= h:
            sw = 512
            sh = max(256, round(h * 512 / w / 8) * 8)
        else:
            sh = 512
            sw = max(256, round(w * 512 / h / 8) * 8)
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

    # 메인

    def generate_product(
        self,
        image: Image.Image,
        mask: Image.Image,
        product_image: Image.Image,
        category: str,
        style: str,
        num_inference_steps: int = 40,
        guidance_scale: float = 9.0,
        ip_adapter_scale: float = 0.4,
        controlnet_scale: float = 0.6,
        lora_scale: float = 0.65,
        context_region: tuple | None = None,  # 하위 호환, 내부 미사용
    ) -> Image.Image:
        iw, ih = image.size

        # 1. 마스크에서 제품 bbox 추출
        mask_np = np.array(mask.convert("L"))
        ys, xs  = np.where(mask_np > 127)
        if len(xs) == 0:
            print(f"  [generate_product] {category} mask 비어있음 — skip")
            return image

        x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
        pw, ph = max(1, x2 - x1), max(1, y2 - y1)

        # 2. context crop: 제품 bbox 주변 50% 패딩 (너무 크면 SD 해상도에서 마스크 비율이 너무 작아짐)
        pad = min(int(max(pw, ph) * 0.5), 160)
        cx1 = max(0, x1 - pad)
        cy1 = max(0, y1 - pad)
        cx2 = min(iw, x2 + pad)
        cy2 = min(ih, y2 + pad)

        crop_img  = image.crop((cx1, cy1, cx2, cy2))
        crop_mask = mask.crop((cx1, cy1, cx2, cy2))
        cw, ch    = crop_img.size
        sd_w, sd_h = self._sd_size(cw, ch)

        img_sd  = crop_img.resize((sd_w, sd_h), Image.Resampling.LANCZOS).convert("RGB")
        mask_sd = crop_mask.resize((sd_w, sd_h), Image.Resampling.NEAREST).convert("L")

        _white_px = int((np.array(mask_sd) > 127).sum())
        print(f"  [generate_product] {category} white_px={_white_px} crop={cw}x{ch} sd={sd_w}x{sd_h}")

        # 3. ControlNet: 책상 crop 기준 depth/canny (배경 구조 유지)
        #    IP-Adapter가 제품 외형을 담당하므로 ControlNet은 책상 표면 구조만 전달
        depth_sd = self._get_depth(img_sd)
        canny_sd = self._get_canny(img_sd)

        # 4. IP-Adapter: 제품 이미지 외형 참조
        prod_ip = product_image.resize((512, 512), Image.Resampling.LANCZOS).convert("RGB")

        cat      = category.upper()
        cat_desc = _CAT_PROMPT.get(cat, "product on desk, natural lighting")
        lora_token = "JU_Style, " if self._has_lora else ""
        prompt   = (
            f"{lora_token}{cat_desc}, {style} color scheme, "
            "placed on desk surface, drop shadow, realistic product photo, "
            "sharp focus, high detail, photorealistic, 8k"
        )
        negative_prompt = _NEGATIVE_PROMPT + (
            ", " + _CAT_NEGATIVE[cat] if cat in _CAT_NEGATIVE else ""
        )

        self.pipe.set_ip_adapter_scale(ip_adapter_scale)
        self.pipe.to(self.device)

        pipe_kwargs = dict(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=img_sd,
            mask_image=mask_sd,
            control_image=[depth_sd, canny_sd],
            ip_adapter_image=prod_ip,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            controlnet_conditioning_scale=[0.35, 0.20],
            width=sd_w,
            height=sd_h,
        )
        if self._has_lora:
            pipe_kwargs["cross_attention_kwargs"] = {"scale": lora_scale}

        result_sd = self.pipe(**pipe_kwargs).images[0]

        self.pipe.to("cpu")
        torch.cuda.empty_cache()

        # 5. 원본 크기로 복원 후 soft mask 블렌딩 (사각형 경계 제거)
        result_crop = result_sd.resize((cw, ch), Image.Resampling.LANCZOS)
        output = image.copy()
        soft_mask = crop_mask.convert("L").filter(
            __import__("PIL.ImageFilter", fromlist=["GaussianBlur"]).GaussianBlur(radius=8)
        )
        output.paste(result_crop, (cx1, cy1), mask=soft_mask)
        print(f"  [ControlNet+IP] {cat} 생성 완료 (SD {sd_w}×{sd_h})")
        return output


_instance: ControlNetInpaintProcessor | None = None


def get_controlnet_inpaint_processor() -> ControlNetInpaintProcessor:
    global _instance
    if _instance is None:
        _instance = ControlNetInpaintProcessor()
    return _instance
