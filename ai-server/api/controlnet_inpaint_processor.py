import torch
import numpy as np
import cv2
from pathlib import Path
from PIL import Image

_CAT_PROMPT = {
    "KEYBOARD":     "mechanical keyboard on desk mat, natural lighting, sharp",
    "MOUSE":        "wireless mouse on desk, natural lighting, sharp",
    "MOUSEPAD":     "mouse pad on desk surface, natural lighting",
    "MONITOR":      "monitor on desk, natural lighting, sharp screen",
    "SPEAKER":      "desktop speaker on desk, natural lighting",
    "DESK_LAMP":    "desk lamp on desk, warm lighting",
    "DESK_SHELF":   "monitor riser shelf on desk, natural lighting",
    "LAPTOP_STAND": "laptop stand on desk, natural lighting",
    "DECO":         "desk decoration, natural lighting",
    "CLOCK":        "desk clock, natural lighting",
}

_NEGATIVE_PROMPT = (
    "blurry, low quality, distorted, watermark, text, person, face, "
    "floating, deformed, ugly, empty desk, bare desk, no product, "
    "missing object, invisible, transparent, same as background"
)


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
        ip_adapter_scale: float = 0.8,
        controlnet_scale: float = 0.6,
        lora_scale: float = 0.65,
        context_region: tuple | None = None,
    ) -> Image.Image:
        # context_region=(x1,y1,x2,y2): 지정 시 해당 영역을 비율 유지한 채 SD 입력
        # None이면 전체 이미지 사용 (후면 큰 제품용)
        iw, ih = image.size

        if context_region:
            cx1, cy1, cx2, cy2 = context_region
            ctx_img  = image.crop((cx1, cy1, cx2, cy2))
            ctx_mask = mask.crop((cx1, cy1, cx2, cy2))
        else:
            cx1, cy1 = 0, 0
            ctx_img  = image
            ctx_mask = mask

        cw, ch = ctx_img.size
        sd_w, sd_h = self._sd_size(cw, ch)  # 비율 유지 SD 크기

        img_sd   = ctx_img.resize((sd_w, sd_h), Image.Resampling.LANCZOS).convert("RGB")
        mask_sd  = ctx_mask.resize((sd_w, sd_h), Image.Resampling.NEAREST).convert("L")

        import numpy as _np
        _mask_arr = _np.array(mask_sd)
        _white_px = int((_mask_arr > 127).sum())
        print(f"  [generate_product] {category} mask white_px={_white_px} sd={sd_w}x{sd_h}")
        depth_sd = self._get_depth(img_sd)
        canny_sd = self._get_canny(img_sd)
        prod_sd  = product_image.resize((512, 512), Image.Resampling.LANCZOS).convert("RGB")

        cat      = category.upper()
        cat_desc = _CAT_PROMPT.get(cat, "product on desk, natural lighting")
        lora_token = "JU_Style, " if self._has_lora else ""
        prompt   = (
            f"{lora_token}{cat_desc}, {style} color scheme, "
            "placed on desk surface, drop shadow, realistic product photo, "
            "sharp focus, high detail, photorealistic, 8k"
        )

        self.pipe.set_ip_adapter_scale(ip_adapter_scale)
        self.pipe.to(self.device)

        pipe_kwargs = dict(
            prompt=prompt,
            negative_prompt=_NEGATIVE_PROMPT,
            image=img_sd,
            mask_image=mask_sd,
            control_image=[depth_sd, canny_sd],
            ip_adapter_image=prod_sd,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            controlnet_conditioning_scale=[controlnet_scale, controlnet_scale * 0.7],
            width=sd_w,
            height=sd_h,
        )
        if self._has_lora:
            pipe_kwargs["cross_attention_kwargs"] = {"scale": lora_scale}

        result_sd = self.pipe(**pipe_kwargs).images[0]

        self.pipe.to("cpu")
        torch.cuda.empty_cache()

        result_ctx = result_sd.resize((cw, ch), Image.Resampling.LANCZOS)
        output = image.copy()
        output.paste(result_ctx, (cx1, cy1), mask=ctx_mask.convert("L"))
        print(f"  [ControlNet+IP] {cat} 생성 완료 (SD {sd_w}×{sd_h})")
        return output


_instance: ControlNetInpaintProcessor | None = None


def get_controlnet_inpaint_processor() -> ControlNetInpaintProcessor:
    global _instance
    if _instance is None:
        _instance = ControlNetInpaintProcessor()
    return _instance
