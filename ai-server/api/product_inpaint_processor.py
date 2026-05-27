import torch
import numpy as np
import cv2
from pathlib import Path
from PIL import Image
from diffusers import StableDiffusionImg2ImgPipeline, StableDiffusionInpaintPipeline, DPMSolverMultistepScheduler
from PIL import ImageOps
import yaml


def load_config():
    config_path = Path(__file__).parent.parent / "configs" / "config.yaml"
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


class ProductInpaintProcessor:
    # CV 합성 + SD img2img + LoRA 하이브리드 파이프라인
    def __init__(self):
        self.config = load_config()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32

        gen_cfg = self.config.get("generation", {})
        self.negative_prompt = gen_cfg.get(
            "negative_prompt",
            "blurry, low quality, distorted, watermark, text, person, face, floating, unrealistic",
        )

        print("[ProductInpaint] 모델 로드 중...")
        self._load_pipeline()
        print("[ProductInpaint] 로드 완료.")

    def _load_pipeline(self):
        # img2img: 스타일 전체 보정용 (필요 시 사용)
        img2img = StableDiffusionImg2ImgPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=self.dtype,
        )
        img2img.scheduler = DPMSolverMultistepScheduler.from_config(
            img2img.scheduler.config,
            algorithm_type="dpmsolver++",
            use_karras_sigmas=True,
        )
        img2img.safety_checker = None
        img2img.vae.enable_slicing()

        # inpaint: 제품 경계 자연화 전용
        inpaint = StableDiffusionInpaintPipeline.from_pretrained(
            "runwayml/stable-diffusion-inpainting",
            torch_dtype=self.dtype,
        )
        inpaint.scheduler = DPMSolverMultistepScheduler.from_config(
            inpaint.scheduler.config,
            algorithm_type="dpmsolver++",
            use_karras_sigmas=True,
        )
        inpaint.safety_checker = None
        inpaint.vae.enable_slicing()

        # IP-Adapter-Plus: inpaint 파이프에만 로드 (generate_product에서 사용)
        inpaint.load_ip_adapter(
            "h94/IP-Adapter",
            subfolder="models",
            weight_name="ip-adapter-plus_sd15.bin",
        )
        inpaint.set_ip_adapter_scale(0.7)

        lora_path = (
            Path(__file__).parent.parent
            / "outputs" / "models" / "style_lora"
            / "style_lora_final" / "unet_lora"
        )
        for p in (img2img, inpaint):
            if lora_path.exists():
                try:
                    p.load_lora_weights(str(lora_path), weight_name="adapter_model.safetensors")
                except Exception:
                    pass  # inpaint 모델 아키텍처 불일치 시 스킵

        if lora_path.exists():
            print("[ProductInpaint] style LoRA 로드 완료.")
        else:
            print("[ProductInpaint] style LoRA 없음, 스킵.")

        self.pipe         = img2img.to(self.device)
        self.inpaint_pipe = inpaint.to(self.device)

    def composite_products(self, image: Image.Image, products: list) -> Image.Image:
        # products: [{"image_path", "region": (x1,y1,x2,y2), "category"}]
        # 렌더링 순서: 후면(모니터/스피커 등) → 전면(키보드/마우스)
        output = image.copy().convert("RGBA")

        BACK_CATS  = {"MONITOR", "SPEAKER", "DESK_LAMP", "DESK_SHELF", "LAPTOP_STAND", "CLOCK", "DECO"}
        FRONT_CATS = {"KEYBOARD", "MOUSE"}

        def _paste_one(item):
            path = item.get("image_path", "")
            region = item.get("region")
            if not path or not Path(path).exists() or not region:
                return
            x1, y1, x2, y2 = region
            rw, rh = x2 - x1, y2 - y1
            if rw <= 0 or rh <= 0:
                return

            prod = _remove_white_bg(Image.open(path))
            prod_w, prod_h = prod.size
            scale = min(rw / prod_w, rh / prod_h)
            prod = prod.resize(
                (int(prod_w * scale), int(prod_h * scale)),
                Image.Resampling.LANCZOS,
            )
            pw, ph = prod.size
            px = x1 + (rw - pw) // 2
            py = y2 - ph  # 하단 정렬
            output.paste(prod, (px, py), mask=prod.split()[3])

        for item in products:
            if item.get("category", "").upper() in BACK_CATS:
                _paste_one(item)
        for item in products:
            cat = item.get("category", "").upper()
            if cat in FRONT_CATS or cat not in BACK_CATS:
                _paste_one(item)

        return output.convert("RGB")

    def composite_products_homography(self, image: Image.Image, products: list) -> Image.Image:
        # products: [{"image_path", "front_corners": np.ndarray(4×2 TL/TR/BR/BL), "category"}]
        # 각 제품을 front_corners 평행사변형에 맞게 원근 변환 후 알파 블렌딩
        output = np.array(image.convert("RGB"), dtype=np.uint8)
        h, w = output.shape[:2]

        BACK_CATS  = {"MONITOR", "SPEAKER", "DESK_LAMP", "DESK_SHELF", "LAPTOP_STAND", "CLOCK", "DECO"}
        FRONT_CATS = {"KEYBOARD", "MOUSE"}

        def _warp_and_paste(item):
            path = item.get("image_path", "")
            front_corners = item.get("front_corners")
            if not path or not Path(path).exists() or front_corners is None:
                return

            prod = _remove_white_bg(Image.open(path))
            prod_w, prod_h = prod.size
            prod_arr = np.array(prod)

            src = np.array([
                [0,          0         ],
                [prod_w - 1, 0         ],
                [prod_w - 1, prod_h - 1],
                [0,          prod_h - 1],
            ], dtype=np.float32)
            dst = np.array(front_corners, dtype=np.float32)

            M = cv2.getPerspectiveTransform(src, dst)
            warped_rgb   = cv2.warpPerspective(prod_arr[:, :, :3], M, (w, h),
                               flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            warped_alpha = cv2.warpPerspective(prod_arr[:, :, 3],  M, (w, h),
                               flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

            alpha = warped_alpha.astype(np.float32) / 255.0
            for c in range(3):
                output[:, :, c] = (warped_rgb[:, :, c] * alpha + output[:, :, c] * (1 - alpha)).astype(np.uint8)

        for item in products:
            if item.get("category", "").upper() in BACK_CATS:
                _warp_and_paste(item)
        for item in products:
            cat = item.get("category", "").upper()
            if cat in FRONT_CATS or cat not in BACK_CATS:
                _warp_and_paste(item)

        return Image.fromarray(output)

    def refine_per_region(
        self,
        image: Image.Image,
        regions: list,
        style: str,
        lora_scale: float = 0.75,
        strength: float = 0.25,
        num_inference_steps: int = 25,
        guidance_scale: float = 7.5,
        padding_ratio: float = 0.25,
    ) -> Image.Image:
        # regions: [{"region": (x1,y1,x2,y2), "category"}]
        # 영역별 512×512 → SD → 원래 크기 복원 (전체 압축 없이 뭉개짐 방지)
        _CAT_PROMPT = {
            "KEYBOARD":     "mechanical keyboard on desk, natural lighting, sharp",
            "MOUSE":        "wireless mouse on desk, natural lighting, sharp",
            "MOUSEPAD":     "mouse pad on desk, natural lighting",
            "MONITOR":      "monitor on desk, natural lighting, sharp screen",
            "SPEAKER":      "desktop speaker on desk, natural lighting",
            "DESK_LAMP":    "desk lamp on desk, warm lighting",
            "DESK_SHELF":   "monitor riser shelf on desk, natural lighting",
            "LAPTOP_STAND": "laptop stand on desk, natural lighting",
            "DECO":         "desk decoration, natural lighting",
            "CLOCK":        "desk clock, natural lighting",
        }

        iw, ih = image.size
        output = image.copy()

        for item in regions:
            x1, y1, x2, y2 = item["region"]
            cat = item.get("category", "").upper()

            rw, rh = x2 - x1, y2 - y1
            pad_x = int(rw * padding_ratio)
            pad_y = int(rh * padding_ratio)
            px1 = max(0,  x1 - pad_x)
            py1 = max(0,  y1 - pad_y)
            px2 = min(iw, x2 + pad_x)
            py2 = min(ih, y2 + pad_y)

            crop = output.crop((px1, py1, px2, py2))
            cw, ch = crop.size
            crop_512 = crop.resize((512, 512), Image.Resampling.LANCZOS).convert("RGB")

            cat_desc = _CAT_PROMPT.get(cat, "product on desk, natural lighting")
            prompt   = f"JU_Style, {style} style desk setup, {cat_desc}, photorealistic"

            result = self.pipe(
                prompt=prompt,
                negative_prompt=self.negative_prompt,
                image=crop_512,
                strength=strength,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                cross_attention_kwargs={"scale": lora_scale},
            ).images[0]

            output.paste(result.resize((cw, ch), Image.Resampling.LANCZOS), (px1, py1))
            print(f"  [Refine] {cat} 완료 ({px2-px1}×{py2-py1}px)")

        return output

    def refine_with_style(
        self,
        image: Image.Image,
        style: str,
        desk_bbox: tuple | None = None,
        lora_scale: float = 0.75,
        strength: float = 0.35,
        num_inference_steps: int = 30,
        guidance_scale: float = 8.0,
    ) -> Image.Image:
        # 책상 영역 전체를 SD img2img로 자연화 (/product_place 엔드포인트용)
        w, h = image.size
        if desk_bbox:
            dx1, dy1, dx2, dy2 = desk_bbox
            crop = image.crop((dx1, dy1, dx2, dy2))
            cw, ch = crop.size
        else:
            crop = image
            cw, ch = w, h

        prompt   = f"JU_Style, {style} style desk setup, photorealistic, natural lighting, clean desk"
        crop_512 = crop.resize((512, 512)).convert("RGB")

        result = self.pipe(
            prompt=prompt,
            negative_prompt=self.negative_prompt,
            image=crop_512,
            strength=strength,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            cross_attention_kwargs={"scale": lora_scale},
        ).images[0]

        result_resized = result.resize((cw, ch), Image.Resampling.LANCZOS)
        if desk_bbox:
            output = image.copy()
            output.paste(result_resized, (dx1, dy1))
            return output
        return result_resized


    def generate_product(
        self,
        image: Image.Image,
        mask: Image.Image,
        category: str,
        style: str,
        product_image: Image.Image | None = None,
        ip_adapter_scale: float = 0.7,
        num_inference_steps: int = 30,
        guidance_scale: float = 7.5,
        lora_scale: float = 0.65,
        context_pad: int = 80,
    ) -> Image.Image:
        # LaMa가 지운 영역(mask)에 SD Inpainting으로 제품 직접 생성
        # image: LaMa-cleaned PIL RGB, mask: PIL L (흰색=생성할 영역)
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

        iw, ih = image.size
        mask_arr = np.array(mask.convert("L"))
        ys, xs = np.where(mask_arr > 127)
        if len(xs) == 0:
            return image

        # 마스크 bbox + 주변 맥락 패딩
        x1, y1 = int(xs.min()), int(ys.min())
        x2, y2 = int(xs.max()), int(ys.max())
        px1 = max(0, x1 - context_pad);  py1 = max(0, y1 - context_pad)
        px2 = min(iw, x2 + context_pad); py2 = min(ih, y2 + context_pad)

        crop_img  = image.crop((px1, py1, px2, py2))
        crop_mask = mask.crop((px1, py1, px2, py2))
        cw, ch    = crop_img.size

        crop_512 = crop_img.resize((512, 512), Image.Resampling.LANCZOS).convert("RGB")
        mask_512 = crop_mask.resize((512, 512), Image.Resampling.NEAREST).convert("L")

        cat      = category.upper()
        cat_desc = _CAT_PROMPT.get(cat, "product on desk, natural lighting")
        prompt   = f"JU_Style, {style} style desk setup, {cat_desc}, photorealistic"

        self.inpaint_pipe.to(self.device)
        if product_image is not None:
            self.inpaint_pipe.set_ip_adapter_scale(ip_adapter_scale)
            prod_512 = product_image.resize((512, 512), Image.Resampling.LANCZOS).convert("RGB")
        else:
            self.inpaint_pipe.set_ip_adapter_scale(0.0)
            prod_512 = None

        pipe_kwargs = dict(
            prompt=prompt,
            negative_prompt=self.negative_prompt,
            image=crop_512,
            mask_image=mask_512,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            cross_attention_kwargs={"scale": lora_scale},
        )
        if prod_512 is not None:
            pipe_kwargs["ip_adapter_image"] = prod_512

        result = self.inpaint_pipe(**pipe_kwargs).images[0]
        self.inpaint_pipe.to("cpu")
        torch.cuda.empty_cache()

        output = image.copy()
        output.paste(result.resize((cw, ch), Image.Resampling.LANCZOS), (px1, py1))
        print(f"  [SD Inpaint] {cat} 생성 완료 ({cw}×{ch}px)")
        return output


def _remove_white_bg(img: Image.Image, threshold: int = 240) -> Image.Image:
    # EXIF 회전 적용 후 흰 배경 투명 처리
    img = ImageOps.exif_transpose(img)
    data = np.array(img.convert("RGBA"))
    r, g, b = data[:, :, 0], data[:, :, 1], data[:, :, 2]
    data[(r >= threshold) & (g >= threshold) & (b >= threshold), 3] = 0
    return Image.fromarray(data, "RGBA")


_instance: ProductInpaintProcessor | None = None


def get_product_inpaint_processor() -> ProductInpaintProcessor:
    global _instance
    if _instance is None:
        _instance = ProductInpaintProcessor()
    return _instance
