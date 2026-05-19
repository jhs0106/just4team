import torch
import numpy as np
import cv2
from pathlib import Path
from PIL import Image
from diffusers import StableDiffusionImg2ImgPipeline, DPMSolverMultistepScheduler
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
        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=self.dtype,
        )
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            pipe.scheduler.config,
            algorithm_type="dpmsolver++",
            use_karras_sigmas=True,
        )
        pipe.safety_checker = None
        pipe.vae.enable_slicing()

        lora_path = (
            Path(__file__).parent.parent
            / "outputs" / "models" / "style_lora"
            / "style_lora_final" / "unet_lora"
        )
        if lora_path.exists():
            pipe.load_lora_weights(str(lora_path), weight_name="adapter_model.safetensors")
            print("[ProductInpaint] style LoRA 로드 완료.")
        else:
            print("[ProductInpaint] style LoRA 없음, 스킵.")

        self.pipe = pipe.to(self.device)

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


def _remove_white_bg(img: Image.Image, threshold: int = 240) -> Image.Image:
    # RGB 세 채널 모두 threshold 이상이면 투명 처리
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
