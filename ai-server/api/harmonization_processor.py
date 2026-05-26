# ╔══════════════════════════════════════════════════════════════════╗
# ║  [DEPRECATED] 이 모듈은 더 이상 사용되지 않습니다.                 ║
# ║                                                                  ║
# ║  2026-05-23~24 실험: CV composite + 단일 SD harmonization으로     ║
# ║  Visual-RAG faithfulness 보장하려 했으나, 환경 영역(seam/그림자)   ║
# ║  에서 SD가 hallucinated 객체(유령 받침대, 가짜 글로우 등)를       ║
# ║  생성하는 문제로 폐기.                                            ║
# ║                                                                  ║
# ║  현재는 controlnet_inpaint_processor.py의 per-product SD          ║
# ║  generation 방식을 사용. main.py는 이 파일을 import하지 않음.     ║
# ║                                                                  ║
# ║  파일은 실험 흔적으로 보존. 임의로 호출하지 말 것.                 ║
# ║  자세한 내용은 docs/ARCHITECTURE.md §4.3 참조.                    ║
# ╚══════════════════════════════════════════════════════════════════╝

import json
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image


# 한 번의 텍스트 프롬프트에 들어갈 카테고리 자연어 단어
_CAT_KEYWORDS = {
    "MONITOR":      "monitor",
    "KEYBOARD":     "keyboard",
    "MOUSE":        "wireless mouse",
    "MOUSEPAD":     "desk mat",
    "SPEAKER":      "desktop speaker",
    "DESK_LAMP":    "desk lamp",
    "DESK_SHELF":   "monitor riser shelf",
    "LAPTOP_STAND": "laptop stand",
    "DECO":         "small desk decoration",
    "CLOCK":        "desk clock",
    "LIGHTING":     "horizontal screen light bar on top of monitor",
}

_NEGATIVE_PROMPT = (
    "blurry, low quality, distorted, watermark, text, person, face, "
    "deformed, ugly, missing object, extra objects, duplicate objects, "
    "floating, levitating, melted, smeared, "
    "colorful screen, bright screen, screen content, display image, glowing screen, "
    "wrong perspective, tilted, rotated, replaced object"
)


# style별 mood/lighting 키워드 — SD가 분위기를 강하게 잡도록 명시적 차등화
_STYLE_PROMPT_MODIFIER: dict[str, str] = {
    "white":      ("minimalist clean white desk setup, bright soft daylight, "
                   "scandinavian aesthetic, light wood tones, airy and spacious"),
    "black":      ("dark moody black desk setup, dim warm ambient lighting, "
                   "sleek matte black surfaces, dramatic shadows, sophisticated atmosphere"),
    "gaming":     ("RGB illuminated gaming desk setup, vibrant neon accent lights "
                   "(purple, cyan, red glow), dramatic colored ambient lighting, "
                   "dark surroundings with bright LED highlights, esports vibe"),
    "cozy":       ("cozy warm desk setup, soft amber lighting, natural wood textures, "
                   "comfortable and intimate atmosphere, hygge style"),
    "nordic":     ("nordic minimalist desk setup, neutral tones, soft natural light, "
                   "clean lines, scandinavian simplicity, light beige and white"),
    "modern":     ("modern contemporary desk setup, balanced neutral lighting, "
                   "sleek surfaces, refined and professional aesthetic"),
    "retro":      ("retro vintage desk setup, warm tungsten lighting, "
                   "70s 80s aesthetic, earthy tones, nostalgic mood"),
    "industrial": ("industrial loft desk setup, exposed metal and concrete, "
                   "moody warehouse lighting, raw textures, urban aesthetic"),
    "general":    ("clean desk setup, balanced natural lighting"),
}


class HarmonizationProcessor:
    # Visual-RAG의 [G] Generation 컴포넌트.
    # Augmented context(composite_full)를 받아 retrieved fact 주변(seam/그림자/조명)만
    # 조건부 생성. 제품 픽셀은 strength_map의 0.0 영역으로 100% 보존 (faithfulness guarantee).
    # SD 1회만 호출 → 제품 간 조명 일관성 확보.

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype  = torch.float16 if self.device == "cuda" else torch.float32
        print("[Harmonization] 모델 로드 중...")
        self._load_depth_model()
        self._load_pipeline()
        print("[Harmonization] 로드 완료.")

    def _load_depth_model(self):
        from transformers import DPTForDepthEstimation, DPTImageProcessor
        self._depth_proc  = DPTImageProcessor.from_pretrained("Intel/dpt-large")
        self._depth_model = DPTForDepthEstimation.from_pretrained(
            "Intel/dpt-large", low_cpu_mem_usage=False,
        ).to(self.device)
        self._depth_model.eval()

    def _load_pipeline(self):
        from diffusers import (
            ControlNetModel,
            DPMSolverMultistepScheduler,
            StableDiffusionControlNetInpaintPipeline,
        )

        cn_depth = ControlNetModel.from_pretrained(
            "lllyasviel/sd-controlnet-depth",
            torch_dtype=self.dtype, low_cpu_mem_usage=False,
        )
        cn_canny = ControlNetModel.from_pretrained(
            "lllyasviel/sd-controlnet-canny",
            torch_dtype=self.dtype, low_cpu_mem_usage=False,
        )
        pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            controlnet=[cn_depth, cn_canny],
            torch_dtype=self.dtype, low_cpu_mem_usage=False,
        )
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            pipe.scheduler.config,
            algorithm_type="dpmsolver++",
            use_karras_sigmas=True,
        )
        pipe.safety_checker = None
        pipe.vae.enable_slicing()

        # LoRA — 배경/조명에만 약하게 (제품 영역은 strength=0이라 어차피 미적용)
        lora_path = (
            Path(__file__).parent.parent / "outputs" / "models" / "lora_external"
        )
        self._has_lora = False
        if lora_path.exists():
            try:
                from peft import PeftModel
                pipe.unet = PeftModel.from_pretrained(pipe.unet, str(lora_path))
                self._has_lora = True
                print("[Harmonization] style LoRA 로드 완료 (PEFT).")
            except Exception as e:
                print(f"[Harmonization] LoRA 로드 실패 (스킵): {e}")
        else:
            print("[Harmonization] style LoRA 없음.")

        self.pipe = pipe

    def _sd_size(self, w: int, h: int, base: int = 512) -> tuple[int, int]:
        # 비율 유지, 긴 변=base, 8의 배수
        if w >= h:
            sw = base
            sh = max(256, round(h * base / w / 8) * 8)
        else:
            sh = base
            sw = max(256, round(w * base / h / 8) * 8)
        return sw, sh

    def _get_depth(self, image: Image.Image) -> Image.Image:
        inputs = self._depth_proc(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            depth = self._depth_model(**inputs).predicted_depth
        depth = torch.nn.functional.interpolate(
            depth.unsqueeze(1), size=image.size[::-1],
            mode="bicubic", align_corners=False,
        ).squeeze().cpu().numpy()
        depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8) * 255
        return Image.fromarray(depth.astype(np.uint8)).convert("RGB")

    def _get_canny(self, image: Image.Image, low: int = 80, high: int = 180) -> Image.Image:
        arr = cv2.Canny(np.array(image.convert("L")), low, high)
        return Image.fromarray(np.stack([arr] * 3, axis=-1))

    def _build_strength_map(
        self,
        size:            tuple[int, int],
        silhouettes:     list[Image.Image],
        seam_width:      int   = 28,
        shadow_height:   int   = 80,
        seam_strength:   float = 0.65,
        shadow_strength: float = 0.55,
    ) -> tuple[Image.Image, Image.Image]:
        # 반환: (strength_map_gray_0_255, binary_mask_for_pipe)
        # strength_map: per-pixel 강도. 제품 내부=0, seam ring=seam_strength, 아래 그림자 영역=gradient.
        # binary_mask: SD inpaint pipe에 들어갈 0/255 마스크 (strength>0 영역).
        w, h = size

        # 1. 모든 제품 silhouette → 단일 binary mask
        combined = np.zeros((h, w), dtype=np.uint8)
        for sil in silhouettes:
            sa = np.array(sil.resize((w, h), Image.NEAREST).convert("L"))
            combined = np.maximum(combined, (sa > 127).astype(np.uint8) * 255)

        # 2. seam ring (제품 경계선 띠): dilate - erode
        k_dilate = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (seam_width * 2 + 1, seam_width * 2 + 1)
        )
        k_erode  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        dilated  = cv2.dilate(combined, k_dilate)
        eroded   = cv2.erode(combined,  k_erode)
        seam_ring = cv2.subtract(dilated, eroded)
        seam_blur = cv2.GaussianBlur(seam_ring, (9, 9), 0)
        seam_f    = (seam_blur.astype(np.float32) / 255.0) * seam_strength

        # 3. 그림자 영역: 제품 아래로 shadow_height px gradient
        not_product = cv2.bitwise_not(combined)
        shadow_f    = np.zeros((h, w), dtype=np.float32)
        for shift in range(1, shadow_height + 1):
            shifted = np.zeros_like(combined)
            shifted[shift:, :] = combined[:-shift, :]
            below   = cv2.bitwise_and(shifted, not_product)
            strength = (below > 127).astype(np.float32) * \
                       (1.0 - (shift - 1) / shadow_height) * shadow_strength
            shadow_f = np.maximum(shadow_f, strength)
        shadow_f = cv2.GaussianBlur(shadow_f, (11, 11), 0)

        # 4. 합치기
        sm       = np.maximum(seam_f, shadow_f)
        sm_uint8 = np.clip(sm * 255, 0, 255).astype(np.uint8)

        # 5. binary mask: strength > ~2% 영역만 inpaint 대상
        binary = (sm_uint8 > 5).astype(np.uint8) * 255
        binary = cv2.morphologyEx(
            binary, cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        )

        return (
            Image.fromarray(sm_uint8).convert("L"),
            Image.fromarray(binary).convert("L"),
        )

    def harmonize(
        self,
        composite_full:      Image.Image,
        silhouettes:         list[Image.Image],
        categories:          list[str],
        style:               str,
        num_inference_steps: int   = 30,
        guidance_scale:      float = 7.5,
        seam_strength:       float = 0.65,
        shadow_strength:     float = 0.55,
        seam_width_px:       int   = 28,
        shadow_height_px:    int   = 80,
        cn_depth_scale:      float = 0.40,
        cn_canny_scale:      float = 0.35,
        lora_scale:          float = 0.30,
        debug_dir:           Path | None = None,
    ) -> Image.Image:
        # 입력:
        #   composite_full — 모든 제품이 CV로 합성된 정답 이미지 (원본 해상도)
        #   silhouettes    — 제품별 grayscale silhouette (원본 해상도)
        #   categories     — silhouettes와 같은 순서의 카테고리 이름
        # 출력: 책상 영역 + seam + 그림자만 SD가 손본 최종 이미지
        orig_w, orig_h = composite_full.size

        strength_full, binary_full = self._build_strength_map(
            (orig_w, orig_h), silhouettes,
            seam_width=seam_width_px, shadow_height=shadow_height_px,
            seam_strength=seam_strength, shadow_strength=shadow_strength,
        )

        sd_w, sd_h  = self._sd_size(orig_w, orig_h, base=768)
        comp_sd     = composite_full.resize((sd_w, sd_h), Image.LANCZOS).convert("RGB")
        binary_sd   = binary_full.resize((sd_w, sd_h), Image.NEAREST)
        strength_sd = strength_full.resize((sd_w, sd_h), Image.BILINEAR)

        depth_sd = self._get_depth(comp_sd)
        canny_sd = self._get_canny(comp_sd, low=80, high=180)

        unique_cats: list[str] = []
        for c in categories:
            if c not in unique_cats:
                unique_cats.append(c)
        product_terms = [_CAT_KEYWORDS.get(c, c.lower()) for c in unique_cats]
        product_str   = ", ".join(product_terms) if product_terms else "desk accessories"

        lora_token   = "JU_Style, " if (self._has_lora and lora_scale > 0) else ""
        style_modifier = _STYLE_PROMPT_MODIFIER.get(style, _STYLE_PROMPT_MODIFIER["general"])
        prompt = (
            f"{lora_token}{style_modifier}, "
            f"with {product_str}, "
            "objects firmly grounded on desk surface, "
            "realistic dark contact shadows directly beneath each object, "
            "soft ambient cast shadows, "
            "cohesive color temperature matching the style, "
            "no floating objects, "
            "photorealistic interior photography, sharp focus, professional"
        )
        # gaming style은 negative에서 일부 "colorful screen" 같은 제약을 풀어 RGB 표현 허용
        if style == "gaming":
            negative_prompt = (
                "blurry, low quality, distorted, watermark, text, person, face, "
                "deformed, ugly, missing object, extra objects, duplicate objects, "
                "floating, levitating, melted, smeared, "
                "wrong perspective, tilted, rotated, replaced object, "
                "bright white background, plain office lighting"
            )
        else:
            negative_prompt = _NEGATIVE_PROMPT

        self.pipe.to(self.device)
        pipe_kwargs = dict(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=comp_sd,
            mask_image=binary_sd,
            control_image=[depth_sd, canny_sd],
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            controlnet_conditioning_scale=[cn_depth_scale, cn_canny_scale],
            width=sd_w,
            height=sd_h,
        )
        if self._has_lora and lora_scale > 0:
            pipe_kwargs["cross_attention_kwargs"] = {"scale": lora_scale}

        result_sd = self.pipe(**pipe_kwargs).images[0]

        # per-pixel differential blend: composite × (1-s) + SD × s
        # 제품 내부(s=0) 완전 보존, seam/그림자(s>0)만 SD 결과 반영
        comp_arr = np.array(comp_sd).astype(np.float32)
        res_arr  = np.array(result_sd).astype(np.float32)
        s_arr    = (np.array(strength_sd).astype(np.float32) / 255.0)[:, :, np.newaxis]
        blended_sd = np.clip(comp_arr * (1.0 - s_arr) + res_arr * s_arr, 0, 255).astype(np.uint8)
        blended_sd_img = Image.fromarray(blended_sd)

        result_full = blended_sd_img.resize((orig_w, orig_h), Image.LANCZOS)

        self.pipe.to("cpu")
        torch.cuda.empty_cache()

        if debug_dir is not None:
            try:
                _dd = Path(debug_dir)
                _dd.mkdir(parents=True, exist_ok=True)
                composite_full.save(_dd / "harmonize_input_composite.png")
                strength_full.save(_dd / "harmonize_strength_map.png")
                binary_full.save(_dd / "harmonize_binary_mask.png")
                depth_sd.save(_dd / "harmonize_depth.png")
                canny_sd.save(_dd / "harmonize_canny.png")
                result_sd.save(_dd / "harmonize_sd_raw.png")
                result_full.save(_dd / "harmonize_final.png")
                _meta = {
                    "categories":          categories,
                    "style":               style,
                    "num_inference_steps": num_inference_steps,
                    "guidance_scale":      guidance_scale,
                    "seam_strength":       seam_strength,
                    "shadow_strength":     shadow_strength,
                    "seam_width_px":       seam_width_px,
                    "shadow_height_px":    shadow_height_px,
                    "cn_depth_scale":      cn_depth_scale,
                    "cn_canny_scale":      cn_canny_scale,
                    "lora_scale":          lora_scale if self._has_lora else None,
                    "has_lora":            self._has_lora,
                    "sd_size":             f"{sd_w}x{sd_h}",
                    "orig_size":           f"{orig_w}x{orig_h}",
                    "prompt":              prompt,
                    "negative_prompt":     negative_prompt,
                }
                (_dd / "harmonize_meta.json").write_text(
                    json.dumps(_meta, indent=2, ensure_ascii=False), encoding="utf-8"
                )
            except Exception as e:
                print(f"[Harmonization debug save error] {e}")

        return result_full


_instance: HarmonizationProcessor | None = None


def get_harmonization_processor() -> HarmonizationProcessor:
    global _instance
    if _instance is None:
        _instance = HarmonizationProcessor()
    return _instance
