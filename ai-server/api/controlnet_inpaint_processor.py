import json
import torch
import numpy as np
import cv2
from pathlib import Path
from PIL import Image


def _letterbox_512(img: Image.Image) -> Image.Image:
    # RGBA: transparent 영역을 중립 회색으로 합성 후 비율 유지 패딩
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
        "computer monitor with completely black screen powered off, "
        "thin bezel, narrow silver stand, standing on wooden desk, "
        "front view, blank dark screen, no display content, natural lighting"
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
        "keyboard, laptop, shelf, wallpaper, sunset, landscape, "
        "image on screen, glowing screen, lit screen, bright display, neon"
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
        context_region: tuple | None = None,
        debug_dir: Path | None = None,
        debug_meta: dict | None = None,
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
        _prod_orig_w, _prod_orig_h = prod_rgba.width, prod_rgba.height

        # fit product into SD placement bbox (category max scale 적용, 1.0 cap 제거)
        _cat_max_scale = _CAT_MAX_SCALE.get(cat, 2.0)
        _sc  = min(bw / max(prod_rgba.width, 1), bh / max(prod_rgba.height, 1), _cat_max_scale)
        _fpw = max(8, int(prod_rgba.width * _sc))
        _fph = max(8, int(prod_rgba.height * _sc))

        # KEYBOARD: 실제 배치 이미지 ar이 bbox ar보다 작을 때 horizontal stretch
        # 키보드는 책상에서 가로로 길게 보여야 하므로 비율보다 bbox 채우기를 우선
        if cat == "KEYBOARD" and _fpw < bw * 0.80:
            _fpw = max(8, int(bw * 0.90))
            print(f"  [KEYBOARD stretch] width → {_fpw} (bbox_w={bw:.0f}, fill={_fpw/max(bw,1):.2f})")

        _prod_fit = prod_rgba.resize((_fpw, _fph), Image.Resampling.LANCZOS)

        _ppx  = bx1 + (bw - _fpw) // 2   # center x
        _ppy  = by2 - _fph                # bottom-align y
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

        # 4. ControlNet: depth=빈 배경(책상 구조), canny=composite(제품 엣지)
        depth_sd = self._get_depth(img_sd)
        canny_sd = self._get_canny(composite_sd)

        # 5. IP-Adapter: letterbox 비율 유지 512×512
        prod_ip = _letterbox_512(product_image)

        # 카테고리별 depth/canny 가중치
        # MONITOR: canny 약하게 → 배경 조명/질감에 섞이도록 (black screen 목적)
        # KEYBOARD: canny 강하게 → 키 형태 유지 (이제 ControlNet 미통과이지만 예비)
        if cat == "MONITOR":
            cn_scales = [0.08, 0.20]
        elif cat == "KEYBOARD":
            cn_scales = [0.10, 0.25]
        else:
            cn_scales = [0.15, 0.28]

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
            controlnet_conditioning_scale=cn_scales,
            width=sd_w,
            height=sd_h,
        )
        if self._has_lora:
            pipe_kwargs["cross_attention_kwargs"] = {"scale": lora_scale}

        result_sd = self.pipe(**pipe_kwargs).images[0]

        self.pipe.to("cpu")
        torch.cuda.empty_cache()

        # 6. 원본 크기 복원: inpaint에 쓴 mask_sd 기준 final paste (silhouette/bbox 공통)
        result_crop = result_sd.resize((cw, ch), Image.Resampling.LANCZOS)
        output = image.copy()
        final_paste_mask = mask_sd.resize((cw, ch), Image.Resampling.LANCZOS).filter(
            __import__("PIL.ImageFilter", fromlist=["GaussianBlur"]).GaussianBlur(radius=3)
        )
        output.paste(result_crop, (cx1, cy1), mask=final_paste_mask)

        # debug 파일 저장
        if debug_dir is not None:
            try:
                _ddir = Path(debug_dir)
                _ddir.mkdir(parents=True, exist_ok=True)
                img_sd.save(_ddir / f"{cat}_crop_img.png")
                composite_sd.save(_ddir / f"{cat}_composite_sd.png")
                mask_sd.save(_ddir / f"{cat}_mask_sd.png")
                final_paste_mask.save(_ddir / f"{cat}_final_paste_mask.png")
                prod_ip.save(_ddir / f"{cat}_prod_ip.png")
                _m = debug_meta or {}
                # actual_composite: 원본 이미지 좌표계로 변환 (target_bbox와 직접 비교 가능)
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
                    "final_paste_mask_type":        mask_type,
                }
                (_ddir / f"{cat}_debug.json").write_text(
                    json.dumps(_dbg_json, indent=2, ensure_ascii=False), encoding="utf-8"
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
