import json
import torch
import numpy as np
import cv2
from pathlib import Path
from PIL import Image


_SD3_MODEL_ID = "stabilityai/stable-diffusion-3.5-medium"
_SD3_BASE_RES = 768


_CAT_PROMPT = {
    "KEYBOARD":     "keyboard flat on desk, front view, natural lighting",
    "MOUSE":        "wireless mouse on desk, top-front view, natural lighting",
    "MOUSEPAD":     "desk mat on desk surface, thin flat rectangle, natural lighting",
    "MONITOR":      "monitor black screen, thin bezel, front view, natural lighting",
    "SPEAKER":      "desktop speaker on desk, front view, natural lighting",
    "DESK_LAMP":    "desk lamp with base and arm, standing on desk, natural lighting",
    "DESK_SHELF":   "monitor riser shelf on desk, open storage below, natural lighting",
    "LAPTOP_STAND": "laptop stand on desk, angled metal, natural lighting",
    "DECO":         "desk decoration on desk surface, natural lighting",
    "CLOCK":        "digital desk clock, visible face, natural lighting",
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


class SD35InpaintProcessor:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype  = torch.float16 if self.device == "cuda" else torch.float32
        # LoRA/IP-Adapter 미사용. main.py 호환을 위해 더미 속성 유지
        self._has_lora = False
        print(f"[SD3.5 Inpaint] 모델 로드 중... ({_SD3_MODEL_ID})")
        self._load_pipeline()
        print("[SD3.5 Inpaint] 로드 완료.")

    def _load_pipeline(self):
        from diffusers import StableDiffusion3InpaintPipeline

        pipe = StableDiffusion3InpaintPipeline.from_pretrained(
            _SD3_MODEL_ID,
            torch_dtype=self.dtype,
        )
        # 12GB VRAM 안전마진 — model_cpu_offload (sequential보다 빠름, 풀로드보다 안전)
        if self.device == "cuda":
            pipe.enable_model_cpu_offload()
        else:
            pipe.to(self.device)
        # VAE slicing — 큰 해상도 메모리 추가 절감
        try:
            pipe.vae.enable_slicing()
        except Exception:
            pass

        self.pipe = pipe

    def _sd_size(self, w: int, h: int) -> tuple[int, int]:
        # 비율 유지 SD3 호환 크기: 긴 변=_SD3_BASE_RES, 16의 배수
        base = _SD3_BASE_RES
        if w >= h:
            sw = base
            sh = max(256, round(h * base / w / 16) * 16)
        else:
            sh = base
            sw = max(256, round(w * base / h / 16) * 16)
        return sw, sh

    def generate_product(
        self,
        image: Image.Image,
        mask: Image.Image,
        product_image: Image.Image,
        category: str,
        style: str,
        num_inference_steps: int = 28,
        guidance_scale: float = 7.0,
        strength: float = 0.85,
        ip_adapter_scale: float = 0.0,
        controlnet_scale: float = 0.0,
        lora_scale: float = 0.0,
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
        _prod_orig_w, _prod_orig_h = prod_rgba.width, prod_rgba.height

        _cat_max_scale = _CAT_MAX_SCALE.get(cat, 2.0)
        _sc  = min(bw / max(prod_rgba.width, 1), bh / max(prod_rgba.height, 1), _cat_max_scale)
        _fpw = max(8, int(prod_rgba.width * _sc))
        _fph = max(8, int(prod_rgba.height * _sc))

        # KEYBOARD horizontal stretch (MOUSE는 비율 유지)
        if cat == "KEYBOARD" and _fpw < bw * 0.80:
            _fpw = max(8, int(bw * 0.90))
            print(f"  [KEYBOARD stretch] width → {_fpw} (bbox_w={bw:.0f}, fill={_fpw/max(bw,1):.2f})")

        # MONITOR 최소 fill 0.82
        _monitor_min_fill_applied = False
        if cat == "MONITOR" and _fpw < bw * 0.82:
            _mon_w = max(8, int(bw * 0.88))
            _mon_h = int(prod_rgba.height * _mon_w / max(prod_rgba.width, 1))
            if _mon_h <= bh:
                _fpw, _fph = _mon_w, _mon_h
                _monitor_min_fill_applied = True
            else:
                _mon_w2 = int(bh * prod_rgba.width / max(prod_rgba.height, 1))
                if _mon_w2 > _fpw:
                    _fpw, _fph = _mon_w2, bh
                    _monitor_min_fill_applied = True
            if _monitor_min_fill_applied:
                print(f"  [MONITOR min fill] width → {_fpw} (fill={_fpw/max(bw,1):.2f})")

        _prod_fit = prod_rgba.resize((_fpw, _fph), Image.Resampling.LANCZOS)

        _ppx = bx1 + (bw - _fpw) // 2
        # contact alignment: alpha object bottom → bbox bottom
        _alpha_arr_raw = np.array(prod_rgba.getchannel("A"))
        _rows_with_alpha = np.where((_alpha_arr_raw > 127).any(axis=1))[0]
        _alpha_bottom_norm = (_rows_with_alpha[-1] / max(prod_rgba.height - 1, 1)) if len(_rows_with_alpha) > 0 else 1.0
        _obj_bottom_in_fit = max(1, int(_fph * _alpha_bottom_norm))
        _ppy = by2 - _obj_bottom_in_fit
        _applied_contact_shift = _fph - _obj_bottom_in_fit
        print(f"  [contact_align] {cat} alpha_bottom={_alpha_bottom_norm:.3f} obj_bottom_fit={_obj_bottom_in_fit} shift={_applied_contact_shift}")

        _pc1x = max(0, _ppx);             _pc2x = min(sd_w, _ppx + _fpw)
        _pc1y = max(0, _ppy);             _pc2y = min(sd_h, _ppy + _fph)
        _src1x = max(0, -_ppx);           _src1y = max(0, -_ppy)
        _src2x = _src1x + (_pc2x - _pc1x); _src2y = _src1y + (_pc2y - _pc1y)

        print(f"  [generate_product] {cat} tight={_prod_orig_w}x{_prod_orig_h} "
              f"scale={_sc:.3f}(max={_cat_max_scale}) fit={_fpw}x{_fph} bbox_sd={bw}x{bh}")

        # B. CV composite 선행: img_sd에 제품 합성 → SD3 inpainting의 conditioning 역할
        composite_sd = img_sd.copy().convert("RGBA")
        if _pc2x > _pc1x and _pc2y > _pc1y:
            _slice = _prod_fit.crop((_src1x, _src1y, _src2x, _src2y))
            composite_sd.alpha_composite(_slice, (_pc1x, _pc1y))
        composite_sd = composite_sd.convert("RGB")

        # C. silhouette mask: alpha 있으면 전 카테고리 적용
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

        # 4. 프롬프트 — SD3는 더 자연어 친화적
        cat_desc = _CAT_PROMPT.get(cat, "product on desk, natural lighting")
        prompt = (
            f"A photorealistic {cat_desc}, {style} style desk setup, "
            f"sitting naturally on the wooden desk surface with soft drop shadow, "
            f"sharp focus, high detail, professional product photography"
        )
        negative_prompt = _NEGATIVE_PROMPT + (
            ", " + _CAT_NEGATIVE[cat] if cat in _CAT_NEGATIVE else ""
        )

        # 5. SD3.5 inpainting: composite_sd를 base image로 사용 → 강력한 conditioning
        pipe_kwargs = dict(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=composite_sd,
            mask_image=mask_sd,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            strength=strength,
            width=sd_w,
            height=sd_h,
        )

        result_sd_pass1 = self.pipe(**pipe_kwargs).images[0]
        result_sd = result_sd_pass1

        # 3-zone blend: inner(65% composite) / edge ring(25%) / background(0%)
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
            # dark rim occlusion: KEYBOARD/MOUSE — contact 하단
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

        torch.cuda.empty_cache()

        # 6. 원본 크기 복원: silhouette mask 기준 paste
        result_crop = result_sd.resize((cw, ch), Image.Resampling.LANCZOS)
        output      = image.copy()
        final_paste_mask = mask_sd.resize((cw, ch), Image.Resampling.LANCZOS).filter(
            __import__("PIL.ImageFilter", fromlist=["GaussianBlur"]).GaussianBlur(radius=3)
        )
        output.paste(result_crop, (cx1, cy1), mask=final_paste_mask)

        # debug 저장
        if debug_dir is not None:
            try:
                _ddir = Path(debug_dir)
                _ddir.mkdir(parents=True, exist_ok=True)
                _prefix = variant_prefix or cat
                img_sd.save(_ddir / f"{_prefix}_crop_img.png")
                composite_sd.save(_ddir / f"{_prefix}_composite_sd.png")
                mask_sd.save(_ddir / f"{_prefix}_mask_sd.png")
                final_paste_mask.save(_ddir / f"{_prefix}_final_paste_mask.png")
                result_sd_pass1.save(_ddir / f"{_prefix}_pass1_result_sd.png")
                if _blend_inner_arr is not None:
                    Image.fromarray(_blend_inner_arr).save(_ddir / f"{_prefix}_inner_mask.png")
                    Image.fromarray(_blend_edge_arr).save(_ddir / f"{_prefix}_edge_ring_mask.png")
                    Image.fromarray(_blend_result_arr).save(_ddir / f"{_prefix}_blend_result.png")
                _m = debug_meta or {}
                _actual_img_w = int(_fpw * cw / max(sd_w, 1))
                _actual_img_h = int(_fph * ch / max(sd_h, 1))
                _dbg_json = {
                    "category":                     cat,
                    "model":                        _SD3_MODEL_ID,
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
                    "monitor_min_fill_applied":     _monitor_min_fill_applied,
                    "num_inference_steps":          num_inference_steps,
                    "guidance_scale":               guidance_scale,
                    "strength":                     strength,
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

        print(f"  [SD3.5 Inpaint] {cat} 생성 완료 (SD {sd_w}×{sd_h})")
        return output


_instance: SD35InpaintProcessor | None = None


def get_sd35_inpaint_processor() -> SD35InpaintProcessor:
    global _instance
    if _instance is None:
        _instance = SD35InpaintProcessor()
    return _instance
