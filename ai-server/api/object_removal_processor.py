import torch
import base64
import numpy as np
import cv2
from io import BytesIO
from PIL import Image

from .sam2_processor import get_sam2_processor


def image_to_b64(image: Image.Image) -> str:
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def _dilate_mask(mask_pil: Image.Image, dilation_size: int) -> Image.Image:
    """
    MaxFilter(square) 대신 Ellipse kernel 팽창 사용.
    - 커버리지 크기(반경)는 동일하게 유지
    - 모서리가 자연스럽게 처리되어 LaMa 인페인팅 품질 향상
    - 팽창 후 400px 미만 노이즈 스펙 제거
    """
    arr = np.array(mask_pil)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilation_size, dilation_size))
    dilated = cv2.dilate(arr, kernel, iterations=1)

    # 팽창 과정에서 생긴 작은 노이즈 스펙 제거
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(dilated, connectivity=8)
    clean = np.zeros_like(dilated)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= 400:
            clean[labels == i] = 255

    return Image.fromarray(clean, mode="L")


class ObjectRemovalProcessor:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._mask_generator = None
        print("[ObjectRemoval] 초기화 완료.")

    def _get_mask_generator(self):
        """SAM2AutomaticMaskGenerator — SAM-2 모델 재사용 (중복 로드 없음)"""
        if self._mask_generator is None:
            from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
            sam_model = get_sam2_processor().predictor.model
            self._mask_generator = SAM2AutomaticMaskGenerator(
                model=sam_model,
                points_per_side=32,
                pred_iou_thresh=0.82,
                stability_score_thresh=0.90,
                min_mask_region_area=200,
            )
        return self._mask_generator

    def detect_and_mask(
        self,
        image: Image.Image,
        max_size: int = 512,
        min_area_ratio: float = 0.0005,
        max_area_ratio: float = 0.20,
        y_min_ratio: float = 0.25,
        dilation_size: int = 21,
    ) -> dict:
        """
        SAM-2 Automatic으로 물체 감지 후 마스크 생성.

        - 크기 필터: 너무 크거나(배경) 너무 작은(노이즈) 마스크 제외
        - Y축 필터: 이미지 상단 y_min_ratio 이내 중심 마스크 제외 (모니터/벽)
        - 정렬: 작은 물체부터 — 순차 제거 시 이미 정리된 책상 표면이 이후 맥락으로 사용됨

        Returns
        -------
        mask             : 합산 팽창 마스크 (b64) — 오버레이 표시용
        overlay          : 원본 + 마스크 오버레이 (b64)
        num_objects      : 감지 물체 수
        individual_masks : 개별 팽창 마스크 목록 (PIL L, 원본 크기, 작은 것부터)
        """
        orig_w, orig_h = image.size
        if orig_w > max_size or orig_h > max_size:
            scale = min(max_size / orig_w, max_size / orig_h)
            new_w, new_h = int(orig_w * scale), int(orig_h * scale)
            image_resized = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
        else:
            image_resized = image

        img_array = np.array(image_resized.convert("RGB"))
        h, w = img_array.shape[:2]
        total_area = w * h
        min_area = int(total_area * min_area_ratio)
        max_area = int(total_area * max_area_ratio)
        y_min_px = h * y_min_ratio

        generator = self._get_mask_generator()
        with torch.inference_mode():
            masks = generator.generate(img_array)

        print(f"[ObjectRemoval] SAM-2 감지: 전체 {len(masks)}개 마스크")

        def centroid_y(m):
            x, y, bw, bh = m["bbox"]
            return y + bh / 2

        object_masks = [
            m for m in masks
            if min_area <= m["area"] <= max_area
            and centroid_y(m) >= y_min_px
        ]
        # 작은 물체부터 정렬 — 순차 제거 시 주변 책상 표면이 맥락으로 노출됨
        object_masks.sort(key=lambda m: m["area"])

        print(f"[ObjectRemoval] 크기+위치 필터 후: {len(object_masks)}개 "
              f"(면적 {min_area_ratio*100:.2f}%~{max_area_ratio*100:.0f}%, "
              f"Y >= {y_min_ratio*100:.0f}%)")

        if not object_masks:
            empty = Image.new("L", (orig_w, orig_h), 0)
            return {
                "mask": image_to_b64(empty),
                "overlay": image_to_b64(image),
                "num_objects": 0,
                "individual_masks": [],
            }

        # 개별 마스크: 원본 크기 복원 + 팽창
        individual_masks = []
        combined = np.zeros((h, w), dtype=bool)
        for m in object_masks:
            combined |= m["segmentation"]
            mask_small = Image.fromarray((m["segmentation"] * 255).astype(np.uint8), mode="L")
            mask_orig = mask_small.resize((orig_w, orig_h), Image.Resampling.NEAREST)
            individual_masks.append(_dilate_mask(mask_orig, dilation_size))

        # 합산 마스크 — 오버레이 표시용
        combined_img = Image.fromarray((combined * 255).astype(np.uint8), mode="L")
        combined_img = combined_img.resize((orig_w, orig_h), Image.Resampling.NEAREST)
        combined_dilated = _dilate_mask(combined_img, dilation_size)

        overlay = _make_overlay(image, combined_dilated)
        return {
            "mask": image_to_b64(combined_dilated),
            "overlay": image_to_b64(overlay),
            "num_objects": len(object_masks),
            "individual_masks": individual_masks,
        }


    def detect_with_prompt(
        self,
        image: Image.Image,
        prompt: str,
        max_size: int = 512,
        box_threshold: float = 0.30,
        max_area_ratio: float = 0.20,
        dilation_size: int = 21,
    ) -> dict:
        """
        Grounding DINO → SAM-2 박스 기반 세그멘테이션 → LaMa 제거용 마스크 생성.

        SAM-2 Auto 대비 장점: 텍스트로 원하는 물체만 정확히 타겟팅.
        max_area_ratio 초과 마스크(마우스패드·책상면)는 자동 제외.
        prompt 예시: "keyboard. mouse. headset. desk lamp."
        """
        from .dino_processor import run_grounding_dino
        from .sam2_processor import get_sam2_processor

        orig_w, orig_h = image.size
        if orig_w > max_size or orig_h > max_size:
            scale = min(max_size / orig_w, max_size / orig_h)
            image_resized = image.resize(
                (int(orig_w * scale), int(orig_h * scale)), Image.Resampling.LANCZOS
            )
        else:
            image_resized = image

        img_array = np.array(image_resized.convert("RGB"))
        img_bgr = img_array[:, :, ::-1].copy()
        h, w = img_array.shape[:2]
        max_area_px = int(h * w * max_area_ratio)

        detections = run_grounding_dino(img_bgr, prompt, box_threshold=box_threshold)
        print(f"[ObjectRemoval] DINO 감지: {len(detections)}개 / prompt='{prompt[:50]}'")

        if not detections:
            empty = Image.new("L", (orig_w, orig_h), 0)
            return {
                "mask": image_to_b64(empty),
                "overlay": image_to_b64(image),
                "num_objects": 0,
                "individual_masks": [],
            }

        predictor = get_sam2_processor().predictor
        predictor.set_image(img_array)

        individual_masks = []
        combined = np.zeros((h, w), dtype=bool)
        skipped = 0

        with torch.inference_mode():
            for det in detections:
                x1, y1, x2, y2 = det.box_xyxy
                box_np = np.array([x1, y1, x2, y2], dtype=np.float32)
                masks, scores, _ = predictor.predict(
                    point_coords=None,
                    point_labels=None,
                    box=box_np,
                    multimask_output=True,
                )
                if masks is None or len(masks) == 0:
                    continue
                best_mask = masks[int(np.argmax(scores))].astype(bool)

                # 마우스패드·책상면처럼 너무 큰 마스크 제외
                if best_mask.sum() > max_area_px:
                    skipped += 1
                    continue

                combined |= best_mask

                mask_small = Image.fromarray((best_mask * 255).astype(np.uint8), mode="L")
                mask_orig = mask_small.resize((orig_w, orig_h), Image.Resampling.NEAREST)
                individual_masks.append(_dilate_mask(mask_orig, dilation_size))

        print(f"[ObjectRemoval] 면적 필터({max_area_ratio*100:.0f}%) 후: {len(individual_masks)}개 사용, {skipped}개 제외")

        if not individual_masks:
            empty = Image.new("L", (orig_w, orig_h), 0)
            return {
                "mask": image_to_b64(empty),
                "overlay": image_to_b64(image),
                "num_objects": 0,
                "individual_masks": [],
            }

        combined_img = Image.fromarray((combined * 255).astype(np.uint8), mode="L")
        combined_img = combined_img.resize((orig_w, orig_h), Image.Resampling.NEAREST)
        combined_dilated = _dilate_mask(combined_img, dilation_size)

        overlay = _make_overlay(image, combined_dilated)
        return {
            "mask": image_to_b64(combined_dilated),
            "overlay": image_to_b64(overlay),
            "num_objects": len(individual_masks),
            "individual_masks": individual_masks,
        }

    def detect_from_top_view(
        self,
        front_view: Image.Image,
        top_view: Image.Image,
        max_size: int = 512,
        min_area_ratio: float = 0.005,
        max_area_ratio: float = 0.35,
        dilation_size: int = 15,
    ) -> dict:
        """
        탑뷰에서 SAM-2로 물체 감지 → 정면 뷰 크기로 마스크 스케일링.

        탑뷰는 원근감 왜곡 없이 물체 경계가 명확하고 겹침이 적음.
        Y축 필터 불필요 (모니터/벽이 이미지 상단에 걸리는 문제 없음).
        마스크는 비율 리사이즈로 정면 뷰에 근사 적용.
        """
        tv_w, tv_h = top_view.size
        if tv_w > max_size or tv_h > max_size:
            scale = min(max_size / tv_w, max_size / tv_h)
            tv_resized = top_view.resize((int(tv_w * scale), int(tv_h * scale)), Image.Resampling.LANCZOS)
        else:
            tv_resized = top_view

        tv_array = np.array(tv_resized.convert("RGB"))
        h, w = tv_array.shape[:2]
        total_area = w * h
        min_area = int(total_area * min_area_ratio)
        max_area = int(total_area * max_area_ratio)

        generator = self._get_mask_generator()
        with torch.inference_mode():
            masks = generator.generate(tv_array)

        print(f"[ObjectRemoval] 탑뷰 SAM-2 감지: 전체 {len(masks)}개 마스크")

        object_masks = [
            m for m in masks
            if min_area <= m["area"] <= max_area
        ]
        object_masks.sort(key=lambda m: m["area"])

        print(f"[ObjectRemoval] 탑뷰 필터 후: {len(object_masks)}개 "
              f"(면적 {min_area_ratio*100:.1f}%~{max_area_ratio*100:.0f}%)")

        orig_fw, orig_fh = front_view.size

        if not object_masks:
            empty = Image.new("L", (orig_fw, orig_fh), 0)
            return {
                "mask": image_to_b64(empty),
                "overlay": image_to_b64(front_view),
                "num_objects": 0,
                "individual_masks": [],
            }

        # 탑뷰 마스크 → 정면 뷰 크기로 비율 스케일
        individual_masks = []
        combined = np.zeros((h, w), dtype=bool)
        for m in object_masks:
            combined |= m["segmentation"]
            mask_tv = Image.fromarray((m["segmentation"] * 255).astype(np.uint8), mode="L")
            mask_fv = mask_tv.resize((orig_fw, orig_fh), Image.Resampling.NEAREST)
            individual_masks.append(_dilate_mask(mask_fv, dilation_size))

        combined_tv = Image.fromarray((combined * 255).astype(np.uint8), mode="L")
        combined_fv = combined_tv.resize((orig_fw, orig_fh), Image.Resampling.NEAREST)
        combined_dilated = _dilate_mask(combined_fv, dilation_size)

        overlay = _make_overlay(front_view, combined_dilated)
        return {
            "mask": image_to_b64(combined_dilated),
            "overlay": image_to_b64(overlay),
            "num_objects": len(object_masks),
            "individual_masks": individual_masks,
        }


def _make_overlay(image: Image.Image, mask: Image.Image) -> Image.Image:
    overlay = image.copy().convert("RGBA")
    mask_arr = np.array(mask)
    colored = np.zeros((*mask_arr.shape, 4), dtype=np.uint8)
    colored[mask_arr > 127] = [255, 80, 80, 140]
    overlay = Image.alpha_composite(overlay, Image.fromarray(colored))
    return overlay.convert("RGB")


# 싱글톤
_removal_instance: ObjectRemovalProcessor | None = None

def get_object_removal_processor() -> ObjectRemovalProcessor:
    global _removal_instance
    if _removal_instance is None:
        _removal_instance = ObjectRemovalProcessor()
    return _removal_instance
