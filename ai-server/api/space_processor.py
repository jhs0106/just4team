from __future__ import annotations

import base64
import torch
import numpy as np
import cv2
from dataclasses import dataclass, asdict
from io import BytesIO
from PIL import Image


def image_to_b64(image: Image.Image) -> str:
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


@dataclass
class SpaceMetrics:
    desk_area_cm2: float
    occupied_area_cm2: float
    available_area_cm2: float
    occupancy_ratio: float
    available_ratio: float
    connected_available_regions_cm2: list[float]


def _compute_available(desk_mask: np.ndarray, occupied_mask: np.ndarray) -> np.ndarray:
    desk = (desk_mask > 0).astype(np.uint8) * 255
    occ = (occupied_mask > 0).astype(np.uint8) * 255
    return cv2.bitwise_and(desk, cv2.bitwise_not(occ))


def _analyze_space(
    desk_mask: np.ndarray,
    occupied_mask: np.ndarray,
    desk_width_cm: float,
    desk_depth_cm: float,
) -> tuple[SpaceMetrics, np.ndarray]:
    available = _compute_available(desk_mask, occupied_mask)

    h, w = desk_mask.shape[:2]
    area_factor = (desk_width_cm / float(w)) * (desk_depth_cm / float(h))

    desk_px = float(np.count_nonzero(desk_mask))
    occ_px = float(np.count_nonzero(occupied_mask & desk_mask))
    avail_px = float(np.count_nonzero(available))

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        (available > 0).astype(np.uint8), 8
    )
    regions = sorted(
        [stats[i, cv2.CC_STAT_AREA] * area_factor for i in range(1, num_labels)],
        reverse=True,
    )

    metrics = SpaceMetrics(
        desk_area_cm2=round(desk_px * area_factor, 2),
        occupied_area_cm2=round(occ_px * area_factor, 2),
        available_area_cm2=round(avail_px * area_factor, 2),
        occupancy_ratio=round(occ_px / desk_px, 3) if desk_px else 0.0,
        available_ratio=round(avail_px / desk_px, 3) if desk_px else 0.0,
        connected_available_regions_cm2=[round(r, 2) for r in regions],
    )
    return metrics, available


def _find_placement_center(available: np.ndarray) -> list[int] | None:
    """가용 공간에서 가장 넓은 연결 영역의 중심 좌표 반환."""
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        (available > 0).astype(np.uint8), 8
    )
    if num_labels <= 1:
        return None
    best_idx = int(np.argmax(stats[1:, cv2.CC_STAT_AREA])) + 1
    cx, cy = int(centroids[best_idx][0]), int(centroids[best_idx][1])
    return [cx, cy]


class SpaceProcessor:
    """
    탑뷰 이미지에서 SAM-2로 점유 영역을 감지하고 가용 공간을 분석.
    기존 ObjectRemovalProcessor의 mask_generator를 재사용해 중복 로드 없음.
    """

    def analyze(
        self,
        top_view: Image.Image,
        desk_width_cm: float,
        desk_depth_cm: float,
        max_size: int = 512,
        min_area_ratio: float = 0.003,
        max_area_ratio: float = 0.40,
    ) -> dict:
        img = top_view.convert("RGB")
        w, h = img.size
        if max(w, h) > max_size:
            scale = max_size / max(w, h)
            img = img.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)

        img_array = np.array(img)
        oh, ow = img_array.shape[:2]
        total_area = oh * ow
        min_area = int(total_area * min_area_ratio)
        max_area = int(total_area * max_area_ratio)

        from .object_removal_processor import get_object_removal_processor
        generator = get_object_removal_processor()._get_mask_generator()

        with torch.inference_mode():
            masks = generator.generate(img_array)

        object_masks = [m for m in masks if min_area <= m["area"] <= max_area]

        occupied = np.zeros((oh, ow), dtype=np.uint8)
        for m in object_masks:
            occupied = np.maximum(occupied, (m["segmentation"] * 255).astype(np.uint8))

        desk_mask = np.full((oh, ow), 255, dtype=np.uint8)
        metrics, available = _analyze_space(desk_mask, occupied, desk_width_cm, desk_depth_cm)
        placement_center = _find_placement_center(available)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return {
            "metrics": asdict(metrics),
            "available_mask": image_to_b64(Image.fromarray(available)),
            "occupied_mask": image_to_b64(Image.fromarray(occupied)),
            "placement_center": placement_center,
            "num_objects": len(object_masks),
        }


_space_instance: SpaceProcessor | None = None


def get_space_processor() -> SpaceProcessor:
    global _space_instance
    if _space_instance is None:
        _space_instance = SpaceProcessor()
    return _space_instance
