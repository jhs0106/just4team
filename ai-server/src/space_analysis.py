from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

import cv2
import numpy as np


@dataclass
class SpaceMetrics:
    desk_area_cm2: float
    occupied_area_cm2: float
    available_area_cm2: float
    occupancy_ratio: float
    available_ratio: float
    connected_available_regions_cm2: list[float]


def compute_available_space(desk_mask: np.ndarray, occupied_mask: np.ndarray) -> np.ndarray:
    desk = (desk_mask > 0).astype(np.uint8) * 255
    occ = (occupied_mask > 0).astype(np.uint8) * 255
    inv_occ = cv2.bitwise_not(occ)
    return cv2.bitwise_and(desk, inv_occ)


def analyze_space(
        desk_mask: np.ndarray,
        occupied_mask: np.ndarray,
        desk_width_cm: float,
        desk_depth_cm: float,
) -> tuple[SpaceMetrics, np.ndarray]:
    available = compute_available_space(desk_mask, occupied_mask)

    h, w = desk_mask.shape[:2]
    cm_per_px_x = desk_width_cm / float(w)
    cm_per_px_y = desk_depth_cm / float(h)
    area_factor = cm_per_px_x * cm_per_px_y

    desk_px = float(np.count_nonzero(desk_mask))
    occ_px = float(np.count_nonzero(occupied_mask & desk_mask))
    avail_px = float(np.count_nonzero(available))

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats((available > 0).astype(np.uint8), 8)
    regions = []
    for i in range(1, num_labels):
        area_px = stats[i, cv2.CC_STAT_AREA]
        regions.append(area_px * area_factor)
    regions.sort(reverse=True)

    metrics = SpaceMetrics(
        desk_area_cm2=desk_px * area_factor,
        occupied_area_cm2=occ_px * area_factor,
        available_area_cm2=avail_px * area_factor,
        occupancy_ratio=(occ_px / desk_px) if desk_px else 0.0,
        available_ratio=(avail_px / desk_px) if desk_px else 0.0,
        connected_available_regions_cm2=regions,
    )
    return metrics, available


def placement_candidates_placeholder(available_space_mask: np.ndarray, product_width_cm: float, product_depth_cm: float) -> dict[str, Any]:
    _ = available_space_mask, product_width_cm, product_depth_cm
    return {"status": "TODO", "message": "상품 배치 가능성 평가는 후속 단계에서 구현 예정"}


def metrics_to_dict(metrics: SpaceMetrics) -> dict[str, Any]:
    return asdict(metrics)
