from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


def make_full_desk_mask(image_shape) -> np.ndarray:
    h, w = image_shape[:2]
    return np.ones((h, w), dtype=np.uint8) * 255


def make_rect_desk_mask(image_shape, roi: Tuple[int, int, int, int]) -> np.ndarray:
    h, w = image_shape[:2]
    x1, y1, x2, y2 = roi
    x1 = max(0, min(w - 1, int(x1)))
    y1 = max(0, min(h - 1, int(y1)))
    x2 = max(0, min(w,     int(x2)))
    y2 = max(0, min(h,     int(y2)))
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
    return mask


def keep_largest_component(mask: np.ndarray) -> np.ndarray:
    binary = (mask > 0).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if num_labels <= 1:
        return mask
    largest = max(range(1, num_labels), key=lambda i: stats[i, cv2.CC_STAT_AREA])
    cleaned = np.zeros_like(mask)
    cleaned[labels == largest] = 255
    return cleaned


def remove_small_regions(mask: np.ndarray, min_area_px: int = 500) -> np.ndarray:
    binary = (mask > 0).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    cleaned = np.zeros_like(mask)
    for label_id in range(1, num_labels):
        if stats[label_id, cv2.CC_STAT_AREA] >= min_area_px:
            cleaned[labels == label_id] = 255
    return cleaned


def build_available_mask(
    occupied_mask: np.ndarray,
    desk_mask: np.ndarray,
    remove_mask: Optional[np.ndarray] = None,
    min_region_area_px: int = 500,
) -> np.ndarray:
    """
    available = desk_mask - occupied_mask
    replace 모드: remove_mask 영역은 occupied에서 제외 → 다시 available로
    """
    desk_mask     = (desk_mask     > 0).astype(np.uint8) * 255
    occupied_mask = (occupied_mask > 0).astype(np.uint8) * 255

    if remove_mask is not None:
        remove_mask = (remove_mask > 0).astype(np.uint8) * 255
        keep_mask   = cv2.bitwise_and(occupied_mask, cv2.bitwise_not(remove_mask))
    else:
        keep_mask = occupied_mask

    occupied_on_desk = cv2.bitwise_and(keep_mask, desk_mask)
    available_mask   = cv2.bitwise_and(desk_mask, cv2.bitwise_not(occupied_on_desk))
    available_mask   = remove_small_regions(available_mask, min_area_px=min_region_area_px)
    return available_mask


def extract_available_regions(
    available_mask: np.ndarray,
    cm_per_px_x: float,
    cm_per_px_y: float,
) -> List[Dict]:
    binary = (available_mask > 0).astype(np.uint8)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    regions: List[Dict] = []
    for label_id in range(1, num_labels):
        x        = int(stats[label_id, cv2.CC_STAT_LEFT])
        y        = int(stats[label_id, cv2.CC_STAT_TOP])
        width_px = int(stats[label_id, cv2.CC_STAT_WIDTH])
        height_px= int(stats[label_id, cv2.CC_STAT_HEIGHT])
        area_px  = int(stats[label_id, cv2.CC_STAT_AREA])
        cx, cy   = centroids[label_id]
        regions.append({
            "region_id": int(label_id),
            "bbox_px":   {"x": x, "y": y, "width": width_px, "height": height_px},
            "center_px": {"x": round(float(cx), 2), "y": round(float(cy), 2)},
            "size_cm":   {
                "width": round(float(width_px  * cm_per_px_x), 2),
                "depth": round(float(height_px * cm_per_px_y), 2),
            },
            "area_px":   area_px,
            "area_cm2":  round(float(area_px * cm_per_px_x * cm_per_px_y), 2),
        })
    regions.sort(key=lambda r: r["area_px"], reverse=True)
    return regions


def analyze_space(
    occupied_mask: np.ndarray,
    desk_width_cm: float,
    desk_depth_cm: float,
    desk_mask: np.ndarray,
    remove_mask: Optional[np.ndarray] = None,
    min_region_area_px: int = 500,
) -> Dict:
    h, w = occupied_mask.shape[:2]
    desk_mask     = (desk_mask     > 0).astype(np.uint8) * 255
    occupied_mask = (occupied_mask > 0).astype(np.uint8) * 255

    available_mask = build_available_mask(
        occupied_mask=occupied_mask,
        desk_mask=desk_mask,
        remove_mask=remove_mask,
        min_region_area_px=min_region_area_px,
    )

    if remove_mask is not None:
        remove_mask = (remove_mask > 0).astype(np.uint8) * 255
        keep_mask   = cv2.bitwise_and(occupied_mask, cv2.bitwise_not(remove_mask))
    else:
        keep_mask = occupied_mask

    occupied_on_desk  = cv2.bitwise_and(keep_mask, desk_mask)
    desk_area_px      = int(np.count_nonzero(desk_mask))
    occupied_area_px  = int(np.count_nonzero(occupied_on_desk))
    available_area_px = int(np.count_nonzero(available_mask))

    ys, xs = np.where(desk_mask > 0)
    if len(xs) > 0:
        desk_px_width = max(1, int(xs.max() - xs.min() + 1))
        desk_px_depth = max(1, int(ys.max() - ys.min() + 1))
    else:
        desk_px_width, desk_px_depth = w, h

    cm_per_px_x = desk_width_cm / desk_px_width
    cm_per_px_y = desk_depth_cm / desk_px_depth

    regions = extract_available_regions(available_mask, cm_per_px_x, cm_per_px_y)

    return {
        "image_size_px":    {"width": int(w), "height": int(h)},
        "desk_area_px":     desk_area_px,
        "occupied_area_px": occupied_area_px,
        "available_area_px":available_area_px,
        "available_area_cm2": round(float(available_area_px * cm_per_px_x * cm_per_px_y), 2),
        "cm_per_px":        {"x": float(cm_per_px_x), "y": float(cm_per_px_y)},
        "available_regions": regions,
    }


def draw_available_overlay(
    image_bgr: np.ndarray,
    available_mask: np.ndarray,
    desk_mask: Optional[np.ndarray] = None,
    alpha: float = 0.40,
) -> np.ndarray:
    vis   = image_bgr.copy()
    green = np.zeros_like(image_bgr)
    green[:, :, 1] = 255
    available_bool = available_mask > 0
    if np.count_nonzero(available_bool) > 0:
        blended = cv2.addWeighted(image_bgr, 1 - alpha, green, alpha, 0)
        vis[available_bool] = blended[available_bool]
    if desk_mask is not None:
        contours, _ = cv2.findContours(
            (desk_mask > 0).astype(np.uint8) * 255,
            cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
        )
        cv2.drawContours(vis, contours, -1, (0, 255, 255), 2)
    return vis
