from __future__ import annotations

import cv2
import numpy as np


def postprocess_occupied_mask(
        raw_mask: np.ndarray,
        min_component_area: int = 600,
        dilation_kernel: int = 7,
        dilation_iter: int = 1,
        feather_ksize: int = 0,
) -> np.ndarray:
    mask = (raw_mask > 0).astype(np.uint8) * 255

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    filtered = np.zeros_like(mask)
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_component_area:
            filtered[labels == i] = 255

    if dilation_kernel > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilation_kernel, dilation_kernel))
        filtered = cv2.dilate(filtered, k, iterations=dilation_iter)

    if feather_ksize and feather_ksize > 1:
        ksize = feather_ksize if feather_ksize % 2 == 1 else feather_ksize + 1
        filtered = cv2.GaussianBlur(filtered, (ksize, ksize), 0)
        _, filtered = cv2.threshold(filtered, 127, 255, cv2.THRESH_BINARY)

    return filtered


def make_full_image_desk_mask(image_shape: tuple[int, int, int]) -> np.ndarray:
    h, w = image_shape[:2]
    return np.full((h, w), 255, dtype=np.uint8)
