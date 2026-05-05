from __future__ import annotations

from pathlib import Path
from typing import Iterable, Tuple

import cv2
import numpy as np


def segment_with_sam2_from_boxes(
        image_bgr: np.ndarray,
        boxes_xyxy: Iterable[Tuple[int, int, int, int]],
        sam2_checkpoint: str | None = None,
        sam2_config: str | None = None,
) -> np.ndarray:
    if sam2_checkpoint and not Path(sam2_checkpoint).exists():
        raise FileNotFoundError(f"SAM2 checkpoint 파일이 없습니다: {sam2_checkpoint}")
    if sam2_config and not Path(sam2_config).exists():
        raise FileNotFoundError(f"SAM2 config 파일이 없습니다: {sam2_config}")

    # Fallback 구현: bbox 영역을 occupied로 채운 후 GrabCut으로 대략 정제.
    h, w = image_bgr.shape[:2]
    occ = np.zeros((h, w), dtype=np.uint8)
    for (x1, y1, x2, y2) in boxes_xyxy:
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)
        if x2 <= x1 or y2 <= y1:
            continue
        rect = (x1, y1, x2 - x1, y2 - y1)
        mask = np.zeros((h, w), np.uint8)
        bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
        try:
            cv2.grabCut(image_bgr, mask, rect, bgd, fgd, 2, cv2.GC_INIT_WITH_RECT)
            fg = np.where((mask == 1) | (mask == 3), 255, 0).astype(np.uint8)
            occ = cv2.bitwise_or(occ, fg)
        except cv2.error:
            occ[y1:y2, x1:x2] = 255
    return occ


def overlay_mask(image_bgr: np.ndarray, mask: np.ndarray, color=(30, 30, 255), alpha=0.4) -> np.ndarray:
    overlay = image_bgr.copy()
    colored = np.zeros_like(image_bgr)
    colored[:, :] = color
    sel = mask > 0
    overlay[sel] = cv2.addWeighted(image_bgr, 1 - alpha, colored, alpha, 0)[sel]
    return overlay
