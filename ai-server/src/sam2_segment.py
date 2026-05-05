from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Iterable, Tuple

import cv2
import numpy as np


DEFAULT_SAM2_CHECKPOINT = "/app/weights/sam2.1_hiera_tiny.pt"
DEFAULT_SAM2_CONFIG = "/app/external/sam2/sam2/configs/sam2.1/sam2.1_hiera_t.yaml"


def _grabcut_fallback(
    image_bgr: np.ndarray,
    boxes_xyxy: Iterable[Tuple[int, int, int, int]],
) -> np.ndarray:
    h, w = image_bgr.shape[:2]
    occ = np.zeros((h, w), dtype=np.uint8)

    for (x1, y1, x2, y2) in boxes_xyxy:
        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2, y2 = min(w - 1, int(x2)), min(h - 1, int(y2))

        if x2 <= x1 or y2 <= y1:
            continue

        rect = (x1, y1, x2 - x1, y2 - y1)
        mask = np.zeros((h, w), np.uint8)
        bgd = np.zeros((1, 65), np.float64)
        fgd = np.zeros((1, 65), np.float64)

        try:
            cv2.grabCut(image_bgr, mask, rect, bgd, fgd, 2, cv2.GC_INIT_WITH_RECT)
            fg = np.where((mask == 1) | (mask == 3), 255, 0).astype(np.uint8)
            occ = cv2.bitwise_or(occ, fg)
        except cv2.error:
            occ[y1:y2, x1:x2] = 255

    return occ


@lru_cache(maxsize=2)
def _load_sam2_predictor(
    sam2_checkpoint: str = DEFAULT_SAM2_CHECKPOINT,
    sam2_config: str = DEFAULT_SAM2_CONFIG,
):
    try:
        import torch
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
    except Exception as exc:
        raise RuntimeError(
            "SAM2 import 실패. /app/external/sam2에서 pip install -e .가 성공했는지 확인하세요."
        ) from exc

    checkpoint = Path(sam2_checkpoint)
    if not checkpoint.exists():
        raise FileNotFoundError(f"SAM2 checkpoint 파일이 없습니다: {sam2_checkpoint}")

    config = Path(sam2_config)
    if not config.exists():
        raise FileNotFoundError(f"SAM2 config 파일이 없습니다: {sam2_config}")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # SAM2 build_sam2는 config 파일명 또는 repo 기준 config 경로를 받는 경우가 있어
    # 절대경로 실패 시 상대 config 경로로 한 번 더 시도한다.
    try:
        model = build_sam2(str(config), str(checkpoint), device=device)
    except Exception:
        relative_config = "configs/sam2.1/sam2.1_hiera_t.yaml"
        model = build_sam2(relative_config, str(checkpoint), device=device)

    predictor = SAM2ImagePredictor(model)
    return predictor, device


def segment_with_sam2_from_boxes(
    image_bgr: np.ndarray,
    boxes_xyxy: Iterable[Tuple[int, int, int, int]],
    sam2_checkpoint: str | None = None,
    sam2_config: str | None = None,
    use_fallback: bool = False,
) -> np.ndarray:
    boxes = [tuple(map(int, box)) for box in boxes_xyxy]

    h, w = image_bgr.shape[:2]
    if not boxes:
        return np.zeros((h, w), dtype=np.uint8)

    if use_fallback:
        return _grabcut_fallback(image_bgr, boxes)

    checkpoint = sam2_checkpoint or DEFAULT_SAM2_CHECKPOINT
    config = sam2_config or DEFAULT_SAM2_CONFIG

    try:
        import torch

        predictor, device = _load_sam2_predictor(checkpoint, config)

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        predictor.set_image(image_rgb)

        occ = np.zeros((h, w), dtype=np.uint8)

        for (x1, y1, x2, y2) in boxes:
            x1 = max(0, min(w - 1, int(x1)))
            y1 = max(0, min(h - 1, int(y1)))
            x2 = max(0, min(w - 1, int(x2)))
            y2 = max(0, min(h - 1, int(y2)))

            if x2 <= x1 or y2 <= y1:
                continue

            box_np = np.array([x1, y1, x2, y2], dtype=np.float32)

            with torch.inference_mode():
                masks, scores, _ = predictor.predict(
                    point_coords=None,
                    point_labels=None,
                    box=box_np,
                    multimask_output=True,
                )

            if masks is None or len(masks) == 0:
                continue

            best_idx = int(np.argmax(scores)) if scores is not None else 0
            best_mask = masks[best_idx].astype(np.uint8) * 255
            occ = cv2.bitwise_or(occ, best_mask)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return occ

    except Exception as exc:
        print(f"[WARN] SAM2 실행 실패, GrabCut fallback 사용: {exc}")
        return _grabcut_fallback(image_bgr, boxes)


def overlay_mask(
    image_bgr: np.ndarray,
    mask: np.ndarray,
    color=(30, 30, 255),
    alpha=0.4,
) -> np.ndarray:
    overlay = image_bgr.copy()
    colored = np.zeros_like(image_bgr)
    colored[:, :] = color

    sel = mask > 0
    if np.any(sel):
        blended = cv2.addWeighted(image_bgr, 1 - alpha, colored, alpha, 0)
        overlay[sel] = blended[sel]

    return overlay