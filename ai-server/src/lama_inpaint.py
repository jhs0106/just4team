from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def inpaint_with_lama_placeholder(
        image_bgr: np.ndarray,
        mask: np.ndarray,
        lama_model_dir: str | None = None,
) -> np.ndarray:
    if lama_model_dir and not Path(lama_model_dir).exists():
        raise FileNotFoundError(f"LaMa 모델 디렉토리가 없습니다: {lama_model_dir}")

    # Placeholder: OpenCV Telea inpaint 사용 (LaMa 인터페이스 대체)
    mask_u8 = (mask > 0).astype(np.uint8) * 255
    return cv2.inpaint(image_bgr, mask_u8, 3, cv2.INPAINT_TELEA)
