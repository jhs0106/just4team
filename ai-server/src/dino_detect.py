from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np


@dataclass
class Detection:
    label: str
    score: float
    box_xyxy: Tuple[int, int, int, int]


def _ensure_grounding_dino_importable() -> None:
    try:
        import groundingdino  # noqa: F401
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "Grounding DINO import 실패. 필요한 패키지/코드와 weight를 확인하세요. "
            "예: pip install -e GroundingDINO, dino config/weight 지정"
        ) from exc


def run_grounding_dino(
        image_bgr: np.ndarray,
        prompt: str,
        box_threshold: float = 0.25,
        text_threshold: float = 0.2,
        config_path: str | None = None,
        weights_path: str | None = None,
) -> List[Detection]:
    """Grounding DINO wrapper.

    실제 모델 연결 전에도 파이프라인 검증을 위해 OpenCV fallback을 제공한다.
    Grounding DINO 설치/weight 준비 시 아래 fallback을 교체하여 사용하면 된다.
    """
    if weights_path and not Path(weights_path).exists():
        raise FileNotFoundError(f"Grounding DINO weight 파일이 없습니다: {weights_path}")
    if config_path and not Path(config_path).exists():
        raise FileNotFoundError(f"Grounding DINO config 파일이 없습니다: {config_path}")

    labels = [token.strip() for token in prompt.split(".") if token.strip()]
    if not labels:
        return []

    # Lightweight fallback: salient regions as proxy detections.
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thr = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    detections: List[Detection] = []
    idx = 0
    h, w = gray.shape
    min_area = max(200, (h * w) // 800)
    for cnt in sorted(contours, key=cv2.contourArea, reverse=True):
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        x, y, bw, bh = cv2.boundingRect(cnt)
        score = float(min(0.99, max(0.5, area / (h * w) * 8)))
        if score < box_threshold:
            continue
        label = labels[idx % len(labels)]
        detections.append(Detection(label=label, score=score, box_xyxy=(x, y, x + bw, y + bh)))
        idx += 1
        if idx >= len(labels):
            break
    return detections


def draw_detections(image_bgr: np.ndarray, detections: List[Detection]) -> np.ndarray:
    vis = image_bgr.copy()
    for det in detections:
        x1, y1, x2, y2 = det.box_xyxy
        cv2.rectangle(vis, (x1, y1), (x2, y2), (80, 255, 80), 2)
        txt = f"{det.label}: {det.score:.2f}"
        cv2.putText(vis, txt, (x1, max(18, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 3)
        cv2.putText(vis, txt, (x1, max(18, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    return vis
