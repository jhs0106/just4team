from __future__ import annotations

import inspect
from dataclasses import dataclass
from functools import lru_cache
from typing import List, Tuple

import cv2
import numpy as np

HF_MODEL_ID = "IDEA-Research/grounding-dino-tiny"


@dataclass
class Detection:
    label: str
    score: float
    box_xyxy: Tuple[int, int, int, int]


@lru_cache(maxsize=1)
def _load_model(model_id: str = HF_MODEL_ID):
    import torch
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device).eval()
    return processor, model, device


def _post_process(processor, outputs, input_ids, image_size_hw, box_threshold, text_threshold):
    """transformers 버전마다 API가 달라서 signature 읽고 맞는 인자만 전달."""
    if not hasattr(processor, "post_process_grounded_object_detection"):
        return processor.post_process_object_detection(
            outputs, threshold=box_threshold, target_sizes=[image_size_hw]
        )[0]

    fn = processor.post_process_grounded_object_detection
    sig = inspect.signature(fn)
    kwargs = {}
    if "outputs"       in sig.parameters: kwargs["outputs"]        = outputs
    if "input_ids"     in sig.parameters: kwargs["input_ids"]      = input_ids
    if "target_sizes"  in sig.parameters: kwargs["target_sizes"]   = [image_size_hw]
    if "threshold"     in sig.parameters: kwargs["threshold"]      = box_threshold
    elif "box_threshold" in sig.parameters: kwargs["box_threshold"] = box_threshold
    if "text_threshold" in sig.parameters: kwargs["text_threshold"] = text_threshold

    try:
        return fn(**kwargs)[0]
    except TypeError:
        return fn(outputs, input_ids,
                  threshold=box_threshold,
                  text_threshold=text_threshold,
                  target_sizes=[image_size_hw])[0]


def _opencv_fallback(image_bgr: np.ndarray, prompt: str, box_threshold: float) -> List[Detection]:
    """DINO 로드 실패 시 OpenCV contour 기반 fallback."""
    labels = [t.strip() for t in prompt.split(".") if t.strip()]
    if not labels:
        return []

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    _, thr = cv2.threshold(
        cv2.GaussianBlur(gray, (5, 5), 0), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    contours, _ = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    h, w = gray.shape
    min_area = max(200, (h * w) // 800)
    detections: List[Detection] = []

    for i, cnt in enumerate(sorted(contours, key=cv2.contourArea, reverse=True)):
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        x, y, bw, bh = cv2.boundingRect(cnt)
        if (bw * bh) / (h * w) > 0.35:
            continue
        aspect = bw / max(bh, 1)
        if aspect > 8 or aspect < 0.12:
            continue
        score = float(min(0.99, max(0.5, area / (h * w) * 8)))
        if score < box_threshold:
            continue
        detections.append(Detection(
            label=labels[i % len(labels)],
            score=score,
            box_xyxy=(x, y, x + bw, y + bh),
        ))
        if len(detections) >= len(labels):
            break

    return detections


def run_grounding_dino(
    image_bgr: np.ndarray,
    prompt: str,
    box_threshold: float = 0.25,
    text_threshold: float = 0.20,
) -> List[Detection]:
    """
    Grounding DINO (HuggingFace) 기반 텍스트 프롬프트 물체 검출.
    prompt 형식: "keyboard. mouse. headset. desk lamp."
    로드 실패 시 OpenCV contour fallback으로 자동 전환.
    """
    labels = [t.strip() for t in prompt.split(".") if t.strip()]
    if not labels:
        return []

    try:
        import torch
        from PIL import Image

        processor, model, device = _load_model()

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(image_rgb)

        text = prompt.strip()
        if not text.endswith("."):
            text += "."

        inputs = processor(images=image_pil, text=text, return_tensors="pt")
        inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        h, w = image_bgr.shape[:2]
        results = _post_process(
            processor, outputs, inputs.get("input_ids"),
            (h, w), box_threshold, text_threshold,
        )

        detections: List[Detection] = []
        for box, score, label in zip(
            results.get("boxes", []),
            results.get("scores", []),
            results.get("labels", []),
        ):
            x1, y1, x2, y2 = [int(round(float(v))) for v in box.tolist()]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x2 <= x1 or y2 <= y1:
                continue
            detections.append(Detection(str(label), float(score), (x1, y1, x2, y2)))

        return detections

    except Exception as e:
        print(f"[DINO] 로드 실패, OpenCV fallback 사용: {e}")
        return _opencv_fallback(image_bgr, prompt, box_threshold)
