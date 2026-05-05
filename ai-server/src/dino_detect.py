from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import List, Tuple
import inspect

import cv2
import numpy as np


@dataclass
class Detection:
    label: str
    score: float
    box_xyxy: Tuple[int, int, int, int]


DEFAULT_HF_MODEL_ID = "IDEA-Research/grounding-dino-tiny"


def _labels_from_prompt(prompt: str) -> list[str]:
    return [token.strip() for token in prompt.split(".") if token.strip()]


@lru_cache(maxsize=2)
def _load_hf_grounding_dino(model_id: str = DEFAULT_HF_MODEL_ID):
    try:
        import torch
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
    except Exception as exc:
        raise RuntimeError(
            "Hugging Face Grounding DINO 로드에 필요한 패키지가 없습니다. "
            "requirements.txt에 transformers, accelerate, safetensors를 추가하고 다시 설치하세요."
        ) from exc

    device = "cuda" if torch.cuda.is_available() else "cpu"

    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id)
    model.to(device)
    model.eval()

    return processor, model, device


def _run_opencv_fallback(
    image_bgr: np.ndarray,
    prompt: str,
    box_threshold: float = 0.25,
) -> List[Detection]:
    labels = _labels_from_prompt(prompt)
    if not labels:
        return []

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

        # fallback이 이미지 전체를 하나의 객체로 잡는 문제 방지
        box_area_ratio = (bw * bh) / float(h * w)
        if box_area_ratio > 0.35:
            continue

        # 벽/책상 경계처럼 지나치게 길쭉한 contour 제거
        aspect = bw / max(bh, 1)
        if aspect > 8 or aspect < 0.12:
            continue

        score = float(min(0.99, max(0.5, area / (h * w) * 8)))
        if score < box_threshold:
            continue

        label = labels[idx % len(labels)]
        detections.append(
            Detection(
                label=label,
                score=score,
                box_xyxy=(x, y, x + bw, y + bh),
            )
        )

        idx += 1
        if idx >= len(labels):
            break

    return detections


def _post_process_grounding_dino(
    processor,
    outputs,
    input_ids,
    image_size_hw: tuple[int, int],
    box_threshold: float,
    text_threshold: float,
):
    """
    Transformers 버전마다 post_process_grounded_object_detection()의
    인자명이 조금씩 달라서 현재 설치된 버전의 signature를 읽고 맞는 인자만 넣는다.
    """
    if not hasattr(processor, "post_process_grounded_object_detection"):
        return processor.post_process_object_detection(
            outputs,
            threshold=box_threshold,
            target_sizes=[image_size_hw],
        )[0]

    post_fn = processor.post_process_grounded_object_detection
    sig = inspect.signature(post_fn)

    kwargs = {}

    if "outputs" in sig.parameters:
        kwargs["outputs"] = outputs

    if "input_ids" in sig.parameters:
        kwargs["input_ids"] = input_ids

    if "target_sizes" in sig.parameters:
        kwargs["target_sizes"] = [image_size_hw]

    # Transformers 5.x 계열은 threshold를 쓰는 경우가 많고,
    # 일부 예전 예제/버전은 box_threshold를 사용한다.
    if "threshold" in sig.parameters:
        kwargs["threshold"] = box_threshold
    elif "box_threshold" in sig.parameters:
        kwargs["box_threshold"] = box_threshold

    if "text_threshold" in sig.parameters:
        kwargs["text_threshold"] = text_threshold

    # outputs가 positional-only 또는 signature에 안 잡히는 경우 대비
    try:
        results = post_fn(**kwargs)
    except TypeError:
        results = post_fn(
            outputs,
            input_ids,
            threshold=box_threshold,
            text_threshold=text_threshold,
            target_sizes=[image_size_hw],
        )

    return results[0]


def run_grounding_dino(
    image_bgr: np.ndarray,
    prompt: str,
    box_threshold: float = 0.25,
    text_threshold: float = 0.2,
    config_path: str | None = None,
    weights_path: str | None = None,
    use_fallback: bool = False,
    hf_model_id: str = DEFAULT_HF_MODEL_ID,
) -> List[Detection]:
    """
    Hugging Face Transformers 기반 Grounding DINO detector.

    기존 IDEA-Research/GroundingDINO repo 방식은 CUDA extension 컴파일이 필요하고,
    PyTorch nightly / RTX 5070 환경에서 충돌할 수 있어 사용하지 않는다.

    config_path, weights_path는 기존 CLI 호환성을 위해 남겨두지만 HF 방식에서는 사용하지 않는다.
    """
    _ = config_path, weights_path

    if use_fallback:
        return _run_opencv_fallback(image_bgr, prompt, box_threshold)

    labels = _labels_from_prompt(prompt)
    if not labels:
        return []

    try:
        import torch
        from PIL import Image
    except Exception as exc:
        raise RuntimeError("Pillow 또는 torch import 실패") from exc

    try:
        processor, model, device = _load_hf_grounding_dino(hf_model_id)

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(image_rgb)

        text = prompt.strip()
        if not text.endswith("."):
            text += "."

        inputs = processor(
            images=image_pil,
            text=text,
            return_tensors="pt",
        )

        inputs = {
            key: value.to(device) if hasattr(value, "to") else value
            for key, value in inputs.items()
        }

        with torch.no_grad():
            outputs = model(**inputs)

        h, w = image_bgr.shape[:2]
        results = _post_process_grounding_dino(
            processor=processor,
            outputs=outputs,
            input_ids=inputs.get("input_ids"),
            image_size_hw=(h, w),
            box_threshold=box_threshold,
            text_threshold=text_threshold,
        )

        detections: List[Detection] = []

        boxes = results.get("boxes", [])
        scores = results.get("scores", [])
        result_labels = results.get("labels", [])

        for box, score, label in zip(boxes, scores, result_labels):
            x1, y1, x2, y2 = [int(round(float(v))) for v in box.tolist()]

            x1 = max(0, min(w - 1, x1))
            y1 = max(0, min(h - 1, y1))
            x2 = max(0, min(w - 1, x2))
            y2 = max(0, min(h - 1, y2))

            if x2 <= x1 or y2 <= y1:
                continue

            detections.append(
                Detection(
                    label=str(label),
                    score=float(score),
                    box_xyxy=(x1, y1, x2, y2),
                )
            )

        return detections

    except Exception as exc:
        print(f"[WARN] Hugging Face Grounding DINO 실행 실패, OpenCV fallback 사용: {exc}")
        return _run_opencv_fallback(image_bgr, prompt, box_threshold)


def draw_detections(image_bgr: np.ndarray, detections: List[Detection]) -> np.ndarray:
    vis = image_bgr.copy()

    for det in detections:
        x1, y1, x2, y2 = det.box_xyxy

        cv2.rectangle(vis, (x1, y1), (x2, y2), (80, 255, 80), 2)

        txt = f"{det.label}: {det.score:.2f}"

        cv2.putText(
            vis,
            txt,
            (x1, max(18, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (20, 20, 20),
            3,
        )
        cv2.putText(
            vis,
            txt,
            (x1, max(18, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
        )

    return vis