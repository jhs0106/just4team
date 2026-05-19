from __future__ import annotations

import inspect
from dataclasses import dataclass
from functools import lru_cache
from typing import List, Tuple

import cv2
import numpy as np

HF_MODEL_ID = "IDEA-Research/grounding-dino-tiny"

# 너무 모호해서 마스크로 만들면 안 되는 label 키워드
_BLACKLIST = {"object", "item", "office item", "desktop object", "desk object"}

# 이미지 면적 40% 초과여도 제거/감지 대상으로 허용하는 물체 (실제로 크기가 큰 물체)
_LARGE_ALLOWED = {"monitor", "keyboard", "mouse pad", "mousepad", "desk", "table"}


@dataclass
class Detection:
    label: str
    score: float
    box_xyxy: Tuple[int, int, int, int]


# ── 모델 싱글톤 ───────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _load_model(model_id: str = HF_MODEL_ID):
    import torch
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device).eval()
    return processor, model, device


# ── NMS ──────────────────────────────────────────────────────────────────────

def _box_area(box: Tuple[int, int, int, int]) -> int:
    x1, y1, x2, y2 = box
    return max(0, x2 - x1) * max(0, y2 - y1)


def _iou(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = _box_area(a) + _box_area(b) - inter
    return inter / union if union > 0 else 0.0


def _nms(detections: List[Detection], iou_threshold: float = 0.50) -> List[Detection]:
    # score 순 정렬 후 IoU 기반 중복 제거
    detections = sorted(detections, key=lambda d: d.score, reverse=True)
    kept: List[Detection] = []
    for det in detections:
        if not any(_iou(det.box_xyxy, k.box_xyxy) >= iou_threshold for k in kept):
            kept.append(det)
    return kept


# ── 필터 ─────────────────────────────────────────────────────────────────────

def _normalize_label(label) -> str:
    if isinstance(label, (list, tuple)):
        label = " ".join(map(str, label))
    return str(label).lower().strip()


def _filter_detections(
    detections: List[Detection],
    image_shape,
    max_area_ratio: float,
) -> List[Detection]:
    # blacklist 키워드 제거 + 면적 필터 (모니터/키보드/마우스패드/책상은 대형 허용)
    h, w = image_shape[:2]
    image_area = h * w
    result = []
    for det in detections:
        label = _normalize_label(det.label)
        if any(k in label for k in _BLACKLIST):
            continue
        area_ratio = _box_area(det.box_xyxy) / max(1, image_area)
        if area_ratio > max_area_ratio:
            continue
        is_large_allowed = any(k in label for k in _LARGE_ALLOWED)
        if not is_large_allowed and area_ratio > 0.40:
            continue
        result.append(det)
    return result


_DESK_VALID = {"desk", "table", "tabletop", "desk surface", "table surface", "desk top"}


def _filter_desk_detections(
    detections: List[Detection],
    image_shape,
    max_area_ratio: float = 0.98,
    min_area_ratio: float = 0.05,
) -> List[Detection]:
    # 책상/테이블 label만 통과. 크기 범위 필터. 가장 큰 것 우선 정렬.
    h, w = image_shape[:2]
    image_area = h * w
    result = []
    for det in detections:
        label = _normalize_label(det.label)
        if not any(k in label for k in _DESK_VALID):
            continue
        area_ratio = _box_area(det.box_xyxy) / max(1, image_area)
        if not (min_area_ratio <= area_ratio <= max_area_ratio):
            continue
        result.append(det)
    return sorted(result, key=lambda d: _box_area(d.box_xyxy), reverse=True)


# ── Post-process (transformers 버전 대응) ────────────────────────────────────

def _post_process(processor, outputs, input_ids, image_size_hw, box_threshold, text_threshold):
    # transformers 버전마다 API가 달라서 signature 읽고 맞는 인자만 전달
    if not hasattr(processor, "post_process_grounded_object_detection"):
        return processor.post_process_object_detection(
            outputs, threshold=box_threshold, target_sizes=[image_size_hw]
        )[0]

    fn = processor.post_process_grounded_object_detection
    sig = inspect.signature(fn)
    kwargs = {}
    if "outputs"        in sig.parameters: kwargs["outputs"]        = outputs
    if "input_ids"      in sig.parameters: kwargs["input_ids"]      = input_ids
    if "target_sizes"   in sig.parameters: kwargs["target_sizes"]   = [image_size_hw]
    if "threshold"      in sig.parameters: kwargs["threshold"]      = box_threshold
    elif "box_threshold" in sig.parameters: kwargs["box_threshold"] = box_threshold
    if "text_threshold" in sig.parameters: kwargs["text_threshold"] = text_threshold

    try:
        return fn(**kwargs)[0]
    except TypeError:
        return fn(outputs, input_ids,
                  threshold=box_threshold,
                  text_threshold=text_threshold,
                  target_sizes=[image_size_hw])[0]


# ── OpenCV fallback ───────────────────────────────────────────────────────────

def _opencv_fallback(image_bgr: np.ndarray, prompt: str, box_threshold: float) -> List[Detection]:
    # DINO 로드 실패 시 OpenCV contour 기반 fallback
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


# ── 메인 ─────────────────────────────────────────────────────────────────────

def run_grounding_dino(
    image_bgr: np.ndarray,
    prompt: str,
    box_threshold: float = 0.25,
    text_threshold: float = 0.20,
    max_area_ratio: float = 0.60,
    nms_iou_threshold: float = 0.50,
    mode: str = "object",  # "object" | "desk"
) -> List[Detection]:
    # DINO 텍스트 프롬프트 물체 검출 → NMS → 면적 필터. 실패 시 OpenCV fallback
    if not prompt.strip():
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

        # text_labels (신규 transformers) / labels (구버전) 모두 대응
        raw_labels = results.get("text_labels", results.get("labels", []))

        raw: List[Detection] = []
        for box, score, label in zip(
            results.get("boxes", []),
            results.get("scores", []),
            raw_labels,
        ):
            x1, y1, x2, y2 = [int(round(float(v))) for v in box.tolist()]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x2 <= x1 or y2 <= y1:
                continue
            raw.append(Detection(_normalize_label(label), float(score), (x1, y1, x2, y2)))

        # NMS → mode별 필터
        deduped = _nms(raw, iou_threshold=nms_iou_threshold)
        if mode == "desk":
            filtered = _filter_desk_detections(deduped, image_bgr.shape)
        else:
            filtered = _filter_detections(deduped, image_bgr.shape, max_area_ratio)
        print(f"[DINO:{mode}] raw={len(raw)} → NMS={len(deduped)} → filter={len(filtered)}")
        return filtered

    except Exception as e:
        print(f"[DINO] 로드 실패, OpenCV fallback 사용: {e}")
        return _opencv_fallback(image_bgr, prompt, box_threshold)


def draw_detections(image_bgr: np.ndarray, detections: List[Detection]) -> np.ndarray:
    # 검출 bbox를 이미지 위에 그려 디버깅용 이미지 생성
    vis = image_bgr.copy()
    for det in detections:
        x1, y1, x2, y2 = det.box_xyxy
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            vis, f"{det.label} {det.score:.2f}",
            (x1, max(20, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2, cv2.LINE_AA,
        )
    return vis
