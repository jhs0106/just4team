import inspect
from dataclasses import dataclass
from typing import List, Optional

import cv2
import numpy as np


@dataclass
class Detection:
    """
    DINO 검출 결과 하나를 저장하는 자료구조.
    """
    label: str
    score: float
    box_xyxy: List[int]  # [x1, y1, x2, y2]


def normalize_label(label) -> str:
    """label을 비교하기 쉬운 소문자 문자열로 정리한다."""
    if isinstance(label, (list, tuple)):
        label = " ".join(map(str, label))
    return str(label).lower().strip()


def box_area(box: List[int]) -> int:
    """bbox 면적 계산."""
    x1, y1, x2, y2 = box
    return max(0, x2 - x1) * max(0, y2 - y1)


def iou(a: List[int], b: List[int]) -> float:
    """두 bbox의 IoU 계산. NMS에서 사용."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = box_area(a) + box_area(b) - inter

    if union <= 0:
        return 0.0
    return inter / union


def nms(detections: List[Detection], iou_threshold: float = 0.50) -> List[Detection]:
    """
    중복 bbox 제거.
    같은 물체가 여러 label로 중복 검출되는 것을 줄인다.
    """
    if not detections:
        return []

    detections = sorted(detections, key=lambda d: d.score, reverse=True)
    kept: List[Detection] = []

    for det in detections:
        duplicated = False

        for old in kept:
            if iou(det.box_xyxy, old.box_xyxy) >= iou_threshold:
                duplicated = True
                break

        if not duplicated:
            kept.append(det)

    return kept


def filter_object_detections(
    detections: List[Detection],
    image_shape,
    max_box_area_ratio: float = 0.60,
) -> List[Detection]:
    """
    일반 객체 검출 결과 후처리.
    object/item 같은 일반 label과 과도하게 큰 bbox를 제거한다.
    """
    h, w = image_shape[:2]
    image_area = h * w

    blacklist_keywords = [
        "object",
        "item",
        "office item",
        "desktop object",
        "desk object",
    ]

    large_allowed_keywords = [
        "monitor",
        "keyboard",
        "mouse pad",
        "mousepad",
    ]

    filtered: List[Detection] = []

    for det in detections:
        label = normalize_label(det.label)
        area_ratio = box_area(det.box_xyxy) / max(1, image_area)

        if any(k in label for k in blacklist_keywords):
            continue

        if area_ratio > max_box_area_ratio:
            continue

        is_large_allowed = any(k in label for k in large_allowed_keywords)

        if not is_large_allowed and area_ratio > 0.40:
            continue

        filtered.append(det)

    return filtered


def filter_desk_detections(
    detections: List[Detection],
    image_shape,
    max_box_area_ratio: float = 0.98,
    min_box_area_ratio: float = 0.05,
) -> List[Detection]:
    """
    책상 상판 검출 결과 후처리.
    책상은 크기 때문에 object와 다르게 큰 bbox를 허용한다.
    """
    h, w = image_shape[:2]
    image_area = h * w

    valid_keywords = [
        "desk",
        "table",
        "tabletop",
        "desk surface",
        "table surface",
        "desk top",
    ]

    filtered: List[Detection] = []

    for det in detections:
        label = normalize_label(det.label)
        area_ratio = box_area(det.box_xyxy) / max(1, image_area)

        if not any(k in label for k in valid_keywords):
            continue

        if area_ratio < min_box_area_ratio:
            continue

        if area_ratio > max_box_area_ratio:
            continue

        filtered.append(det)

    # 가장 큰 책상 후보가 보통 상판일 가능성이 높다.
    filtered = sorted(filtered, key=lambda d: box_area(d.box_xyxy), reverse=True)

    return filtered


def _post_process_grounding_dino(
    processor,
    outputs,
    input_ids,
    target_sizes,
    box_threshold: float,
    text_threshold: float,
):
    """
    transformers 버전마다 post_process 함수 인자가 조금씩 달라서
    현재 버전에 맞춰 안전하게 호출한다.
    """
    fn = processor.post_process_grounded_object_detection
    sig = inspect.signature(fn)

    kwargs = {
        "outputs": outputs,
        "input_ids": input_ids,
        "target_sizes": target_sizes,
    }

    if "box_threshold" in sig.parameters:
        kwargs["box_threshold"] = box_threshold

    if "text_threshold" in sig.parameters:
        kwargs["text_threshold"] = text_threshold

    if "threshold" in sig.parameters:
        kwargs["threshold"] = box_threshold

    return fn(**kwargs)


def run_grounding_dino(
    image_bgr: np.ndarray,
    prompt: str,
    box_threshold: float = 0.25,
    text_threshold: float = 0.20,
    nms_iou_threshold: float = 0.50,
    max_box_area_ratio: float = 0.60,
    mode: str = "object",
    config_path: Optional[str] = None,
    weights_path: Optional[str] = None,
) -> List[Detection]:
    """
    Grounding DINO로 prompt에 해당하는 객체 bbox를 검출한다.

    mode:
    - object: 책상 위 물체 검출
    - desk: 책상 상판 검출
    """
    try:
        import torch
        from PIL import Image
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        model_id = "IDEA-Research/grounding-dino-tiny"

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(image_rgb)

        device = "cuda" if torch.cuda.is_available() else "cpu"

        processor = AutoProcessor.from_pretrained(model_id)
        model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)
        model.eval()

        # DINO는 문장 끝에 마침표가 있는 prompt 형식을 선호
        if not prompt.strip().endswith("."):
            prompt = prompt.strip() + "."

        inputs = processor(
            images=image_pil,
            text=prompt,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)

        target_sizes = torch.tensor([image_pil.size[::-1]], device=device)

        results = _post_process_grounding_dino(
            processor=processor,
            outputs=outputs,
            input_ids=inputs.get("input_ids"),
            target_sizes=target_sizes,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
        )[0]

        boxes = results.get("boxes", [])
        scores = results.get("scores", [])
        labels = results.get("text_labels", results.get("labels", []))

        h, w = image_bgr.shape[:2]
        raw: List[Detection] = []

        for box, score, label in zip(boxes, scores, labels):
            x1, y1, x2, y2 = box.detach().cpu().numpy().tolist()

            x1 = int(max(0, min(w - 1, round(x1))))
            y1 = int(max(0, min(h - 1, round(y1))))
            x2 = int(max(0, min(w - 1, round(x2))))
            y2 = int(max(0, min(h - 1, round(y2))))

            if x2 <= x1 or y2 <= y1:
                continue

            raw.append(
                Detection(
                    label=normalize_label(label),
                    score=float(score.detach().cpu().item()),
                    box_xyxy=[x1, y1, x2, y2],
                )
            )

        if mode == "desk":
            filtered = filter_desk_detections(
                raw,
                image_shape=image_bgr.shape,
                max_box_area_ratio=0.98,
                min_box_area_ratio=0.05,
            )
        else:
            filtered = filter_object_detections(
                raw,
                image_shape=image_bgr.shape,
                max_box_area_ratio=max_box_area_ratio,
            )
            filtered = nms(filtered, iou_threshold=nms_iou_threshold)

        print(f"[DINO:{mode}] raw={len(raw)}, filtered={len(filtered)}")
        return filtered

    except Exception as e:
        print(f"[WARN] Grounding DINO 실행 실패: {e}")
        return []


def draw_detections(image_bgr: np.ndarray, detections: List[Detection]) -> np.ndarray:
    """검출 bbox를 이미지 위에 그려 확인용 이미지를 만든다."""
    vis = image_bgr.copy()

    for det in detections:
        x1, y1, x2, y2 = det.box_xyxy
        text = f"{det.label} {det.score:.2f}"

        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)

        cv2.putText(
            vis,
            text,
            (x1, max(20, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

    return vis