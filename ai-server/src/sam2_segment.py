from typing import Dict, List, Optional

import cv2
import numpy as np

from dino_detect import Detection


def empty_mask(image_bgr: np.ndarray) -> np.ndarray:
    """이미지 크기와 같은 빈 mask 생성."""
    h, w = image_bgr.shape[:2]
    return np.zeros((h, w), dtype=np.uint8)


def segment_instances_with_sam2(
    image_bgr: np.ndarray,
    detections: List[Detection],
    sam2_checkpoint: str,
    sam2_config: str,
) -> Dict:
    """
    DINO bbox를 SAM2 box prompt로 넣어 객체별 mask를 만든다.

    반환:
    - merged_mask: 모든 객체 mask를 합친 mask
    - instances: 객체별 mask 리스트
    """
    if not detections:
        return {
            "merged_mask": empty_mask(image_bgr),
            "instances": [],
        }

    try:
        import torch
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        device = "cuda" if torch.cuda.is_available() else "cpu"

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        sam2_model = build_sam2(
            config_file=sam2_config,
            ckpt_path=sam2_checkpoint,
            device=device,
        )

        predictor = SAM2ImagePredictor(sam2_model)
        predictor.set_image(image_rgb)

        merged_mask = empty_mask(image_bgr)
        instances = []

        for det in detections:
            box_np = np.array(det.box_xyxy, dtype=np.float32)

            masks, scores, _ = predictor.predict(
                box=box_np,
                multimask_output=True,
            )

            if masks is None or len(masks) == 0:
                continue

            best_idx = int(np.argmax(scores))
            best_mask = masks[best_idx].astype(np.uint8) * 255

            merged_mask = cv2.bitwise_or(merged_mask, best_mask)

            instances.append(
                {
                    "label": det.label,
                    "score": det.score,
                    "box_xyxy": det.box_xyxy,
                    "mask": best_mask,
                }
            )

        return {
            "merged_mask": merged_mask,
            "instances": instances,
        }

    except Exception as e:
        print(f"[WARN] SAM2 실행 실패: {e}")
        return {
            "merged_mask": empty_mask(image_bgr),
            "instances": [],
        }


def postprocess_mask(
    mask: Optional[np.ndarray],
    dilate_kernel: int = 7,
    dilate_iter: int = 1,
    close_kernel: int = 5,
) -> np.ndarray:
    """
    mask 후처리.

    close:
        작은 구멍을 메운다.
    dilation:
        객체 경계가 남지 않게 mask를 약간 확장한다.
    """
    if mask is None:
        return None

    mask = (mask > 0).astype(np.uint8) * 255

    if close_kernel > 0:
        k = np.ones((close_kernel, close_kernel), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)

    if dilate_kernel > 0 and dilate_iter > 0:
        k = np.ones((dilate_kernel, dilate_kernel), np.uint8)
        mask = cv2.dilate(mask, k, iterations=dilate_iter)

    return mask


def postprocess_instances(
    instances: List[Dict],
    dilate_kernel: int = 7,
    dilate_iter: int = 1,
    close_kernel: int = 5,
) -> List[Dict]:
    """객체별 mask에도 동일한 후처리를 적용한다."""
    processed = []

    for inst in instances:
        new_inst = dict(inst)
        new_inst["mask"] = postprocess_mask(
            inst["mask"],
            dilate_kernel=dilate_kernel,
            dilate_iter=dilate_iter,
            close_kernel=close_kernel,
        )
        processed.append(new_inst)

    return processed


def merge_instance_masks(
    instances: List[Dict],
    image_shape=None,
) -> np.ndarray:
    """객체별 mask 리스트를 하나의 mask로 합친다."""
    if not instances:
        if image_shape is None:
            return None
        h, w = image_shape[:2]
        return np.zeros((h, w), dtype=np.uint8)

    merged = np.zeros_like(instances[0]["mask"], dtype=np.uint8)

    for inst in instances:
        merged = cv2.bitwise_or(merged, inst["mask"])

    return merged


def build_remove_mask_from_labels(
    instances: List[Dict],
    remove_labels: List[str],
    image_shape=None,
) -> np.ndarray:
    """
    특정 label에 해당하는 객체만 지울 remove_mask 생성.

    예:
    remove_labels = ["cup", "book"]
    """
    if not instances:
        if image_shape is None:
            return None
        h, w = image_shape[:2]
        return np.zeros((h, w), dtype=np.uint8)

    remove_labels = [x.lower().strip() for x in remove_labels]
    remove_mask = np.zeros_like(instances[0]["mask"], dtype=np.uint8)

    for inst in instances:
        label = str(inst["label"]).lower().strip()

        if any(target in label for target in remove_labels):
            remove_mask = cv2.bitwise_or(remove_mask, inst["mask"])

    return remove_mask


def build_remove_mask_by_indices(
    instances: List[Dict],
    indices: List[int],
    image_shape=None,
) -> np.ndarray:
    """
    객체 index 기준으로 remove_mask 생성.
    UI에서 사용자가 특정 객체를 선택하는 경우에 사용하기 좋다.
    """
    if not instances:
        if image_shape is None:
            return None
        h, w = image_shape[:2]
        return np.zeros((h, w), dtype=np.uint8)

    remove_mask = np.zeros_like(instances[0]["mask"], dtype=np.uint8)

    for idx in indices:
        if 0 <= idx < len(instances):
            remove_mask = cv2.bitwise_or(remove_mask, instances[idx]["mask"])

    return remove_mask


def overlay_mask(
    image_bgr: np.ndarray,
    mask: Optional[np.ndarray],
    alpha: float = 0.45,
    color=(255, 0, 0),
) -> np.ndarray:
    """
    mask를 원본 이미지 위에 덮어 확인용 이미지 생성.
    기본 color는 BGR 기준 파란색.
    """
    vis = image_bgr.copy()

    if mask is None:
        return vis

    mask = (mask > 0).astype(np.uint8) * 255
    mask_bool = mask > 0

    colored = np.zeros_like(image_bgr)
    colored[:, :] = color

    if np.count_nonzero(mask_bool) == 0:
        return vis

    blended = cv2.addWeighted(image_bgr, 1 - alpha, colored, alpha, 0)
    vis[mask_bool] = blended[mask_bool]

    return vis


def save_instance_masks(instances: List[Dict], out_dir):
    """객체별 mask를 파일로 저장한다."""
    out_dir.mkdir(parents=True, exist_ok=True)

    for i, inst in enumerate(instances):
        label = str(inst["label"]).replace("/", "_").replace(" ", "_")
        path = out_dir / f"{i:02d}_{label}.png"
        cv2.imwrite(str(path), inst["mask"])