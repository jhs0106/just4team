import argparse
import json
from pathlib import Path
from typing import Optional, Tuple

import os
import cv2
import numpy as np

from dino_detect import box_area, draw_detections, run_grounding_dino
from lama_inpaint import run_lama_inpaint, save_lama_inputs
from sam2_segment import (
    build_remove_mask_by_indices,
    build_remove_mask_from_labels,
    merge_instance_masks,
    overlay_mask,
    postprocess_instances,
    postprocess_mask,
    save_instance_masks,
    segment_instances_with_sam2,
)
from space_analysis import (
    analyze_space,
    build_available_mask,
    draw_available_overlay,
    keep_largest_component,
    make_full_desk_mask,
    make_rect_desk_mask,
)


DEFAULT_OBJECT_PROMPT = (
    "monitor. keyboard. mouse. mouse pad. cup. mug. "
    "book. notebook. speaker. lamp. cable."
)

DEFAULT_DESK_PROMPT = (
    "desk surface. tabletop. desk top. table surface. desk. table."
)


def save(path: Path, image: np.ndarray):
    """이미지 저장."""
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), image)
    print(f"[SAVE] {path}")

def fill_desk_surface_mask(mask: np.ndarray) -> np.ndarray:
    """
    SAM2가 책상 위 객체가 올라간 부분을 desk_mask에서 제외하는 경우가 있어,
    책상 상판 외곽선을 기준으로 내부를 채워 desk_mask를 보정한다.
    """
    mask_u8 = (mask > 0).astype(np.uint8) * 255

    contours, _ = cv2.findContours(
        mask_u8,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    if not contours:
        return mask_u8

    largest_contour = max(contours, key=cv2.contourArea)

    filled = np.zeros_like(mask_u8)
    hull = cv2.convexHull(largest_contour)
    cv2.fillConvexPoly(filled, hull, 255)

    return filled

def make_box_mask(image_shape, box_xyxy) -> np.ndarray:
    """
    DINO가 검출한 bbox 영역을 mask로 만든다.
    desk_mask가 convex hull 보정 후 과하게 확장되는 것을 막기 위해 사용한다.
    """
    h, w = image_shape[:2]
    x1, y1, x2, y2 = map(int, box_xyxy)

    x1 = max(0, min(x1, w - 1))
    x2 = max(0, min(x2, w))
    y1 = max(0, min(y1, h - 1))
    y2 = max(0, min(y2, h))

    box_mask = np.zeros((h, w), dtype=np.uint8)
    box_mask[y1:y2, x1:x2] = 255

    return box_mask

def clip_overextended_bottom(mask: np.ndarray, box_xyxy, max_bottom_ratio: float = 0.90) -> np.ndarray:
    """
    desk_mask가 DINO bbox 하단까지 과하게 확장된 경우에만 하단부를 제한한다.
    고정 좌표가 아니라 bbox 내 mask 하단 비율을 기준으로 판단한다.
    """
    x1, y1, x2, y2 = map(int, box_xyxy)
    bbox_h = max(y2 - y1, 1)

    mask_bool = mask > 0
    ys = np.where(mask_bool)[0]

    if len(ys) == 0:
        return mask

    mask_bottom = int(ys.max())
    bottom_ratio = (mask_bottom - y1) / bbox_h

    if bottom_ratio > max_bottom_ratio:
        surface_y2 = int(y1 + bbox_h * max_bottom_ratio)

        clipped = mask.copy()
        clipped[surface_y2:, :] = 0

        # 하단 과확장 제거 후 남은 상판 외곽 기준으로 다시 내부를 채움
        clipped = fill_desk_surface_mask(clipped)

        # fill 과정에서 bbox 밖 또는 하단 컷 아래로 다시 퍼지는 것을 방지
        box_mask = make_box_mask(mask.shape, box_xyxy)
        clipped = cv2.bitwise_and(clipped, box_mask)
        clipped[surface_y2:, :] = 0

        print(
            f"[DESK] auto bottom clip applied: "
            f"bottom_ratio={bottom_ratio:.3f}, y<{surface_y2}"
        )
        return clipped

    print(f"[DESK] auto bottom clip skipped: bottom_ratio={bottom_ratio:.3f}")
    return mask

def parse_roi(roi_text: Optional[str]) -> Optional[Tuple[int, int, int, int]]:
    """x1,y1,x2,y2 문자열을 tuple로 변환."""
    if not roi_text:
        return None

    parts = [p.strip() for p in roi_text.split(",")]

    if len(parts) != 4:
        raise ValueError("--desk-roi는 x1,y1,x2,y2 형식이어야 합니다.")

    return tuple(map(int, parts))


def parse_csv_list(text: Optional[str]):
    """cup,book 같은 문자열을 리스트로 변환."""
    if not text:
        return []

    return [x.strip() for x in text.split(",") if x.strip()]


def parse_index_list(text: Optional[str]):
    """0,2,3 같은 문자열을 정수 리스트로 변환."""
    if not text:
        return []

    return [int(x.strip()) for x in text.split(",") if x.strip()]


def instance_summary(instances):
    """JSON 저장용 instance 요약. mask 배열은 제외한다."""
    rows = []

    for i, inst in enumerate(instances):
        rows.append(
            {
                "index": i,
                "label": inst["label"],
                "score": round(float(inst["score"]), 4),
                "box_xyxy": inst["box_xyxy"],
                "mask_area_px": int(np.count_nonzero(inst["mask"])),
            }
        )

    return rows


def run_dino_sam2_for_view(
    image: np.ndarray,
    prompt: str,
    view_name: str,
    out_dir: Path,
    sam2_checkpoint: str,
    sam2_config: str,
    box_threshold: float,
    text_threshold: float,
    nms_iou_threshold: float,
    max_box_area_ratio: float,
):
    """
    일반 객체 검출용 DINO + SAM2 실행.
    top-view는 occupied_mask 생성에 사용되고,
    front-view는 LaMa remove_mask 생성에 사용된다.
    """
    print(f"\n[{view_name.upper()}] Object DINO start")

    detections = run_grounding_dino(
        image_bgr=image,
        prompt=prompt,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
        nms_iou_threshold=nms_iou_threshold,
        max_box_area_ratio=max_box_area_ratio,
        mode="object",
    )

    save(out_dir / f"{view_name}_detection_result.png", draw_detections(image, detections))

    print(f"[{view_name.upper()}] Object SAM2 start")

    sam2_result = segment_instances_with_sam2(
        image_bgr=image,
        detections=detections,
        sam2_checkpoint=sam2_checkpoint,
        sam2_config=sam2_config,
    )

    raw_merged_mask = sam2_result["merged_mask"]
    raw_instances = sam2_result["instances"]

    processed_instances = postprocess_instances(
        raw_instances,
        dilate_kernel=7,
        dilate_iter=1,
        close_kernel=5,
    )

    processed_merged_mask = merge_instance_masks(
        processed_instances,
        image_shape=image.shape,
    )

    if processed_merged_mask is None:
        processed_merged_mask = postprocess_mask(raw_merged_mask)

    if processed_merged_mask is None:
        processed_merged_mask = np.zeros(image.shape[:2], dtype=np.uint8)

    save(out_dir / f"{view_name}_raw_occupied_mask.png", raw_merged_mask)
    save(out_dir / f"{view_name}_processed_occupied_mask.png", processed_merged_mask)
    save(out_dir / f"{view_name}_sam2_mask_overlay.png", overlay_mask(image, processed_merged_mask))

    save_instance_masks(processed_instances, out_dir / f"{view_name}_instance_masks")

    return {
        "detections": detections,
        "instances": processed_instances,
        "raw_merged_mask": raw_merged_mask,
        "processed_merged_mask": processed_merged_mask,
    }

def build_auto_desk_mask(
    top_image: np.ndarray,
    desk_prompt: str,
    out_dir: Path,
    sam2_checkpoint: str,
    sam2_config: str,
    box_threshold: float,
    text_threshold: float,
    occupied_mask: Optional[np.ndarray] = None,
):
    """
    DINO + SAM2로 책상 상판 desk_mask를 자동 생성한다.

    핵심:
    DINO가 책상/상판 bbox 검출
    → 가장 큰 desk 후보를 SAM2에 입력
    → desk_mask 생성
    """
    print("\n[DESK] Auto desk mask start")

    desk_detections = run_grounding_dino(
        image_bgr=top_image,
        prompt=desk_prompt,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
        nms_iou_threshold=0.50,
        max_box_area_ratio=0.98,
        mode="desk",
    )

    save(out_dir / "desk_detection_result.png", draw_detections(top_image, desk_detections))

    if not desk_detections:
        print("[WARN] 책상 상판 자동 검출 실패")
        return None, [], "auto_failed"

    # 책상 후보 중 가장 큰 bbox를 사용한다.
    desk_detections = sorted(
        desk_detections,
        key=lambda d: box_area(d.box_xyxy),
        reverse=True,
    )

    best_desk = desk_detections[0]

    print(f"[DESK] selected: {best_desk.label}, score={best_desk.score:.3f}, box={best_desk.box_xyxy}")

    desk_sam2_result = segment_instances_with_sam2(
        image_bgr=top_image,
        detections=[best_desk],
        sam2_checkpoint=sam2_checkpoint,
        sam2_config=sam2_config,
    )

    desk_mask = desk_sam2_result["merged_mask"]

    desk_mask = postprocess_mask(
        desk_mask,
        dilate_kernel=3,
        dilate_iter=1,
        close_kernel=21,
    )

    if desk_mask is None or np.count_nonzero(desk_mask) == 0:
        print("[WARN] SAM2 책상 mask 생성 실패")
        return None, desk_detections, "auto_sam2_failed"

    desk_mask = keep_largest_component(desk_mask)

    # 책/노트북/마우스 등 책상 위 객체가 SAM2 desk_mask에서 제외될 수 있으므로,
    # desk_mask 복원 단계에서는 occupied_mask도 책상 상판 seed로 함께 사용한다.
    if occupied_mask is not None:
        occupied_mask_u8 = (occupied_mask > 0).astype(np.uint8) * 255

        # desk bbox 내부의 객체만 사용한다.
        desk_box_mask = make_box_mask(top_image.shape, best_desk.box_xyxy)
        occupied_inside_desk_box = cv2.bitwise_and(occupied_mask_u8, desk_box_mask)

        before_seed_merge = int(np.count_nonzero(desk_mask))
        desk_mask = cv2.bitwise_or(desk_mask, occupied_inside_desk_box)
        after_seed_merge = int(np.count_nonzero(desk_mask))

        print(f"[DESK] occupied seed merge applied: {before_seed_merge} -> {after_seed_merge}")

    before_fill = int(np.count_nonzero(desk_mask))
    desk_mask = fill_desk_surface_mask(desk_mask)
    after_fill = int(np.count_nonzero(desk_mask))

    print(f"[DESK] fill_desk_surface_mask applied: {before_fill} -> {after_fill}")

    # convex hull 보정 후 desk_mask가 책상 바깥으로 과하게 확장되는 것을 방지하기 위해
    # DINO가 검출한 desk bbox 내부로 제한한다.
    desk_box_mask = make_box_mask(top_image.shape, best_desk.box_xyxy)

    before_box_clip = int(np.count_nonzero(desk_mask))
    desk_mask = cv2.bitwise_and(desk_mask, desk_box_mask)
    after_box_clip = int(np.count_nonzero(desk_mask))

    print(f"[DESK] bbox clip applied: {before_box_clip} -> {after_box_clip}")

    # DINO bbox가 책상 상판뿐 아니라 책상 전면부/아래쪽까지 포함하는 경우,
    # mask가 bbox 하단까지 과하게 확장되었을 때만 조건부로 하단을 제한한다.
    before_bottom_clip = int(np.count_nonzero(desk_mask))
    desk_mask = clip_overextended_bottom(
        desk_mask,
        best_desk.box_xyxy,
        max_bottom_ratio=0.90,
    )
    after_bottom_clip = int(np.count_nonzero(desk_mask))

    print(f"[DESK] bottom clip result: {before_bottom_clip} -> {after_bottom_clip}")

    save(out_dir / "desk_mask.png", desk_mask)
    save(out_dir / "desk_mask_overlay.png", overlay_mask(top_image, desk_mask, color=(0, 255, 255)))

    return desk_mask, desk_detections, "auto_dino_sam2"


def build_remove_mask_for_mode(
    mode: str,
    instances,
    occupied_mask: np.ndarray,
    remove_labels,
    remove_indices,
    image_shape,
):
    """
    mode에 따라 remove_mask 생성.

    own_desk/add:
        기존 객체를 유지하므로 remove_mask 없음.

    replace:
        선택된 객체만 제거.

    empty_desk:
        전체 occupied_mask 제거.
    """
    if mode in ["add", "own_desk"]:
        return None

    if mode == "empty_desk":
        return occupied_mask

    if mode == "replace":
        if remove_indices:
            return build_remove_mask_by_indices(
                instances,
                remove_indices,
                image_shape=image_shape,
            )

        if remove_labels:
            return build_remove_mask_from_labels(
                instances,
                remove_labels,
                image_shape=image_shape,
            )

        print("[WARN] replace 모드인데 remove 대상이 없습니다.")
        return np.zeros(image_shape[:2], dtype=np.uint8)

    return None


def main():
    parser = argparse.ArgumentParser(
        description="DINO + SAM2 + 자동 desk_mask + 가용공간 분석 + LaMa 통합 파이프라인"
    )

    parser.add_argument("--front-image", required=True, help="front-view 이미지. 최종 합성/LaMa 기준")
    parser.add_argument("--top-view-image", required=True, help="top-view 이미지. 가용공간 분석 기준")

    parser.add_argument("--desk-width-cm", type=float, required=True, help="실제 책상 가로 길이(cm)")
    parser.add_argument("--desk-depth-cm", type=float, required=True, help="실제 책상 세로 길이(cm)")

    parser.add_argument(
        "--prompt",
        default=DEFAULT_OBJECT_PROMPT,
        help="책상 위 객체 검출 prompt",
    )

    parser.add_argument(
        "--desk-prompt",
        default=DEFAULT_DESK_PROMPT,
        help="책상 상판 자동 검출 prompt",
    )

    parser.add_argument(
        "--auto-desk-mask",
        action="store_true",
        help="DINO+SAM2로 책상 상판 desk_mask를 자동 생성한다.",
    )

    parser.add_argument(
        "--desk-roi",
        default=None,
        help="자동 desk mask 실패 시 사용할 수동 ROI. 형식: x1,y1,x2,y2",
    )

    parser.add_argument(
        "--mode",
        default="own_desk",
        choices=["add", "own_desk", "replace", "empty_desk"],
        help="own_desk/add: 유지, replace: 선택 객체 제거, empty_desk: 전체 제거",
    )

    parser.add_argument(
        "--remove-labels",
        default=None,
        help="replace 모드에서 제거할 label. 예: cup,book",
    )

    parser.add_argument(
        "--remove-indices",
        default=None,
        help="replace 모드에서 제거할 instance index. 예: 0,2",
    )

    parser.add_argument("--sam2-config", required=True, help="SAM2 config. 예: configs/sam2.1/sam2.1_hiera_t.yaml")
    parser.add_argument("--sam2-checkpoint", required=True, help="SAM2 checkpoint 경로")

    parser.add_argument("--box-threshold", type=float, default=0.25)
    parser.add_argument("--text-threshold", type=float, default=0.20)
    parser.add_argument("--desk-box-threshold", type=float, default=0.20)
    parser.add_argument("--desk-text-threshold", type=float, default=0.15)

    parser.add_argument("--nms-iou-threshold", type=float, default=0.50)
    parser.add_argument("--max-box-area-ratio", type=float, default=0.60)

    parser.add_argument(
        "--min-region-area-px",
        type=int,
        default=1000,
        help="너무 작은 available region 제거 기준",
    )

    parser.add_argument(
        "--disable-lama",
        action="store_true",
        help="LaMa 실행 비활성화",
    )

    parser.add_argument(
        "--no-lama-fallback",
        action="store_true",
        help="LaMa 실패 시 OpenCV fallback 사용 안 함",
    )

    parser.add_argument("--outputs-dir", default="outputs_auto_space")

    args = parser.parse_args()

    out_dir = Path(args.outputs_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    front = cv2.imread(args.front_image)
    top = cv2.imread(args.top_view_image)

    if front is None:
        raise FileNotFoundError(f"front image를 읽을 수 없습니다: {args.front_image}")

    if top is None:
        raise FileNotFoundError(f"top-view image를 읽을 수 없습니다: {args.top_view_image}")

    save(out_dir / "front_input_copy.png", front)
    save(out_dir / "top_view_input_copy.png", top)

    # ------------------------------------------------------------
    # 1. top-view 객체 검출 및 occupied_mask 생성
    # ------------------------------------------------------------
    top_result = run_dino_sam2_for_view(
        image=top,
        prompt=args.prompt,
        view_name="top",
        out_dir=out_dir,
        sam2_checkpoint=args.sam2_checkpoint,
        sam2_config=args.sam2_config,
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold,
        nms_iou_threshold=args.nms_iou_threshold,
        max_box_area_ratio=args.max_box_area_ratio,
    )

    # 기존 출력 파일명 호환
    save(out_dir / "detection_result.png", draw_detections(top, top_result["detections"]))
    save(out_dir / "raw_occupied_mask.png", top_result["raw_merged_mask"])
    save(out_dir / "processed_occupied_mask.png", top_result["processed_merged_mask"])
    save(out_dir / "sam2_mask_overlay.png", overlay_mask(top, top_result["processed_merged_mask"]))

    # ------------------------------------------------------------
    # 2. front-view 객체 검출 및 mask 생성
    # LaMa 객체 제거용으로 사용
    # ------------------------------------------------------------
    front_result = run_dino_sam2_for_view(
        image=front,
        prompt=args.prompt,
        view_name="front",
        out_dir=out_dir,
        sam2_checkpoint=args.sam2_checkpoint,
        sam2_config=args.sam2_config,
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold,
        nms_iou_threshold=args.nms_iou_threshold,
        max_box_area_ratio=args.max_box_area_ratio,
    )

    # ------------------------------------------------------------
    # 3. desk_mask 생성
    # 최우선: 자동 desk_mask
    # fallback 1: desk-roi
    # fallback 2: full image
    # ------------------------------------------------------------
    desk_roi = parse_roi(args.desk_roi)

    desk_mask = None
    desk_detections = []
    desk_mask_mode = None

    if args.auto_desk_mask:
        desk_mask, desk_detections, desk_mask_mode = build_auto_desk_mask(
            top_image=top,
            desk_prompt=args.desk_prompt,
            out_dir=out_dir,
            sam2_checkpoint=args.sam2_checkpoint,
            sam2_config=args.sam2_config,
            box_threshold=args.desk_box_threshold,
            text_threshold=args.desk_text_threshold,
            occupied_mask=top_result["processed_merged_mask"],
        )

    if desk_mask is None and desk_roi is not None:
        print("[DESK] rect ROI fallback 사용")
        desk_mask = make_rect_desk_mask(top.shape, desk_roi)
        desk_mask_mode = "rect_roi"
        save(out_dir / "desk_mask.png", desk_mask)
        save(out_dir / "desk_mask_overlay.png", overlay_mask(top, desk_mask, color=(0, 255, 255)))

    if desk_mask is None:
        print("[DESK] full image fallback 사용")
        desk_mask = make_full_desk_mask(top.shape)
        desk_mask_mode = "full_image"
        save(out_dir / "desk_mask.png", desk_mask)
        save(out_dir / "desk_mask_overlay.png", overlay_mask(top, desk_mask, color=(0, 255, 255)))

    # ------------------------------------------------------------
    # 4. remove_mask 생성
    # replace / empty_desk에서만 사용
    # ------------------------------------------------------------
    remove_labels = parse_csv_list(args.remove_labels)
    remove_indices = parse_index_list(args.remove_indices)

    top_remove_mask = build_remove_mask_for_mode(
        mode=args.mode,
        instances=top_result["instances"],
        occupied_mask=top_result["processed_merged_mask"],
        remove_labels=remove_labels,
        remove_indices=remove_indices,
        image_shape=top.shape,
    )

    front_remove_mask = build_remove_mask_for_mode(
        mode=args.mode,
        instances=front_result["instances"],
        occupied_mask=front_result["processed_merged_mask"],
        remove_labels=remove_labels,
        remove_indices=remove_indices,
        image_shape=front.shape,
    )

    if top_remove_mask is not None:
        save(out_dir / "top_remove_mask.png", top_remove_mask)
        save(out_dir / "top_remove_mask_overlay.png", overlay_mask(top, top_remove_mask, color=(0, 0, 255)))

    if front_remove_mask is not None:
        save(out_dir / "front_remove_mask.png", front_remove_mask)
        save(out_dir / "front_remove_mask_overlay.png", overlay_mask(front, front_remove_mask, color=(0, 0, 255)))

    # ------------------------------------------------------------
    # 5. 가용공간 분석
    # 핵심:
    # available = desk_mask - occupied_mask
    # replace에서는 remove_mask를 제외한 keep_mask 기준
    # ------------------------------------------------------------
    available_mask = build_available_mask(
        occupied_mask=top_result["processed_merged_mask"],
        desk_mask=desk_mask,
        remove_mask=top_remove_mask,
        min_region_area_px=args.min_region_area_px,
    )

    save(out_dir / "available_space_mask.png", available_mask)

    available_overlay = draw_available_overlay(
        image_bgr=top,
        available_mask=available_mask,
        desk_mask=desk_mask,
    )

    save(out_dir / "available_space_overlay.png", available_overlay)

    space_summary = analyze_space(
        occupied_mask=top_result["processed_merged_mask"],
        desk_width_cm=args.desk_width_cm,
        desk_depth_cm=args.desk_depth_cm,
        desk_mask=desk_mask,
        remove_mask=top_remove_mask,
        min_region_area_px=args.min_region_area_px,
    )

    # ------------------------------------------------------------
    # 6. LaMa 객체 제거
    # own_desk/add에서는 보통 실행되지 않는다.
    # replace/empty_desk에서 front_remove_mask가 있을 때 실행한다.
    # ------------------------------------------------------------
    lama_result_path = None

    if not args.disable_lama and front_remove_mask is not None and np.count_nonzero(front_remove_mask) > 0:
        lama_dir = out_dir / "lama_debug"
        save_lama_inputs(lama_dir, front, front_remove_mask)

        cleaned_front = run_lama_inpaint(
            image_bgr=front,
            remove_mask=front_remove_mask,
            use_fallback=not args.no_lama_fallback,
        )

        lama_result_path = "front_lama_cleaned.png"
        save(out_dir / lama_result_path, cleaned_front)

    elif args.mode in ["replace", "empty_desk"]:
        print("[WARN] LaMa 실행 대상 remove_mask가 없어 인페인팅을 건너뜁니다.")
    else:
        print("[INFO] own_desk/add 모드이므로 LaMa를 실행하지 않습니다.")

    # ------------------------------------------------------------
    # 7. summary JSON 저장
    # 추천/합성 파트로 넘길 수 있는 결과
    # ------------------------------------------------------------
    summary = {
        "mode": args.mode,
        "object_prompt": args.prompt,
        "desk_prompt": args.desk_prompt,
        "desk_mask_mode": desk_mask_mode,
        "desk_roi": desk_roi,
        "thresholds": {
            "box_threshold": args.box_threshold,
            "text_threshold": args.text_threshold,
            "desk_box_threshold": args.desk_box_threshold,
            "desk_text_threshold": args.desk_text_threshold,
            "nms_iou_threshold": args.nms_iou_threshold,
            "max_box_area_ratio": args.max_box_area_ratio,
        },
        "remove": {
            "remove_labels": remove_labels,
            "remove_indices": remove_indices,
            "has_top_remove_mask": top_remove_mask is not None and np.count_nonzero(top_remove_mask) > 0,
            "has_front_remove_mask": front_remove_mask is not None and np.count_nonzero(front_remove_mask) > 0,
        },
        "desk_detections": [
            {
                "label": d.label,
                "score": round(float(d.score), 4),
                "box_xyxy": d.box_xyxy,
                "area_px": box_area(d.box_xyxy),
            }
            for d in desk_detections
        ],
        "top_detections": instance_summary(top_result["instances"]),
        "front_detections": instance_summary(front_result["instances"]),
        "space_analysis": space_summary,
        "outputs": {
            "desk_detection_result": "desk_detection_result.png" if args.auto_desk_mask else None,
            "desk_mask": "desk_mask.png",
            "desk_mask_overlay": "desk_mask_overlay.png",
            "occupied_mask": "processed_occupied_mask.png",
            "available_space_mask": "available_space_mask.png",
            "available_space_overlay": "available_space_overlay.png",
            "front_lama_cleaned": lama_result_path,
        },
    }

    summary_path = out_dir / "space_analysis_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"[SAVE] {summary_path}")

    # ------------------------------------------------------------
    # 8. 요약 이미지 저장
    # 원본 / occupied / desk / available 순서
    # ------------------------------------------------------------

    summary_top = cv2.imread(str(out_dir / "top_view_input_copy.png"))
    summary_occ = cv2.imread(str(out_dir / "sam2_mask_overlay.png"))
    summary_desk = cv2.imread(str(out_dir / "desk_mask_overlay.png"))
    summary_avail = cv2.imread(str(out_dir / "available_space_overlay.png"))

    summary_items = [
        ("top_view_input_copy.png", summary_top),
        ("sam2_mask_overlay.png", summary_occ),
        ("desk_mask_overlay.png", summary_desk),
        ("available_space_overlay.png", summary_avail),
    ]

    for name, img in summary_items:
        if img is None:
            raise FileNotFoundError(f"summary image를 읽을 수 없습니다: {out_dir / name}")

        print(f"[SUMMARY] {name}: {img.shape}")

    summary_img = np.hstack(
        [
            cv2.resize(summary_top, (360, 240)),
            cv2.resize(summary_occ, (360, 240)),
            cv2.resize(summary_desk, (360, 240)),
            cv2.resize(summary_avail, (360, 240)),
        ]
    )

    save(out_dir / "pipeline_summary.png", summary_img)


if __name__ == "__main__":
    main()