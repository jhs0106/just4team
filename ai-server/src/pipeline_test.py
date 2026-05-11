from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

from dino_detect import draw_detections, run_grounding_dino
from lama_inpaint import inpaint_with_lama_placeholder
from mask_utils import make_full_image_desk_mask, postprocess_occupied_mask
from sam2_segment import overlay_mask, segment_with_sam2_from_boxes
from space_analysis import analyze_space, metrics_to_dict, placement_candidates_placeholder


def load_and_optionally_resize(path: str, max_long_side: int | None) -> np.ndarray:
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"이미지 로드 실패: {path}")

    if max_long_side and max_long_side > 0:
        h, w = img.shape[:2]
        long_side = max(h, w)

        if long_side > max_long_side:
            scale = max_long_side / long_side
            img = cv2.resize(
                img,
                (int(w * scale), int(h * scale)),
                interpolation=cv2.INTER_AREA,
            )

    return img


def save(path: Path, img: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img)


def compose_pipeline_summary(paths: list[Path], out_path: Path) -> None:
    titles = [
        "front",
        "top",
        "top_detection",
        "top_mask",
        "front_mask",
        "available",
    ]

    imgs = [
        cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)
        if p.exists()
        else np.zeros((300, 300, 3), np.uint8)
        for p in paths
    ]

    plt.figure(figsize=(24, 5), dpi=200)

    for i, (title, img) in enumerate(zip(titles, imgs), 1):
        plt.subplot(1, 6, i)
        plt.imshow(img)
        plt.title(title)
        plt.axis("off")

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close()


def print_gpu_info() -> None:
    print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")


def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument("--front-image", required=True)
    ap.add_argument("--top-view-image", required=True)

    ap.add_argument("--desk-width-cm", type=float, required=True)
    ap.add_argument("--desk-depth-cm", type=float, required=True)

    ap.add_argument("--prompt", type=str, required=True)

    ap.add_argument(
        "--mode",
        choices=["empty_desk", "own_desk", "empty_space"],
        default="own_desk",
    )

    ap.add_argument("--outputs-dir", default="outputs")
    ap.add_argument("--max-long-side", type=int, default=1280)

    ap.add_argument("--dino-config", default=None)
    ap.add_argument("--dino-weights", default=None)

    ap.add_argument("--sam2-config", default=None)
    ap.add_argument("--sam2-checkpoint", default=None)

    ap.add_argument("--lama-model-dir", default=None)

    args = ap.parse_args()

    out = Path(args.outputs_dir)

    print_gpu_info()

    front = load_and_optionally_resize(
        args.front_image,
        args.max_long_side,
    )

    top = load_and_optionally_resize(
        args.top_view_image,
        args.max_long_side,
    )

    save(out / "front_input_copy.png", front)
    save(out / "top_view_input_copy.png", top)

    if args.mode == "empty_space":
        print("empty_space mode: 이미지 분석 생략")
        return

    # =========================================================
    # Top-view : space analysis
    # =========================================================

    top_detections = run_grounding_dino(
        top,
        args.prompt,
        config_path=args.dino_config,
        weights_path=args.dino_weights,
    )

    top_detection_vis = draw_detections(top, top_detections)

    save(out / "detection_result.png", top_detection_vis)
    save(out / "top_detection_result.png", top_detection_vis)

    top_raw_occ = segment_with_sam2_from_boxes(
        top,
        [d.box_xyxy for d in top_detections],
        sam2_checkpoint=args.sam2_checkpoint,
        sam2_config=args.sam2_config,
    )

    save(out / "raw_occupied_mask.png", top_raw_occ)
    save(out / "top_raw_occupied_mask.png", top_raw_occ)

    top_processed_occ = postprocess_occupied_mask(top_raw_occ)

    save(out / "processed_occupied_mask.png", top_processed_occ)
    save(out / "top_processed_occupied_mask.png", top_processed_occ)

    top_overlay = overlay_mask(
        top,
        top_processed_occ,
        color=(255, 50, 50),
        alpha=0.45,
    )

    save(out / "sam2_mask_overlay.png", top_overlay)
    save(out / "top_sam2_mask_overlay.png", top_overlay)

    # =========================================================
    # Front-view : synthesis helper
    # =========================================================

    front_detections = run_grounding_dino(
        front,
        args.prompt,
        config_path=args.dino_config,
        weights_path=args.dino_weights,
    )

    front_detection_vis = draw_detections(front, front_detections)

    save(out / "front_detection_result.png", front_detection_vis)

    front_raw_mask = segment_with_sam2_from_boxes(
        front,
        [d.box_xyxy for d in front_detections],
        sam2_checkpoint=args.sam2_checkpoint,
        sam2_config=args.sam2_config,
    )

    save(out / "front_raw_object_mask.png", front_raw_mask)

    front_processed_mask = postprocess_occupied_mask(front_raw_mask)

    save(out / "front_processed_object_mask.png", front_processed_mask)

    front_overlay = overlay_mask(
        front,
        front_processed_mask,
        color=(255, 50, 50),
        alpha=0.45,
    )

    save(out / "front_sam2_mask_overlay.png", front_overlay)

    # =========================================================
    # Space analysis (TOP ONLY)
    # =========================================================

    desk_mask = make_full_image_desk_mask(top.shape)

    save(out / "desk_mask.png", desk_mask)

    metrics, available_mask = analyze_space(
        desk_mask,
        top_processed_occ,
        args.desk_width_cm,
        args.desk_depth_cm,
    )

    save(out / "available_space_mask.png", available_mask)

    available_overlay = overlay_mask(
        top,
        available_mask,
        color=(50, 220, 50),
        alpha=0.45,
    )

    save(out / "available_space_overlay.png", available_overlay)

    summary = metrics_to_dict(metrics)

    summary["mode"] = args.mode

    summary["placement_candidate"] = placement_candidates_placeholder(
        available_mask,
        10,
        10,
    )

    (
        out / "space_analysis_summary.json"
    ).write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    summary_canvas = np.full((460, 860, 3), 255, dtype=np.uint8)

    y = 45

    for k, v in summary.items():
        txt = f"{k}: {v}"

        cv2.putText(
            summary_canvas,
            txt[:100],
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 0),
            2,
        )

        y += 35

    save(out / "space_analysis_summary.png", summary_canvas)

    cleaned_path = out / "cleaned_result.png"

    if args.mode == "empty_desk":
        cleaned = inpaint_with_lama_placeholder(
            top,
            top_processed_occ,
            args.lama_model_dir,
        )

        save(cleaned_path, cleaned)

    else:
        save(cleaned_path, top)

    compose_pipeline_summary(
        [
            out / "front_input_copy.png",
            out / "top_view_input_copy.png",
            out / "top_detection_result.png",
            out / "top_sam2_mask_overlay.png",
            out / "front_sam2_mask_overlay.png",
            out / "available_space_overlay.png",
        ],
        out / "pipeline_summary.png",
    )

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("완료:", out.resolve())


if __name__ == "__main__":
    main()