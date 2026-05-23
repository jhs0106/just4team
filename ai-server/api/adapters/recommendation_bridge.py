# Recommendation 시스템(feature/recommendation 브랜치)의 SetupRecommender 출력을
# AI 서버의 GenerateRequest로 변환하는 어댑터.
#
# 통합 결정사항(2026-05-24):
#   - 옵션 A: recommendation이 사전에 배경 제거된 PNG를 생성/저장. AI 서버는 image_id로 참조
#   - 옵션 X: 단일 Python 서버에 통합 (FastAPI 안에서 import)
#   - MONITOR_ARM, DESK 카테고리: skip
#   - LIGHTING: AI 서버에 정식 추가됨 (config.py 참조)

from pathlib import Path

from ..models import GenerateRequest, ProductItem, RemoveMode, StyleName
from ..utils import find_product_image
from .style_mapper import map_to_style


# recommendation 카테고리 → AI 서버 카테고리 매핑.
# None이면 AI 서버에서 처리하지 않음 (skip).
_CATEGORY_MAP: dict[str, str | None] = {
    "DESK":         None,            # 사용자 책상 사진 사용, 추천 제품 무시
    "MONITOR":      "MONITOR",
    "KEYBOARD":     "KEYBOARD",
    "MOUSE":        "MOUSE",
    "MONITOR_ARM":  None,            # front-view에서 안 보임, skip
    "LAPTOP_STAND": "LAPTOP_STAND",
    "MOUSEPAD":     "MOUSEPAD",
    "DESK_SHELF":   "DESK_SHELF",
    "LIGHTING":     "LIGHTING",
    "SPEAKER":      "SPEAKER",
    "CLOCK":        "CLOCK",
    "DESK_LAMP":    "DESK_LAMP",
    "DECO":         "DECO",
}


def _extract_size(item: dict) -> tuple[int | None, int | None]:
    # recommendation DB의 metadata JSONB에 width_mm/depth_mm가 있으면 추출,
    # 없으면 (None, None) — AI 서버의 enrich_products_from_csv가 카탈로그 기본값으로 폴백
    metadata = item.get("metadata") or {}
    if isinstance(metadata, str):
        import json
        try:
            metadata = json.loads(metadata)
        except Exception:
            metadata = {}
    w = metadata.get("width_mm")
    d = metadata.get("depth_mm")
    return (int(w) if w else None, int(d) if d else None)


def setup_to_generate_request(
    setup:                dict,
    color_text:           str           = "",
    theme_text:           str           = "",
    desk_image_b64:       str           = "",
    desk_width_mm:        int   | None  = None,
    desk_depth_mm:        int   | None  = None,
    top_view_image_b64:   str   | None  = None,
    mode:                 RemoveMode    = RemoveMode.own_desk,
    generation_mode:      str           = "controlnet",
    removal_strategy:     str           = "combined",
    verify_images:        bool          = True,
    on_missing:           str           = "skip",
) -> GenerateRequest:
    # SetupRecommender.recommend_setup()의 setup dict 1개를 받아 GenerateRequest로 변환.
    #
    # setup 구조 (recommendation):
    #   {
    #     "setup_score": float, "total_price": int, "budget": int|None,
    #     "items": {
    #       "MONITOR":  {"id": 123, "title": "...", "lprice": 200000, "category": "MONITOR",
    #                    "brand": "...", "link": "...", "metadata": {"width_mm": ..., ...}},
    #       "KEYBOARD": {...},
    #       ...
    #     }
    #   }
    #
    # verify_images=True면 각 image_id가 AI 서버의 processed_images/{id}.png에 존재하는지 확인.
    # on_missing: "skip"(누락 제품 제외) | "raise"(예외 발생)
    items = setup.get("items") or {}
    if not items:
        raise ValueError("setup['items']가 비어있음")

    products: list[ProductItem] = []
    missing_ids: list[tuple[str, int]] = []

    for rec_cat, item in items.items():
        ai_cat = _CATEGORY_MAP.get(rec_cat)
        if ai_cat is None:
            continue

        product_id = item.get("id")
        if product_id is None:
            continue
        pid = int(product_id)

        # ID 존재 검증 — recommendation DB id와 AI 서버 파일명 일치 여부 확인
        if verify_images:
            prod_path = find_product_image(pid)
            if prod_path is None:
                msg = f"{ai_cat}(image_id={pid}): processed_images/{pid}.png 없음"
                if on_missing == "raise":
                    raise FileNotFoundError(msg)
                missing_ids.append((ai_cat, pid))
                continue

        width_mm, depth_mm = _extract_size(item)
        products.append(ProductItem(
            category=ai_cat,
            name=str(item.get("title") or ai_cat),
            image_id=pid,
            width_mm=width_mm,
            depth_mm=depth_mm,
        ))

    if missing_ids:
        print(f"[recommendation_bridge] 이미지 누락 제품 {len(missing_ids)}개 skip:")
        for cat, pid in missing_ids:
            print(f"  - {cat}: image_id={pid}")

    if not products:
        raise ValueError(
            "매핑된 제품이 없음 (모두 skip 카테고리 또는 이미지 누락). "
            f"missing_ids={missing_ids}"
        )

    style = map_to_style(color_text=color_text, theme_text=theme_text)

    return GenerateRequest(
        image_base64=desk_image_b64,
        style=style,
        products=products,
        desk_width_mm=desk_width_mm,
        desk_depth_mm=desk_depth_mm,
        top_view_image_base64=top_view_image_b64,
        mode=mode,
        generation_mode=generation_mode,
        removal_strategy=removal_strategy,
    )


def setups_to_generate_requests(
    setups:               list[dict],
    color_text:           str           = "",
    theme_text:           str           = "",
    desk_image_b64:       str           = "",
    desk_width_mm:        int   | None  = None,
    desk_depth_mm:        int   | None  = None,
    top_view_image_b64:   str   | None  = None,
    mode:                 RemoveMode    = RemoveMode.own_desk,
    generation_mode:      str           = "controlnet",
    removal_strategy:     str           = "combined",
    verify_images:        bool          = True,
    on_missing:           str           = "skip",
) -> list[GenerateRequest]:
    # 여러 setup(top-3 등)을 각각 GenerateRequest로 변환.
    return [
        setup_to_generate_request(
            setup=s,
            color_text=color_text,
            theme_text=theme_text,
            desk_image_b64=desk_image_b64,
            desk_width_mm=desk_width_mm,
            desk_depth_mm=desk_depth_mm,
            top_view_image_b64=top_view_image_b64,
            mode=mode,
            generation_mode=generation_mode,
            removal_strategy=removal_strategy,
            verify_images=verify_images,
            on_missing=on_missing,
        )
        for s in setups
    ]
