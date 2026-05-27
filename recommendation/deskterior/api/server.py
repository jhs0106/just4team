# FastAPI wrapper for recommendation engine.
# 실행: uvicorn deskterior.api.server:app --host 0.0.0.0 --port 8001
# 호출: POST http://localhost:8001/recommend  body={"theme","budget","image_base64?"}

import base64

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from deskterior.recommender.config import (
    THEME_PRESETS, MANDATORY_CATEGORIES, OPTIONAL_CATEGORIES,
    THEME_CATEGORY_QUERIES,
    CANDIDATE_LIMIT, BEAM_SIZE_DEFAULT, TOP_M_DEFAULT, TOP_K_DEFAULT,
)
from deskterior.recommender.engine import (
    ScoredProduct, Bundle,
    get_db_connection, retrieve_candidates, compute_theme_evidence,
    normalize_value_scores, compute_item_score, recommend_setup, clamp,
)
from deskterior.retrieval.searcher import embed_text_query, embed_image_query


class RecommendRequest(BaseModel):
    theme: str = Field(..., description="white | black | gaming | wood")
    budget: int = Field(..., gt=0, description="예산 (원, 양수)")
    image_base64: str | None = Field(
        None,
        description="사용자 책상 정면 사진 base64 (옵션). 주어지면 사진 임베딩을 카테고리 쿼리와 가중 평균해 검색.",
    )
    space_constraints: dict[str, list[int]] | None = Field(
        None,
        description="카테고리별 가용 공간 max (width_mm, depth_mm). ai-server가 top-view 분석 후 계산해 전달. 후보 검색 시 사이즈 필터로 사용.",
    )


app = FastAPI(title="Deskterior Recommendation API", version="0.1.0")


def _bundle_to_setup_dict(bundle: Bundle) -> dict:
    # ai-server/api/adapters/recommendation_bridge.py가 받는 형식과 호환되도록 변환.
    return {
        "setup_score": float(bundle.setup_score),
        "final_score": float(bundle.final_score),
        "total_price": int(bundle.total_price),
        "items": {
            sp.product.category: {
                "id":          int(sp.product.id) if sp.product.id else None,
                "title":       sp.product.name,
                "lprice":      sp.product.price,
                "category":    sp.product.category,
                "metadata":    sp.product.metadata or {},
                "image_url":   sp.product.image_url,
                "product_url": sp.product.product_url,
                "item_score":  float(sp.item_score),
            }
            for sp in bundle.items
        },
    }


@app.on_event("startup")
def _warmup():
    # 서버 시작 시 Jina CLIP v2 모델 미리 로드 (첫 요청 빠르게).
    print("[Recommendation API] 모델 워밍업 중...")
    try:
        _ = embed_text_query("warmup")
        print("[Recommendation API] 모델 로드 완료.")
    except Exception as e:
        print(f"[Recommendation API] 모델 워밍업 실패: {e}")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/themes")
def themes() -> dict:
    # 사용 가능한 테마 목록.
    return {
        "themes": [
            {"key": k, "label": v["label"], "description": v["description"]}
            for k, v in THEME_PRESETS.items()
        ]
    }


@app.post("/recommend")
def recommend(req: RecommendRequest) -> dict:
    # 테마 + 예산 + (옵션) 사용자 책상 정면 사진 기반 TOP 1 셋업 추천 → setup dict 반환.
    if req.theme not in THEME_PRESETS:
        raise HTTPException(
            400,
            f"Invalid theme: {req.theme!r}. Must be one of {list(THEME_PRESETS.keys())}",
        )

    # 1. theme prompt embedding (모델 워밍업)
    theme_prompt = THEME_PRESETS[req.theme]["theme_prompt"]
    _ = embed_text_query(theme_prompt)

    # 2. 사용자 사진 임베딩 (옵션)
    user_image_emb: list[float] | None = None
    if req.image_base64:
        _b64 = req.image_base64.strip()
        # 혹시 "data:image/...;base64,XXX" prefix 잔존 시 자동 제거
        if _b64.startswith("data:"):
            _comma = _b64.find(",")
            if _comma >= 0:
                _b64 = _b64[_comma + 1:]
        try:
            image_bytes = base64.b64decode(_b64, validate=False)
        except Exception as e:
            msg = (
                f"image_base64 디코딩 실패: {type(e).__name__}: {e} "
                f"(len={len(req.image_base64)}, head={req.image_base64[:40]!r})"
            )
            print(f"[/recommend ERROR] {msg}", flush=True)
            raise HTTPException(400, msg)
        try:
            user_image_emb = embed_image_query(image_bytes)
        except Exception as e:
            import traceback as _tb
            tb_str = _tb.format_exc()
            msg = (
                f"image_base64 임베딩 실패 (PIL/모델): {type(e).__name__}: {e} "
                f"(bytes_len={len(image_bytes)}, b64_head={req.image_base64[:40]!r})"
            )
            print(f"[/recommend ERROR] {msg}\n{tb_str}", flush=True)
            raise HTTPException(400, msg)

    # 3. DB 연결 확인
    try:
        conn = get_db_connection()
        conn.close()
    except Exception as e:
        raise HTTPException(500, f"DB 연결 실패: {e}")

    # 4. 카테고리별 후보 검색 — space_constraints가 있으면 size 필터 적용
    all_categories = MANDATORY_CATEGORIES + OPTIONAL_CATEGORIES
    raw_by_category: dict[str, list[ScoredProduct]] = {}
    for category in all_categories:
        _size_cap = (req.space_constraints or {}).get(category)
        products = retrieve_candidates(
            req.theme, category,
            user_image_embedding=user_image_emb,
            max_width_mm=_size_cap[0] if _size_cap else None,
            max_depth_mm=_size_cap[1] if _size_cap else None,
        )
        scored: list[ScoredProduct] = []
        for p in products:
            te = compute_theme_evidence(p, req.theme)
            scored.append(ScoredProduct(
                product=p, theme_evidence=te, value_score=0.0, item_score=0.0,
            ))
        raw_by_category[category] = scored

    # 5. ValueScore 정규화 + ItemScore
    normalize_value_scores(raw_by_category)
    for scored_list in raw_by_category.values():
        for sp in scored_list:
            sp.item_score = compute_item_score(
                image_sim      = clamp(sp.product.image_sim),
                text_sim       = clamp(sp.product.text_sim),
                theme_evidence = sp.theme_evidence,
                value_score    = sp.value_score,
            )
        scored_list.sort(key=lambda sp: sp.item_score, reverse=True)

    # 6. Beam search로 TOP-K 추천
    bundles = recommend_setup(
        theme=req.theme, budget=req.budget,
        candidates_by_category=raw_by_category,
        beam_size=BEAM_SIZE_DEFAULT,
        top_m=TOP_M_DEFAULT,
        top_k=TOP_K_DEFAULT,
    )

    if not bundles:
        raise HTTPException(
            404,
            "유효한 추천 결과 없음. 예산을 높이거나 다른 테마를 시도하세요.",
        )

    # 7. TOP 1 setup 반환 (config의 TOP_K_DEFAULT=1)
    return {
        "theme":  req.theme,
        "budget": req.budget,
        "image_used": user_image_emb is not None,
        "setup":  _bundle_to_setup_dict(bundles[0]),
    }
