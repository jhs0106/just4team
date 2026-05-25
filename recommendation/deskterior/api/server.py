# FastAPI wrapper for recommendation engine.
# 실행: uvicorn deskterior.api.server:app --host 0.0.0.0 --port 8001
# 호출: GET http://localhost:8001/recommend?theme=white&budget=1000000

from fastapi import FastAPI, HTTPException, Query

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
from deskterior.retrieval.searcher import embed_text_query


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


@app.get("/recommend")
def recommend(
    theme:  str = Query(..., description="white | black | gaming | wood"),
    budget: int = Query(..., gt=0, description="예산 (원, 양수)"),
) -> dict:
    # 테마 + 예산 기반 TOP 1 셋업 추천 → setup dict 반환.
    if theme not in THEME_PRESETS:
        raise HTTPException(
            400,
            f"Invalid theme: {theme!r}. Must be one of {list(THEME_PRESETS.keys())}",
        )

    # 1. theme prompt embedding (사용 안 해도 모델 워밍업 효과)
    theme_prompt = THEME_PRESETS[theme]["theme_prompt"]
    _ = embed_text_query(theme_prompt)

    # 2. DB 연결 확인
    try:
        conn = get_db_connection()
        conn.close()
    except Exception as e:
        raise HTTPException(500, f"DB 연결 실패: {e}")

    # 3. 카테고리별 후보 검색
    all_categories = MANDATORY_CATEGORIES + OPTIONAL_CATEGORIES
    raw_by_category: dict[str, list[ScoredProduct]] = {}
    for category in all_categories:
        products = retrieve_candidates(theme, category)
        scored: list[ScoredProduct] = []
        for p in products:
            te = compute_theme_evidence(p, theme)
            scored.append(ScoredProduct(
                product=p, theme_evidence=te, value_score=0.0, item_score=0.0,
            ))
        raw_by_category[category] = scored

    # 4. ValueScore 정규화 + ItemScore
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

    # 5. Beam search로 TOP-K 추천
    bundles = recommend_setup(
        theme=theme, budget=budget,
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

    # 6. TOP 1 setup 반환 (config의 TOP_K_DEFAULT=1)
    return {
        "theme":  theme,
        "budget": budget,
        "setup":  _bundle_to_setup_dict(bundles[0]),
    }
