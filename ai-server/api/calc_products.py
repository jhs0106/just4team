import pandas as pd
from scipy.optimize import milp, LinearConstraint, Bounds
import numpy as np
from pathlib import Path

_CSV_PATH = Path(__file__).parent.parent / "data" / "test" / "products.csv"
_catalog_df: pd.DataFrame | None = None


def _load_catalog() -> pd.DataFrame:
    global _catalog_df
    if _catalog_df is None:
        _catalog_df = pd.read_csv(_CSV_PATH)
    return _catalog_df


def calc_products(
    budget: int,
    style_scores: dict[int, float] | None = None,
    placeable_categories: list[str] | None = None,
) -> list[dict]:
    # style_scores: {image_id: clip_similarity_score} — Jina CLIP이 계산해서 넘겨주는 값
    # placeable_categories: AI 서버 /analyze 결과에서 받은 배치 가능한 카테고리 목록
    # 둘 다 None이면 CSV score 사용, 카테고리 필터 없음

    df = _load_catalog().copy()
    products = df.to_dict(orient="records")

    # 가용 공간 제약: 배치 불가 카테고리 제외
    if placeable_categories is not None:
        placeable_set = {c.upper() for c in placeable_categories}
        products = [p for p in products if str(p.get("category", "")).upper() in placeable_set]
        if not products:
            return []

    # 스타일 점수 주입: Jina CLIP 결과가 있으면 CSV score 덮어쓰기
    if style_scores:
        for p in products:
            pid = int(p.get("id", -1))
            if pid in style_scores:
                p["score"] = style_scores[pid]

    num_products = len(products)
    scores  = np.array([float(p.get("score", 0)) for p in products])
    prices  = np.array([float(p.get("price", 0)) for p in products])

    c          = -scores
    integrality = np.ones(num_products)
    bounds     = Bounds(0, 1)

    # 제약 1: 예산
    A_budget = prices.reshape(1, -1)
    lb_budget = np.array([0.0])
    ub_budget = np.array([float(budget)])

    # 제약 2: 카테고리별 정확히 1개
    categories = sorted(set(str(p.get("category", "")) for p in products))
    A_cat = np.array([
        [1.0 if str(p.get("category", "")) == cat else 0.0 for p in products]
        for cat in categories
    ])
    lb_cat = np.ones(len(categories))
    ub_cat = np.ones(len(categories))

    A            = np.vstack([A_budget, A_cat])
    lower_bounds = np.concatenate([lb_budget, lb_cat])
    upper_bounds = np.concatenate([ub_budget, ub_cat])

    constraints = LinearConstraint(A, lower_bounds, upper_bounds)
    res = milp(c=c, constraints=constraints, bounds=bounds, integrality=integrality)

    if not res.success:
        return []

    selected = [products[i] for i in np.where(res.x > 0.5)[0]]
    return selected
