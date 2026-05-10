# === recommender.py — 셋업 추천 (카테고리별 후보 검색 + 조합 스코어링)

import itertools
from dataclasses import dataclass

import numpy as np

from core.config import (
    CATEGORY_LABELS,
    DEFAULT_CANDIDATE_PER_CATEGORY,
    DEFAULT_SETUP_TOP_K,
)
from core.scoring import minmax_scores
from search.searcher import ProductSearcher


def _category_label(category_code: str) -> str:
    return CATEGORY_LABELS.get(category_code, category_code)


@dataclass
class SetupPreference:
    color_text: str
    theme_text: str
    purpose_text: str
    budget: int | None
    categories: list[str]
    candidate_per_category: int = DEFAULT_CANDIDATE_PER_CATEGORY


class SetupRecommender:
    def __init__(self, searcher: ProductSearcher):
        self.searcher = searcher
        self.last_debug_info: dict = {}

    def build_category_queries(self, pref: SetupPreference, category: str) -> list[tuple[str, float]]:
        label = CATEGORY_LABELS.get(category, category)
        queries: list[tuple[str, float]] = []

        if pref.color_text.strip():
            queries.append((f"{pref.color_text.strip()} 데스크셋업에 어울리는 {label}", 0.35))
        if pref.theme_text.strip():
            queries.append((f"{pref.theme_text.strip()} 데스크셋업에 어울리는 {label}", 0.35))
        if pref.purpose_text.strip():
            queries.append((f"{pref.purpose_text.strip()}에 적합한 {label}", 0.20))

        combined = " ".join(x for x in [pref.color_text.strip(), pref.theme_text.strip(), pref.purpose_text.strip()] if x)
        if combined:
            queries.append((f"{combined}에 어울리는 {label}", 0.10))

        if not queries:
            queries.append((f"데스크셋업에 어울리는 {label}", 1.0))

        weight_sum = sum(weight for _, weight in queries)
        if weight_sum > 0:
            queries = [(q, w / weight_sum) for q, w in queries]

        return queries

    def search_category_candidates(
        self,
        pref: SetupPreference,
        category: str,
        mode: str = "image_only",
        image_weight: float = 1.0,
        text_weight: float = 0.0,
    ) -> list[dict]:
        query_specs = self.build_category_queries(pref, category)
        aggregate_scores: dict[int, float] = {}
        product_cache: dict[int, dict] = {}

        max_price = pref.budget if pref.budget is not None else None
        per_query_top_k = max(1, pref.candidate_per_category * 3)

        for query_text, query_weight in query_specs:
            items = self.searcher.search(
                query_text=query_text,
                category_filter=category,
                max_price=max_price,
                top_k=per_query_top_k,
                mode=mode,
                image_weight=image_weight,
                text_weight=text_weight,
            )
            if not items:
                continue

            norm_scores = minmax_scores(items, score_key="final_score")
            for item in items:
                pid = int(item["id"])
                weighted = norm_scores.get(pid, 0.0) * query_weight
                aggregate_scores[pid] = aggregate_scores.get(pid, 0.0) + weighted

                if pid not in product_cache:
                    product_cache[pid] = dict(item)

        if not aggregate_scores:
            return []

        merged: list[dict] = []
        for pid, score in aggregate_scores.items():
            item = dict(product_cache[pid])
            item["product_match_score"] = float(score)
            merged.append(item)

        merged.sort(key=lambda x: x["product_match_score"], reverse=True)
        return merged[: pref.candidate_per_category]

    def recommend_setup(
        self,
        pref: SetupPreference,
        setup_top_k: int = DEFAULT_SETUP_TOP_K,
        mode: str = "image_only",
        image_weight: float = 1.0,
        text_weight: float = 0.0,
    ) -> list[dict]:
        categories = [c.strip() for c in pref.categories if c.strip()]
        if not categories:
            self.last_debug_info = {
                "status": "no_category",
                "message": "카테고리 입력이 비어 있습니다.",
                "candidate_counts": {},
                "categories": [],
                "total_combinations": 0,
                "budget_filtered": 0,
            }
            return []

        candidate_counts: dict[str, int] = {}
        category_candidates: dict[str, list[dict]] = {}
        for category in categories:
            candidates = self.search_category_candidates(
                pref=pref,
                category=category,
                mode=mode,
                image_weight=image_weight,
                text_weight=text_weight,
            )
            candidate_counts[category] = len(candidates)
            if not candidates:
                self.last_debug_info = {
                    "status": "no_candidate",
                    "message": f"카테고리 후보 부족: {category} ({_category_label(category)})",
                    "candidate_counts": candidate_counts,
                    "categories": categories,
                    "total_combinations": 0,
                    "budget_filtered": 0,
                }
                return []
            category_candidates[category] = candidates

        # 카테고리가 많은 경우 조합 폭발 방지를 위해 후보 수를 줄일 수 있도록 기본 방어 로직 적용.
        candidate_limit = pref.candidate_per_category
        if len(categories) >= 5 and candidate_limit > 5:
            candidate_limit = 5

        candidate_lists = [category_candidates[c][:candidate_limit] for c in categories]

        setups: list[dict] = []
        total_combinations = 1
        for items in candidate_lists:
            total_combinations *= len(items)
        budget_filtered = 0

        for combo in itertools.product(*candidate_lists):
            known_prices = [int(item["lprice"]) for item in combo if item.get("lprice") is not None]
            unknown_price_count = sum(1 for item in combo if item.get("lprice") is None)
            total_price = sum(known_prices)

            if pref.budget is not None and total_price > pref.budget:
                budget_filtered += 1
                continue

            avg_match_score = float(np.mean([float(item.get("product_match_score", 0.0)) for item in combo]))

            if pref.budget is None:
                budget_score = 1.0
            else:
                remaining = pref.budget - total_price
                if remaining < 0:
                    budget_filtered += 1
                    continue
                budget_score = 1.0 - abs(remaining / pref.budget) * 0.3 if pref.budget > 0 else 0.0
                budget_score = max(0.0, min(1.0, budget_score))

            completeness_score = 1.0
            setup_score = (0.65 * avg_match_score) + (0.25 * budget_score) + (0.10 * completeness_score)

            if unknown_price_count > 0:
                setup_score -= min(0.15, 0.05 * unknown_price_count)

            setup_items = {category: dict(item) for category, item in zip(categories, combo)}
            setups.append(
                {
                    "setup_score": float(max(0.0, setup_score)),
                    "total_price": int(total_price),
                    "budget": pref.budget,
                    "budget_score": float(budget_score),
                    "avg_match_score": float(avg_match_score),
                    "completeness_score": float(completeness_score),
                    "unknown_price_count": int(unknown_price_count),
                    "items": setup_items,
                }
            )

        setups.sort(key=lambda x: x["setup_score"], reverse=True)
        status = "ok" if setups else "all_filtered"
        message = "정상 생성" if setups else "예산/조건으로 모든 조합이 제거되었습니다."
        self.last_debug_info = {
            "status": status,
            "message": message,
            "candidate_counts": candidate_counts,
            "categories": categories,
            "total_combinations": total_combinations,
            "budget_filtered": budget_filtered,
        }
        return setups[:setup_top_k]
