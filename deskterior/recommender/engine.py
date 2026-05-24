import json
import logging
import math
import os
from dataclasses import dataclass, field

import numpy as np

try:
    import psycopg2
    _PSYCOPG_AVAILABLE = True
except ImportError:
    _PSYCOPG_AVAILABLE = False

from deskterior.recommender.config import (
    CANDIDATE_LIMIT,
    BEAM_SIZE_DEFAULT,
    TOP_M_DEFAULT,
    TOP_K_DEFAULT,
    ALLOW_THEME_GATE_FALLBACK,
    THEME_PRESETS,
    THEME_CATEGORY_QUERIES,
    CATEGORY_EXCLUDE_KEYWORDS,
    CATEGORY_LABELS,
    MANDATORY_CATEGORIES,
    OPTIONAL_CATEGORIES,
    ROLE_PAIR_WEIGHTS,
    OPTIONAL_GAIN_THRESHOLD,
)
from deskterior.retrieval.searcher import embed_text_query
from deskterior.core.config import DB_CONFIG

logger = logging.getLogger(__name__)


def get_db_connection():
    return psycopg2.connect(**DB_CONFIG)


# ============================================================
# DATA CLASSES
# ============================================================

@dataclass
class Product:
    id: str
    name: str
    category: str
    price: int
    color_tags: list = field(default_factory=list)
    style_tags: list = field(default_factory=list)
    image_sim: float = 0.0
    text_sim: float = 0.5
    image_embedding: list | None = None
    text_embedding: list | None = None
    product_url: str | None = None
    image_url: str | None = None
    metadata: dict | None = None


@dataclass
class ScoredProduct:
    product: Product
    theme_evidence: float = 0.0
    value_score: float = 0.0
    item_score: float = 0.0


@dataclass
class Bundle:
    items: list = field(default_factory=list)
    total_price: int = 0
    setup_score: float = 0.0
    final_score: float = 0.0
    setup_theme_evidence: float = 0.0
    item_avg_score: float = 0.0
    role_aware_compatibility: float = 0.0
    mandatory_coverage: float = 0.0
    optional_usefulness: float = 0.0
    value_efficiency: float = 0.0


@dataclass
class _BeamState:
    items: list = field(default_factory=list)
    total_price: int = 0
    covered_categories: set = field(default_factory=set)
    quick_score: float = 0.0

    def to_bundle(self) -> Bundle:
        return Bundle(
            items=list(self.items),
            total_price=self.total_price,
        )
def search_products_by_vector(
    category: str,
    query_embedding: list[float],
    limit: int = CANDIDATE_LIMIT,
) -> list[Product]:
    conn = get_db_connection()
    products: list[Product] = []
    try:
        qvec = np.array(query_embedding, dtype=np.float32)
        vec_str = "[" + ",".join(f"{v:.8f}" for v in qvec.tolist()) + "]"
        sql = """
            SELECT
                id, title, category, lprice, image, link,
                ARRAY[]::text[] AS color_tags,
                ARRAY[]::text[] AS style_tags,
                1 - (embedding_img <=> %s::vector)                   AS image_sim,
                COALESCE(1 - (embedding_txt <=> %s::vector), 0.5)    AS text_sim,
                embedding_img,
                embedding_txt,
                metadata
            FROM products
            WHERE category = %s
              AND lprice IS NOT NULL
              AND lprice > 0
              AND embedding_img IS NOT NULL
            ORDER BY (
                0.60 * (embedding_img <=> %s::vector)
              + COALESCE(0.40 * (embedding_txt <=> %s::vector), 0.20)
            ) ASC
            LIMIT %s
        """
        with conn.cursor() as cur:
            cur.execute(sql, (vec_str, vec_str, category, vec_str, vec_str, limit))
            rows = cur.fetchall()

        for row in rows:
            (pid, name, cat, price, image_url, product_url,
             color_tags, style_tags,
             image_sim, text_sim, img_emb, txt_emb, metadata) = row

            img_emb_list: list[float] | None = None
            if img_emb is not None:
                img_emb_list = json.loads(img_emb) if isinstance(img_emb, str) else np.array(img_emb, dtype=np.float64).tolist()

            txt_emb_list: list[float] | None = None
            if txt_emb is not None:
                txt_emb_list = json.loads(txt_emb) if isinstance(txt_emb, str) else np.array(txt_emb, dtype=np.float64).tolist()

            products.append(Product(
                id=str(pid),
                name=str(name),
                category=str(cat),
                price=int(price),
                color_tags=list(color_tags) if color_tags else [],
                style_tags=list(style_tags) if style_tags else [],
                image_sim=float(image_sim),
                text_sim=float(text_sim),
                image_embedding=img_emb_list,
                text_embedding=txt_emb_list,
                product_url=product_url,
                image_url=image_url,
                metadata=metadata if isinstance(metadata, dict) else (json.loads(metadata) if isinstance(metadata, str) else None),
            ))
    finally:
        conn.close()

    return _apply_exclude_filter(products, category)


def _apply_exclude_filter(products: list[Product], category: str) -> list[Product]:
    exclude_kws = CATEGORY_EXCLUDE_KEYWORDS.get(category, [])
    if not exclude_kws:
        return products
    return [
        p for p in products
        if not any(kw.lower() in p.name.lower() for kw in exclude_kws)
    ]


def retrieve_candidates(theme: str, category: str, limit: int = CANDIDATE_LIMIT) -> list[Product]:
    query_text = THEME_CATEGORY_QUERIES[theme][category]
    query_embedding = embed_text_query(query_text)
    return search_products_by_vector(category, query_embedding, limit)


# ============================================================
# CLI
# ============================================================

def choose_from_menu(title: str, options: list[str], labels: list[str] | None = None) -> str:
    print(f"\n{title}")
    print("-" * len(title))
    display_labels = labels if labels else options
    for i, (opt, lbl) in enumerate(zip(options, display_labels), 1):
        print(f"  {i}. {lbl}")
    while True:
        try:
            raw = input(f"\n번호 입력 (1-{len(options)}): ").strip()
            idx = int(raw) - 1
            if 0 <= idx < len(options):
                return options[idx]
            print(f"  1에서 {len(options)} 사이의 번호를 입력하세요.")
        except (ValueError, EOFError):
            print("  올바른 번호를 입력하세요.")


def choose_budget() -> int:
    while True:
        try:
            raw = input("\n예산 입력 (원, 숫자만): ").strip().replace(",", "")
            amount = int(raw)
            if amount <= 0:
                print("  양수를 입력하세요.")
                continue
            return amount
        except (ValueError, EOFError):
            print("  올바른 숫자를 입력하세요.")


# ============================================================
# SCORING — ITEM LEVEL
# ============================================================

def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def compute_category_theme_bonus(product: Product, theme: str) -> float:
    """카테고리별 강한 테마 증거에 대한 추가 보너스."""
    name = product.name.lower()
    cat  = product.category

    if theme == "gaming":
        if cat == "MONITOR"  and any(k in name for k in ["144hz", "165hz", "게이밍"]):
            return 0.20
        if cat == "KEYBOARD" and any(k in name for k in ["기계식", "rgb", "게이밍"]):
            return 0.20
        if cat == "MOUSE"    and any(k in name for k in ["dpi", "게이밍", "rgb", "경량"]):
            return 0.20
        if cat == "HEADSET"  and any(k in name for k in ["7.1", "서라운드", "게이밍"]):
            return 0.20
        if cat == "MOUSEPAD" and any(k in name for k in ["rgb", "장패드", "게이밍"]):
            return 0.20

    elif theme == "white":
        if any(k in name for k in ["화이트", "아이보리", "크림"]):
            return 0.15

    elif theme == "black":
        if any(k in name for k in ["블랙", "검정", "다크그레이", "무광"]):
            return 0.15

    elif theme == "wood":
        if any(k in name for k in ["우드", "원목", "베이지", "브라운", "오크", "월넛"]):
            return 0.18
        if cat in ["DESK_LAMP", "SPEAKER", "MOUSEPAD"]:
            return 0.08

    return 0.0


def compute_theme_evidence(product: Product, theme: str) -> float:
    """
    ThemeEvidence = positive keyword match - negative keyword penalty + category bonus
    상품명/태그에서 테마 증거를 직접 측정한다.
    """
    text = (
        product.name + " "
        + " ".join(product.color_tags) + " "
        + " ".join(product.style_tags)
    ).lower()

    positive = THEME_PRESETS[theme]["positive_keywords"]
    negative = THEME_PRESETS[theme]["negative_keywords"]

    pos_hits = sum(1 for kw in positive if kw.lower() in text)
    neg_hits = sum(1 for kw in negative if kw.lower() in text)

    pos_score   = min(1.0, pos_hits / 2.0)
    neg_penalty = min(0.6, neg_hits * 0.20)
    cat_bonus   = compute_category_theme_bonus(product, theme)

    return clamp(pos_score + cat_bonus - neg_penalty)


def compute_item_score(
    image_sim:      float,
    text_sim:       float,
    theme_evidence: float,
    value_score:    float,
) -> float:
    """ItemScore = 0.30*ImageSim + 0.25*TextSim + 0.35*ThemeEvidence + 0.10*ValueScore"""
    return clamp(
        0.30 * image_sim
        + 0.25 * text_sim
        + 0.35 * theme_evidence
        + 0.10 * value_score
    )


def normalize_value_scores(scored_by_category: dict[str, list[ScoredProduct]]) -> None:
    """카테고리 내에서 ValueScore를 min-max 정규화 (in-place).

    raw_relevance에 theme_evidence를 포함해 알고리즘 철학과 일치시킨다.
    테마 증거가 약한데 가격만 싼 상품이 value_score에서 살아남지 않도록 한다.
    """
    for _, products in scored_by_category.items():
        if not products:
            continue
        raws = []
        for sp in products:
            price = sp.product.price if sp.product.price > 0 else 1
            raw_relevance = (
                0.30 * clamp(sp.product.image_sim)
                + 0.25 * clamp(sp.product.text_sim)
                + 0.35 * sp.theme_evidence
            )
            raws.append(raw_relevance / math.log(1 + price))

        min_val = min(raws)
        max_val = max(raws)
        span    = max_val - min_val

        for sp, raw in zip(products, raws):
            sp.value_score = clamp((raw - min_val) / span) if span > 1e-9 else 0.5


# ============================================================
# SCORING — SETUP LEVEL
# ============================================================

def compute_setup_theme_evidence(bundle: Bundle) -> float:
    if not bundle.items:
        return 0.0
    return sum(sp.theme_evidence for sp in bundle.items) / len(bundle.items)


def compute_role_aware_compatibility(bundle: Bundle) -> float:
    """
    ROLE_PAIR_WEIGHTS를 이용해 핵심 카테고리 쌍의 테마 일치도를 가중 평균한다.
    키보드-마우스, 마우스-마우스패드 등의 상호보완 관계를 우선시한다.
    """
    if len(bundle.items) <= 1:
        return 0.5

    total_weight = 0.0
    total_score  = 0.0

    for i in range(len(bundle.items)):
        for j in range(i + 1, len(bundle.items)):
            cat_a = bundle.items[i].product.category
            cat_b = bundle.items[j].product.category
            pair  = frozenset((cat_a, cat_b))
            weight = ROLE_PAIR_WEIGHTS.get(pair, 0.3)

            pair_score = (bundle.items[i].theme_evidence + bundle.items[j].theme_evidence) / 2.0
            total_weight += weight
            total_score  += weight * pair_score

    if total_weight < 1e-9:
        return 0.5
    return clamp(total_score / total_weight)


def compute_mandatory_coverage(bundle: Bundle) -> float:
    covered = {sp.product.category for sp in bundle.items}
    hits = sum(1 for c in MANDATORY_CATEGORIES if c in covered)
    return hits / len(MANDATORY_CATEGORIES)


def compute_optional_usefulness(bundle: Bundle, theme: str) -> float:
    optional_items = [sp for sp in bundle.items if sp.product.category in OPTIONAL_CATEGORIES]
    if not optional_items:
        return 0.5

    priority_map = THEME_PRESETS[theme]["optional_priority"]
    scores = [
        sp.theme_evidence * priority_map.get(sp.product.category, 0.5)
        for sp in optional_items
    ]
    return clamp(sum(scores) / len(scores))


def compute_value_efficiency_bundle(bundle: Bundle) -> float:
    if not bundle.items:
        return 0.0
    return sum(sp.value_score for sp in bundle.items) / len(bundle.items)


def compute_setup_score(bundle: Bundle, theme: str) -> Bundle:
    """
    SetupScore = 0.35*SetupThemeEvidence + 0.20*AvgItemScore
               + 0.20*RoleAwareCompatibility + 0.15*MandatoryCoverage
               + 0.05*OptionalUsefulness + 0.05*ValueEfficiency
    """
    if not bundle.items:
        bundle.setup_score = 0.0
        bundle.final_score = 0.0
        return bundle

    bundle.setup_theme_evidence     = compute_setup_theme_evidence(bundle)
    bundle.item_avg_score           = sum(sp.item_score for sp in bundle.items) / len(bundle.items)
    bundle.role_aware_compatibility = compute_role_aware_compatibility(bundle)
    bundle.mandatory_coverage       = compute_mandatory_coverage(bundle)
    bundle.optional_usefulness      = compute_optional_usefulness(bundle, theme)
    bundle.value_efficiency         = compute_value_efficiency_bundle(bundle)

    bundle.setup_score = clamp(
        0.35 * bundle.setup_theme_evidence
        + 0.20 * bundle.item_avg_score
        + 0.20 * bundle.role_aware_compatibility
        + 0.15 * bundle.mandatory_coverage
        + 0.05 * bundle.optional_usefulness
        + 0.05 * bundle.value_efficiency
    )
    bundle.final_score = bundle.setup_score
    return bundle


# ============================================================
# OPTIONAL GAIN
# ============================================================

def compute_optional_gain(
    bundle:    Bundle,
    candidate: ScoredProduct,
    theme:     str,
    category:  str,
) -> float:
    """
    OptionalGain = 0.35*ItemScore + 0.35*ThemeEvidence
                 + 0.20*RoleCompatibility + 0.10*ThemeOptionalPriority
                 - conflict penalty
    """
    if bundle.items:
        weighted_sum = 0.0
        weight_sum   = 0.0
        for sp in bundle.items:
            pair   = frozenset((sp.product.category, category))
            weight = ROLE_PAIR_WEIGHTS.get(pair, 0.3)
            agree  = (sp.theme_evidence + candidate.theme_evidence) / 2.0
            weighted_sum += weight * agree
            weight_sum   += weight
        role_compat = weighted_sum / max(weight_sum, 1e-9)
    else:
        role_compat = 0.5

    priority = THEME_PRESETS[theme]["optional_priority"].get(category, 0.5)

    gain = (
        0.35 * candidate.item_score
        + 0.35 * candidate.theme_evidence
        + 0.20 * role_compat
        + 0.10 * priority
    )

    neg_kws = THEME_PRESETS[theme]["negative_keywords"]
    name    = candidate.product.name.lower()
    if any(kw.lower() in name for kw in neg_kws):
        gain -= 0.15

    return clamp(gain)


# ============================================================
# THEME GATE
# ============================================================

def passes_theme_gate(bundle: Bundle, theme: str) -> bool:
    """테마가 명확하지 않은 조합을 탈락시킨다."""
    avg_evidence = compute_setup_theme_evidence(bundle)
    optional_covered = {
        sp.product.category for sp in bundle.items
        if sp.product.category in OPTIONAL_CATEGORIES
    }

    def mandatory_evidence_count() -> int:
        return sum(
            1 for sp in bundle.items
            if sp.product.category in MANDATORY_CATEGORIES and sp.theme_evidence > 0.2
        )

    def conflict_count() -> int:
        neg_kws = THEME_PRESETS[theme]["negative_keywords"]
        return sum(
            1 for sp in bundle.items
            if any(kw.lower() in sp.product.name.lower() for kw in neg_kws)
        )

    mand_cnt = mandatory_evidence_count()
    conf_cnt = conflict_count()

    if theme == "white":
        return mand_cnt >= 1 and avg_evidence >= 0.58 and conf_cnt <= 1

    if theme == "black":
        return mand_cnt >= 2 and avg_evidence >= 0.60 and conf_cnt <= 1

    if theme == "gaming":
        return mand_cnt >= 2 and avg_evidence >= 0.65

    if theme == "wood":
        wood_optional_ok = sum(
            1 for sp in bundle.items
            if sp.product.category in {"DESK_LAMP", "MOUSEPAD", "SPEAKER"}
            and sp.theme_evidence > 0.2
        )
        if optional_covered:
            return avg_evidence >= 0.53 and wood_optional_ok >= 1 and conf_cnt <= 1
        # 선택 상품이 하나도 없으면 필수 3종만으로 우드 느낌을 내야 하므로 기준 상향
        return avg_evidence >= 0.60 and conf_cnt <= 1

    return False


# ============================================================
# OPTIMIZATION
# ============================================================

def _compute_quick_score(items: list[ScoredProduct]) -> float:
    """Beam 가지치기용 빠른 점수.

    필수 상품의 평균을 base로 삼고, optional은 bonus로 더한다.
    평균을 전체 아이템 수로 나누면 optional 추가 시 점수가 오히려 낮아지는
    문제가 생기므로, 필수 슬롯 수로만 나눈다.
    """
    if not items:
        return 0.0
    mandatory_items = [sp for sp in items if sp.product.category in MANDATORY_CATEGORIES]
    optional_items  = [sp for sp in items if sp.product.category in OPTIONAL_CATEGORIES]

    base           = sum(sp.item_score for sp in mandatory_items) / max(len(mandatory_items), 1)
    optional_bonus = sum(sp.item_score * sp.theme_evidence for sp in optional_items) * 0.30

    return base + optional_bonus


def _has_all_mandatory(state: _BeamState) -> bool:
    return all(cat in state.covered_categories for cat in MANDATORY_CATEGORIES)


def pareto_filter_bundles(bundles: list[Bundle]) -> list[Bundle]:
    """더 비싸면서 점수가 낮거나 같은 조합을 제거한다."""
    dominated: set[int] = set()
    for i, a in enumerate(bundles):
        for j, b in enumerate(bundles):
            if i == j or j in dominated:
                continue
            if (
                a.total_price <= b.total_price
                and a.setup_score >= b.setup_score
                and (a.total_price < b.total_price or a.setup_score > b.setup_score)
            ):
                dominated.add(j)
    return [b for i, b in enumerate(bundles) if i not in dominated]


def diversify_top_bundles(bundles: list[Bundle], top_k: int = 3) -> list[Bundle]:
    """상품 구성이 너무 유사한 조합이 반복되지 않도록 다양성을 보정한다."""
    if len(bundles) <= top_k:
        return bundles

    sorted_b  = sorted(bundles, key=lambda b: b.final_score, reverse=True)
    selected: list[Bundle] = [sorted_b[0]]

    for bundle in sorted_b[1:]:
        if len(selected) >= top_k:
            break
        bundle_ids = {sp.product.id for sp in bundle.items}
        too_similar = any(
            len(bundle_ids & {sp.product.id for sp in sel.items})
            / max(len(bundle_ids | {sp.product.id for sp in sel.items}), 1) > 0.6
            for sel in selected
        )
        if not too_similar:
            selected.append(bundle)

    # 다양성 기준으로 top_k를 채우지 못했으면 점수 순으로 보충
    for bundle in sorted_b:
        if len(selected) >= top_k:
            break
        if bundle not in selected:
            selected.append(bundle)

    return selected


def recommend_setup(
    theme:                 str,
    budget:                int,
    candidates_by_category: dict[str, list[ScoredProduct]],
    beam_size: int = BEAM_SIZE_DEFAULT,
    top_m:     int = TOP_M_DEFAULT,
    top_k:     int = TOP_K_DEFAULT,
) -> list[Bundle]:
    """
    2단계 Beam Search 번들 추천.
    Phase 1: 필수 3종(MONITOR, KEYBOARD, MOUSE) 조합 생성
    Phase 2: 선택 상품 OptionalGain threshold 기준 추가
    """
    # ── Phase 1: 필수 카테고리 Beam Search ──────────────────────────────────
    beams: list[_BeamState] = [_BeamState()]

    for category in MANDATORY_CATEGORIES:
        new_beams: list[_BeamState] = []
        candidates = candidates_by_category.get(category, [])[:top_m]

        for state in beams:
            for cand in candidates:
                new_price = state.total_price + cand.product.price
                if new_price > budget:
                    continue
                new_state = _BeamState(
                    items=state.items + [cand],
                    total_price=new_price,
                    covered_categories=state.covered_categories | {category},
                    quick_score=_compute_quick_score(state.items + [cand]),
                )
                new_beams.append(new_state)

        if not new_beams:
            logger.warning(f"필수 카테고리 {category} — 예산 내 후보 없음. 빈 결과 반환.")
            return []

        new_beams.sort(key=lambda s: s.quick_score, reverse=True)
        beams = new_beams[:beam_size]

    # 필수 슬롯 누락 조합 탈락 (defensive)
    beams = [b for b in beams if _has_all_mandatory(b)]
    if not beams:
        return []

    # ── Phase 2: 선택 상품 추가 ───────────────────────────────────────────
    for category in OPTIONAL_CATEGORIES:
        new_beams = list(beams)  # skip 옵션 유지
        threshold  = OPTIONAL_GAIN_THRESHOLD[theme][category]
        candidates = candidates_by_category.get(category, [])[:top_m]

        for state in beams:
            temp_bundle = state.to_bundle()
            for cand in candidates:
                new_price = state.total_price + cand.product.price
                if new_price > budget:
                    continue
                gain = compute_optional_gain(temp_bundle, cand, theme, category)
                if gain >= threshold:
                    new_state = _BeamState(
                        items=state.items + [cand],
                        total_price=new_price,
                        covered_categories=state.covered_categories | {category},
                        quick_score=_compute_quick_score(state.items + [cand]),
                    )
                    new_beams.append(new_state)

        new_beams.sort(key=lambda s: s.quick_score, reverse=True)
        beams = new_beams[:beam_size]

    # ── Setup Score 계산 + 중복 제거 ────────────────────────────────────────
    # quick_score와 setup_score는 다를 수 있으므로 전체 beam에 대해 계산한다.
    final_bundles:     list[Bundle]        = []
    seen_signatures:   set[frozenset[str]] = set()

    for state in beams:
        bundle = state.to_bundle()
        sig    = frozenset(sp.product.id for sp in bundle.items)
        if sig in seen_signatures:
            continue
        seen_signatures.add(sig)
        compute_setup_score(bundle, theme)
        final_bundles.append(bundle)

    if not final_bundles:
        return []

    # ── Theme Gate ──────────────────────────────────────────────────────────
    gated = [b for b in final_bundles if passes_theme_gate(b, theme)]
    if not gated:
        if ALLOW_THEME_GATE_FALLBACK:
            # 테마 불명확 조합도 허용하되 0.75 패널티 부여
            for b in final_bundles:
                b.final_score = b.setup_score * 0.75
            gated = final_bundles
        else:
            return []

    # ── Pareto Filtering ────────────────────────────────────────────────────
    gated = pareto_filter_bundles(gated)

    # ── 다양성 보정 + 최종 정렬 ────────────────────────────────────────────
    gated = sorted(gated, key=lambda b: b.final_score, reverse=True)
    gated = diversify_top_bundles(gated, top_k)

    return gated[:top_k]


# ============================================================
# OUTPUT
# ============================================================

def generate_explanation(bundle: Bundle, theme: str) -> list[str]:
    reasons: list[str] = []
    theme_label = THEME_PRESETS[theme]["label"]
    covered     = {sp.product.category for sp in bundle.items}

    # 필수 커버리지
    if all(c in covered for c in MANDATORY_CATEGORIES):
        reasons.append("필수 상품(모니터, 키보드, 마우스)을 모두 포함했습니다.")

    # 필수 상품 테마 증거
    mand_with_ev = sum(
        1 for sp in bundle.items
        if sp.product.category in MANDATORY_CATEGORIES and sp.theme_evidence > 0.2
    )
    if mand_with_ev >= 2:
        reasons.append(
            f"필수 상품 {mand_with_ev}개 이상이 {theme_label} 키워드를 포함합니다."
        )

    # 전체 테마 인식도
    avg_ev = bundle.setup_theme_evidence
    if avg_ev >= 0.55:
        reasons.append(
            f"조합 전체의 테마 인식도가 높습니다. (평균 테마 점수: {avg_ev:.2f})"
        )

    # 역할 호환성
    kbd      = next((sp for sp in bundle.items if sp.product.category == "KEYBOARD"),  None)
    mouse    = next((sp for sp in bundle.items if sp.product.category == "MOUSE"),     None)
    mousepad = next((sp for sp in bundle.items if sp.product.category == "MOUSEPAD"), None)
    if kbd and mouse and mousepad:
        reasons.append(
            "키보드, 마우스, 마우스패드가 같은 입력 장치군으로 상호보완성이 높습니다."
        )
    elif kbd and mouse:
        reasons.append("키보드와 마우스 조합의 역할 호환성이 높습니다.")

    # 선택 상품 기여
    optional_items = [sp for sp in bundle.items if sp.product.category in OPTIONAL_CATEGORIES]
    if optional_items:
        labels = [CATEGORY_LABELS.get(sp.product.category, sp.product.category) for sp in optional_items]
        reasons.append(f"선택 상품({', '.join(labels)})이 테마를 강화합니다.")

    reasons.append("예산을 억지로 소진하지 않고 테마에 맞는 상품 조합을 선택했습니다.")
    return reasons


def print_recommendation_results(
    bundles: list[Bundle],
    theme:   str,
    budget:  int,
) -> None:
    theme_label = THEME_PRESETS[theme]["label"]

    print("\n" + "=" * 56)
    print(f"  추천 셋업 TOP {len(bundles)}  —  {theme_label}")
    print("=" * 56)

    if not bundles:
        print("\n추천 결과가 없습니다. 예산을 높이거나 테마를 바꿔 보세요.")
        return

    for rank, bundle in enumerate(bundles, 1):
        usage = bundle.total_price / budget * 100 if budget > 0 else 0

        print(f"\n[추천 셋업 {rank}]")
        print(
            f"총 가격: {bundle.total_price:,}원 / "
            f"예산: {budget:,}원 / "
            f"예산 사용률: {usage:.1f}%"
        )
        print(f"Setup Score: {bundle.setup_score:.3f}  |  Final Score: {bundle.final_score:.3f}")

        print("\n세부 점수:")
        print(f"  - 테마 인식도(SetupThemeEvidence): {bundle.setup_theme_evidence:.2f}")
        print(f"  - 평균 상품 점수(AvgItemScore):     {bundle.item_avg_score:.2f}")
        print(f"  - 역할 호환성(RoleAwareCompat):     {bundle.role_aware_compatibility:.2f}")
        print(f"  - 필수 커버리지(MandatoryCoverage): {bundle.mandatory_coverage:.2f}")
        print(f"  - 선택 유용성(OptionalUsefulness):  {bundle.optional_usefulness:.2f}")
        print(f"  - 가격 효율성(ValueEfficiency):     {bundle.value_efficiency:.2f}")

        mandatory_items = [sp for sp in bundle.items if sp.product.category in MANDATORY_CATEGORIES]
        optional_items  = [sp for sp in bundle.items if sp.product.category in OPTIONAL_CATEGORIES]

        print("\n필수 상품:")
        for sp in mandatory_items:
            _print_product_line(sp)

        if optional_items:
            print("\n추가 상품:")
            for sp in optional_items:
                _print_product_line(sp)

        print("\n추천 이유:")
        for reason in generate_explanation(bundle, theme):
            print(f"  - {reason}")

        if rank < len(bundles):
            print("\n" + "-" * 56)


def _format_size(metadata: dict | None, category: str) -> str:
    if not metadata:
        return ""
    if category == "MONITOR":
        inch = metadata.get("diagonal_inch")
        w    = metadata.get("width_mm")
        h    = metadata.get("height_mm")
        if inch and w and h:
            return f"{inch}인치 ({w:.0f}×{h:.0f}mm)"
        if inch:
            return f"{inch}인치"
    w = metadata.get("width_mm")
    d = metadata.get("depth_mm")
    if w and d:
        return f"{w:.0f}×{d:.0f}mm"
    return ""


def _print_product_line(sp: ScoredProduct) -> None:
    p         = sp.product
    cat_label = CATEGORY_LABELS.get(p.category, p.category)
    url_note  = f"\n      URL: {p.product_url}" if p.product_url else ""

    img_path = os.path.join("data", "raw", "processed_images", f"{p.id}.png")
    img_note = f"\n      이미지: {img_path}" if os.path.exists(img_path) else ""

    size_str  = _format_size(p.metadata, p.category)
    size_note = f"  |  크기: {size_str}" if size_str else ""

    print(
        f"  - [{cat_label}] {p.name}\n"
        f"      가격: {p.price:,}원{size_note}  |  "
        f"image_sim={p.image_sim:.2f}  text_sim={p.text_sim:.2f}  "
        f"theme_ev={sp.theme_evidence:.2f}  item_score={sp.item_score:.2f}"
        f"{url_note}"
        f"{img_note}"
    )


# ============================================================
# OPTIONAL VISUAL HELPERS
# ============================================================

def create_product_collage(bundle: Bundle):
    """[Optional] 상품 이미지를 그리드로 합성해 반환. PIL/requests 없으면 None."""
    try:
        import requests
        from PIL import Image
        from io import BytesIO
        import math as _math

        images = []
        for sp in bundle.items:
            if sp.product.image_url:
                try:
                    resp = requests.get(sp.product.image_url, timeout=5)
                    img  = Image.open(BytesIO(resp.content)).convert("RGB")
                    images.append(img.resize((200, 200)))
                except Exception:
                    continue

        if not images:
            return None

        cols    = min(len(images), 4)
        rows    = _math.ceil(len(images) / cols)
        collage = Image.new("RGB", (cols * 200, rows * 200), (255, 255, 255))
        for idx, img in enumerate(images):
            row, col = divmod(idx, cols)
            collage.paste(img, (col * 200, row * 200))
        return collage
    except Exception:
        return None


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    print("\n" + "=" * 48)
    print("    Deskterior Setup Recommender")
    print("    테마 + 예산 기반 멀티모달 번들 추천")
    print("=" * 48)

    # ── 1. 사용자 입력: 테마 + 예산 ──────────────────────────────────────────
    theme_keys   = list(THEME_PRESETS.keys())
    theme_labels = [
        f"{THEME_PRESETS[k]['label']}  —  {THEME_PRESETS[k]['description']}"
        for k in theme_keys
    ]
    theme  = choose_from_menu("테마 선택", theme_keys, theme_labels)
    budget = choose_budget()

    print(f"\n선택 결과:")
    print(f"  테마:  {THEME_PRESETS[theme]['label']}")
    print(f"  예산:  {budget:,}원")
    print("=" * 48)

    # ── 2. 모델 초기화 ────────────────────────────────────────────────────────
    print("\n모델 초기화 중...")
    theme_prompt = THEME_PRESETS[theme]["theme_prompt"]
    _ = embed_text_query(theme_prompt)
    print(f"  Theme prompt: {theme_prompt}")

    # ── 3. DB 연결 확인 ───────────────────────────────────────────────────────
    if not _PSYCOPG_AVAILABLE:
        raise RuntimeError("psycopg[binary] 또는 pgvector 패키지가 없습니다.")
    try:
        conn = get_db_connection()
        conn.close()
        print("DB 연결 성공. 실제 상품 DB에서 후보를 검색합니다.")
    except Exception as e:
        raise RuntimeError(f"DB 연결 실패: {e}") from e

    # ── 4. 카테고리별 후보 검색 ───────────────────────────────────────────────
    all_categories = MANDATORY_CATEGORIES + OPTIONAL_CATEGORIES
    print(f"\n카테고리별 후보 검색 중... (카테고리당 최대 {CANDIDATE_LIMIT}개)\n")

    raw_by_category: dict[str, list[ScoredProduct]] = {}

    for category in all_categories:
        query = THEME_CATEGORY_QUERIES[theme][category]
        label = CATEGORY_LABELS[category]
        mand  = "[필수]" if category in MANDATORY_CATEGORIES else "[선택]"
        print(f"  {mand} {label}: {query}")

        products = retrieve_candidates(theme, category)

        scored: list[ScoredProduct] = []
        for p in products:
            te = compute_theme_evidence(p, theme)
            scored.append(ScoredProduct(
                product=p,
                theme_evidence=te,
                value_score=0.0,
                item_score=0.0,
            ))

        raw_by_category[category] = scored
        print(f"         → {len(scored)}개 후보")

    # ── 5. ValueScore 정규화 + ItemScore 최종 계산 ────────────────────────────
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

    # ── 6. 추천 알고리즘 실행 ────────────────────────────────────────────────
    print(f"\n추천 알고리즘 실행 중... (beam_size={BEAM_SIZE_DEFAULT})")
    bundles = recommend_setup(
        theme=theme,
        budget=budget,
        candidates_by_category=raw_by_category,
        beam_size=BEAM_SIZE_DEFAULT,
        top_m=TOP_M_DEFAULT,
        top_k=TOP_K_DEFAULT,
    )

    # ── 7. 결과 출력 ─────────────────────────────────────────────────────────
    print_recommendation_results(bundles, theme, budget)

    if not bundles:
        print("\n유효한 추천 결과가 없습니다.")
        print("  - 예산을 높이거나 다른 테마를 선택해 보세요.")
        print("  - DB에 해당 카테고리 상품이 충분한지 확인하세요.")
    else:
        print("\n" + "=" * 56)
        print("  추천 완료")
        print("=" * 56)


if __name__ == "__main__":
    main()
