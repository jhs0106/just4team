# === cli.py — CLI 입력 처리 및 결과 출력

from core.config import (
    CATEGORY_ALIASES,
    CATEGORY_LABELS,
    CATEGORY_KEYWORDS,
    DEFAULT_CANDIDATE_PER_CATEGORY,
    DEFAULT_IMAGE_WEIGHT,
    DEFAULT_SEARCH_MODE,
    DEFAULT_SETUP_TOP_K,
    DEFAULT_TEXT_WEIGHT,
    DEFAULT_TOP_K,
)
from search.recommender import SetupPreference, SetupRecommender, _category_label
from search.searcher import ProductSearcher


def _normalize_category_token(token: str) -> str | None:
    raw = token.strip()
    if not raw:
        return None

    if raw in CATEGORY_KEYWORDS:
        return raw

    upper = raw.upper()
    if upper in CATEGORY_KEYWORDS:
        return upper

    key = raw.lower().replace(" ", "").replace("-", "_")
    return CATEGORY_ALIASES.get(key)


def _parse_category_input(category_raw: str) -> tuple[list[str], list[str]]:
    tokens = [x.strip() for x in category_raw.split(",") if x.strip()]
    if not tokens:
        return ["KEYBOARD", "MOUSE", "DESK_LAMP"], []

    categories: list[str] = []
    unknown: list[str] = []
    seen: set[str] = set()
    for token in tokens:
        code = _normalize_category_token(token)
        if code is None:
            unknown.append(token)
            continue
        if code not in seen:
            seen.add(code)
            categories.append(code)

    return categories, unknown


def _parse_budget(value: str) -> int | None:
    raw = value.strip().replace(",", "")
    if raw == "":
        return None
    try:
        budget = int(raw)
        return budget if budget >= 0 else None
    except ValueError:
        return None


def _prompt_categories_selection() -> tuple[list[str], list[str]]:
    category_codes = list(CATEGORY_LABELS.keys())
    print("카테고리 선택 (번호 여러 개는 콤마로 입력)")
    for idx, code in enumerate(category_codes, 1):
        print(f"  {idx:2}. {CATEGORY_LABELS[code]} ({code})")

    print("입력 예시: 3,4,12 또는 keyboard,mouse,lamp")
    raw = input("카테고리 입력 (엔터=기본 KEYBOARD,MOUSE,DESK_LAMP): ").strip()
    if not raw:
        return ["KEYBOARD", "MOUSE", "DESK_LAMP"], []

    tokens = [t.strip() for t in raw.split(",") if t.strip()]
    if tokens and all(token.isdigit() for token in tokens):
        selected: list[str] = []
        unknown: list[str] = []
        seen: set[str] = set()
        for token in tokens:
            idx = int(token)
            if idx < 1 or idx > len(category_codes):
                unknown.append(token)
                continue
            code = category_codes[idx - 1]
            if code not in seen:
                seen.add(code)
                selected.append(code)
        return selected, unknown

    return _parse_category_input(raw)


def run_search(searcher: ProductSearcher, query: str):
    results = searcher.search(
        query_text=query,
        top_k=DEFAULT_TOP_K,
        mode=DEFAULT_SEARCH_MODE,
        image_weight=DEFAULT_IMAGE_WEIGHT,
        text_weight=DEFAULT_TEXT_WEIGHT,
    )
    if not results:
        print("결과 없음")
        return
    print(f"\n검색: '{query}'  (mode={DEFAULT_SEARCH_MODE}, img={DEFAULT_IMAGE_WEIGHT}, txt={DEFAULT_TEXT_WEIGHT})")
    print("-" * 70)
    for rank, r in enumerate(results, 1):
        print(f"{rank}위  score={r['final_score']:.4f}  #{r['id']}  {(r['title'] or '').strip()}")


def run_setup_recommendation(recommender: SetupRecommender):
    color_text = input("색감 입력 (예: 화이트톤): ").strip()
    theme_text = input("테마 입력 (예: 미니멀): ").strip()
    purpose_text = input("용도 입력 (예: 개발자 데스크셋업): ").strip()
    budget = _parse_budget(input("예산 입력 (예: 300000, 비우면 무제한): "))

    categories, unknown = _prompt_categories_selection()

    if unknown:
        print(f"[알림] 인식하지 못한 카테고리: {', '.join(unknown)}")

    if not categories:
        print("[오류] 유효한 카테고리가 없습니다. 예: keyboard,mouse,lamp")
        return

    print("[적용 카테고리] " + ", ".join(f"{c}({_category_label(c)})" for c in categories))

    pref = SetupPreference(
        color_text=color_text,
        theme_text=theme_text,
        purpose_text=purpose_text,
        budget=budget,
        categories=categories,
        candidate_per_category=DEFAULT_CANDIDATE_PER_CATEGORY,
    )

    setups = recommender.recommend_setup(
        pref=pref,
        setup_top_k=DEFAULT_SETUP_TOP_K,
        mode="image_only",
        image_weight=1.0,
        text_weight=0.0,
    )

    if not setups:
        print("\n셋업 추천 결과 없음")
        debug = recommender.last_debug_info or {}
        if debug:
            print(f"- 원인: {debug.get('message', '알 수 없음')}")
            counts = debug.get("candidate_counts", {})
            if counts:
                print("- 카테고리별 후보 개수:")
                for category, count in counts.items():
                    print(f"  {category}({_category_label(category)}): {count}개")
            if debug.get("total_combinations"):
                print(f"- 생성 조합 수: {debug.get('total_combinations')}개")
            if debug.get("budget_filtered"):
                print(f"- 예산으로 제거된 조합 수: {debug.get('budget_filtered')}개")
        return

    print("\n[셋업 추천 결과]")
    for i, setup in enumerate(setups, 1):
        budget_label = setup["budget"] if setup["budget"] is not None else "제한없음"
        print(f"\n[셋업 {i}위]")
        print(f"setup_score={setup['setup_score']:.4f}")
        print(f"total_price={setup['total_price']} / budget={budget_label}")

        for category in pref.categories:
            item = setup["items"].get(category)
            if not item:
                continue
            price_label = f"{item['lprice']}원" if item.get("lprice") is not None else "가격미상"
            label = _category_label(category)
            print(
                f"- {category}({label}): #{item['id']} {(item['title'] or '').strip()} | {price_label} | match={item.get('product_match_score', 0.0):.2f}"
            )
