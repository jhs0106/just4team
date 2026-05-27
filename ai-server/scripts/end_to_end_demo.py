# End-to-end 데모: 사용자 입력 → recommendation → AI 서버 → 최종 이미지.
#
# 통합 결정(2026-05-24):
#   - 옵션 X (단일 Python 서버) — recommendation 모듈을 import해서 사용
#   - 옵션 A (사전 처리된 이미지) — recommendation이 image_id 반환, AI 서버가 processed_images/ 참조
#
# 실행:
#   python ai-server/scripts/end_to_end_demo.py --mock     # recommendation 없이 mock으로 테스트
#   python ai-server/scripts/end_to_end_demo.py            # 실제 recommendation + AI 서버 호출

import argparse
import base64
import sys
import time
from pathlib import Path

import requests


REPO_ROOT = Path(__file__).resolve().parents[2]
AI_SERVER = "http://localhost:8000"


def to_b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode()


def make_mock_setup() -> dict:
    # recommendation 모듈 없이 동작 확인용 mock setup 데이터.
    # 실제 SetupRecommender.recommend_setup() 출력과 동일한 형식.
    return {
        "setup_score":     0.83,
        "total_price":     280000,
        "budget":          300000,
        "budget_score":    0.93,
        "avg_match_score": 0.78,
        "items": {
            "MONITOR":  {"id": 315, "title": "LG 27인치 4K", "lprice": 200000,
                         "category": "MONITOR", "brand": "LG",
                         "metadata": {"width_mm": 600, "depth_mm": 200}},
            "KEYBOARD": {"id": 449, "title": "키크론 K2",     "lprice": 80000,
                         "category": "KEYBOARD", "brand": "키크론",
                         "metadata": {"width_mm": 313, "depth_mm": 123}},
            "MOUSE":    {"id": 595, "title": "MX Master 3",  "lprice": 50000,
                         "category": "MOUSE", "brand": "Logitech",
                         "metadata": {"width_mm": 124, "depth_mm": 84}},
        },
    }


def call_real_recommendation(color_text: str, theme_text: str,
                             purpose_text: str, budget: int,
                             categories: list[str]) -> dict:
    # 실제 recommendation 모듈 사용. feature/recommendation 브랜치가 merge되어야 동작.
    try:
        from search.recommender   import SetupPreference, SetupRecommender
        from search.searcher      import ProductSearcher
        from db.db_manager        import DBManager
    except ImportError as e:
        print(f"[ERROR] recommendation 모듈 import 실패: {e}")
        print("       feature/recommendation 브랜치가 merge되었는지 확인.")
        sys.exit(1)

    db          = DBManager()
    searcher    = ProductSearcher(db)
    recommender = SetupRecommender(searcher)

    pref = SetupPreference(
        color_text=color_text,
        theme_text=theme_text,
        purpose_text=purpose_text,
        budget=budget,
        categories=categories,
    )
    setups = recommender.recommend_setup(pref, setup_top_k=1)
    db.close()

    if not setups:
        print(f"[ERROR] recommendation 결과 없음. debug_info={recommender.last_debug_info}")
        sys.exit(1)

    return setups[0]


def poll_ai_server(job_id: str, timeout: float = 300.0) -> dict:
    t0 = time.time()
    while True:
        if time.time() - t0 > timeout:
            raise TimeoutError(f"AI 서버 작업 {job_id} 타임아웃")
        res = requests.get(f"{AI_SERVER}/jobs/{job_id}", timeout=10).json()
        st  = res.get("status")
        if st == "done":
            return res
        if st == "failed":
            raise RuntimeError(f"AI 서버 작업 실패: {res.get('error')}")
        time.sleep(3)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mock", action="store_true",
                        help="recommendation 호출 없이 mock setup 사용")
    parser.add_argument("--desk-image", type=Path,
                        default=REPO_ROOT / "ai-server" / "data" / "test" / "desk_image.jpg")
    parser.add_argument("--desk-width-mm", type=int,   default=1400)
    parser.add_argument("--desk-depth-mm", type=int,   default=700)
    parser.add_argument("--top-view",      type=Path,  default=None)
    parser.add_argument("--color-text",    type=str,   default="화이트")
    parser.add_argument("--theme-text",    type=str,   default="미니멀")
    parser.add_argument("--purpose-text",  type=str,   default="업무용")
    parser.add_argument("--budget",        type=int,   default=300000)
    parser.add_argument("--categories",    type=str,
                        default="MONITOR,KEYBOARD,MOUSE,DESK_LAMP,SPEAKER")
    parser.add_argument("--out",           type=Path,
                        default=REPO_ROOT / "ai-server" / "outputs" / "end_to_end_result.png")
    args = parser.parse_args()

    if not args.desk_image.exists():
        print(f"[ERROR] 책상 이미지 없음: {args.desk_image}")
        sys.exit(1)

    # 1. recommendation (실제 또는 mock)
    if args.mock:
        print("[1/3] mock setup 사용")
        setup = make_mock_setup()
    else:
        print(f"[1/3] recommendation 호출: color={args.color_text!r}, theme={args.theme_text!r}, "
              f"purpose={args.purpose_text!r}, budget={args.budget:,}원")
        cats  = [c.strip() for c in args.categories.split(",") if c.strip()]
        setup = call_real_recommendation(
            color_text=args.color_text, theme_text=args.theme_text,
            purpose_text=args.purpose_text, budget=args.budget, categories=cats,
        )

    print(f"      setup_score={setup.get('setup_score'):.3f}, "
          f"total_price={setup.get('total_price'):,}원, "
          f"products={list(setup.get('items', {}).keys())}")

    # 2. setup → GenerateRequest 변환
    sys.path.insert(0, str(REPO_ROOT / "ai-server"))
    from api.adapters.recommendation_bridge import setup_to_generate_request

    print("[2/3] GenerateRequest로 변환")
    req = setup_to_generate_request(
        setup=setup,
        color_text=args.color_text,
        theme_text=args.theme_text,
        desk_image_b64=to_b64(args.desk_image),
        desk_width_mm=args.desk_width_mm,
        desk_depth_mm=args.desk_depth_mm,
        top_view_image_b64=to_b64(args.top_view) if args.top_view else None,
    )
    print(f"      style={req.style.value}, products={len(req.products)}개")

    # 3. AI 서버 호출
    print("[3/3] AI 서버 /generate 호출")
    resp   = requests.post(f"{AI_SERVER}/generate", json=req.model_dump(), timeout=30)
    resp.raise_for_status()
    job_id = resp.json()["job_id"]
    print(f"      job_id={job_id}, polling...")

    result = poll_ai_server(job_id)
    print(f"      완료: num_removed={result.get('num_removed')}, "
          f"num_placed={result.get('num_placed')}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_bytes(base64.b64decode(result["result_image"]))
    print(f"\n결과 이미지: {args.out}")


if __name__ == "__main__":
    main()
