# /recommend-and-generate 한 번에 호출 + 폴링 + 결과 저장 + 자동 열기.
# 매번 job_id 받고 폴링 명령 따로 치는 번거로움 제거용 데모 스크립트.
#
# 사용:
#   python scripts/run_demo.py                              # default (white, 100만, desk_image2.jpg)
#   python scripts/run_demo.py --theme black --budget 700000
#   python scripts/run_demo.py --desk-image data/test/desk_image2.jpg --out my_result.png
#   python scripts/run_demo.py --no-open                    # 결과 자동 열기 안 함

import argparse
import base64
import os
import sys
import time
from pathlib import Path

import requests


AI_SERVER = "http://localhost:8000"
REPO_ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description="recommend-and-generate end-to-end demo")
    parser.add_argument("--theme",  type=str, default="white",
                        choices=["white", "black", "gaming", "wood"])
    parser.add_argument("--budget", type=int, default=1_000_000)
    parser.add_argument("--desk-image",  type=Path,
                        default=REPO_ROOT / "data" / "test" / "desk_image2.jpg")
    parser.add_argument("--top-view",    type=Path, default=None,
                        help="top-view 이미지 (기본: ai-server가 default 사용)")
    parser.add_argument("--desk-width-mm", type=int, default=1200)
    parser.add_argument("--desk-depth-mm", type=int, default=600)
    parser.add_argument("--out",     type=Path, default=REPO_ROOT / "demo_result.png")
    parser.add_argument("--timeout", type=int, default=900, help="job 최대 대기 시간 (초)")
    parser.add_argument("--poll-interval", type=int, default=10, help="폴링 간격 (초)")
    parser.add_argument("--no-open", action="store_true", help="결과 자동 열기 비활성")
    args = parser.parse_args()

    if not args.desk_image.exists():
        print(f"[ERROR] desk_image 없음: {args.desk_image}")
        sys.exit(1)

    print(f"=== run_demo ===")
    print(f"  theme:     {args.theme}")
    print(f"  budget:    {args.budget:,}원")
    print(f"  desk:      {args.desk_image.name}")
    print(f"  top_view:  {args.top_view.name if args.top_view else '(default 자동)'}")
    print(f"  desk_mm:   {args.desk_width_mm} x {args.desk_depth_mm}")
    print(f"  out:       {args.out}")
    print()

    # 1. 호출
    payload = {
        "theme":          args.theme,
        "budget":         args.budget,
        "image_base64":   base64.b64encode(args.desk_image.read_bytes()).decode(),
        "desk_width_mm":  args.desk_width_mm,
        "desk_depth_mm":  args.desk_depth_mm,
    }
    if args.top_view and args.top_view.exists():
        payload["top_view_image_base64"] = base64.b64encode(args.top_view.read_bytes()).decode()

    print("[1] POST /recommend-and-generate ...")
    try:
        resp = requests.post(
            f"{AI_SERVER}/recommend-and-generate", json=payload, timeout=300,
        )
    except requests.RequestException as e:
        print(f"[ERROR] ai-server 호출 실패: {e}")
        print(f"        ai-server uvicorn이 {AI_SERVER}에서 떠있는지 확인.")
        sys.exit(1)

    if resp.status_code != 200:
        print(f"[ERROR] HTTP {resp.status_code}: {resp.text[:500]}")
        sys.exit(1)

    body   = resp.json()
    job_id = body.get("job_id")
    if not job_id:
        print(f"[ERROR] job_id 없음: {body}")
        sys.exit(1)
    print(f"    job_id={job_id}")
    print()

    # 2. 폴링
    print(f"[2] 폴링 중 (간격 {args.poll_interval}s, 최대 {args.timeout}s)...")
    t0     = time.time()
    prev_placed = -1
    while True:
        elapsed = time.time() - t0
        if elapsed > args.timeout:
            print(f"\n[ERROR] {args.timeout}초 초과. job 미완료.")
            sys.exit(1)
        try:
            j = requests.get(f"{AI_SERVER}/jobs/{job_id}", timeout=15).json()
        except requests.RequestException as e:
            print(f"\n[WARN] 폴링 일시 실패: {e}")
            time.sleep(args.poll_interval)
            continue

        status  = j.get("status")
        placed  = j.get("num_placed", 0)
        removed = j.get("num_removed", 0)

        # 진행 상황 한 줄 갱신
        msg = (
            f"\r    [{int(elapsed):4d}s] status={status:7s} "
            f"removed={removed:2d} placed={placed:2d}/5"
        )
        sys.stdout.write(msg)
        sys.stdout.flush()
        if placed != prev_placed:
            prev_placed = placed

        if status == "done":
            print()
            break
        if status == "failed":
            print()
            err = j.get("error") or "(에러 메시지 없음)"
            print(f"[ERROR] job 실패: {err}")
            sys.exit(1)

        time.sleep(args.poll_interval)

    # 3. 결과 저장
    print(f"\n[3] 결과 저장 ...")
    result_b64 = j.get("result_image")
    if not result_b64:
        print(f"[ERROR] status=done인데 result_image 없음.")
        sys.exit(1)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_bytes(base64.b64decode(result_b64))
    size_kb = args.out.stat().st_size // 1024
    print(f"    저장: {args.out} ({size_kb} KB)")

    # 4. 자동 열기
    if not args.no_open:
        try:
            os.startfile(str(args.out))  # Windows
            print(f"    이미지 자동 열림.")
        except Exception:
            print(f"    (자동 열기 실패 — 수동으로 열어주세요)")

    print(f"\n=== 완료 ({int(time.time()-t0)}초 소요) ===")


if __name__ == "__main__":
    main()
