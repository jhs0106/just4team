import argparse
import json
import math
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as _fm

# 한글 폰트 설정 (Malgun Gothic → NanumGothic → 기본)
for _fn in ["Malgun Gothic", "NanumGothic", "Gulim"]:
    if any(_fn.lower() in f.lower() for f in _fm.findSystemFonts()):
        plt.rcParams["font.family"] = _fn
        break
plt.rcParams["axes.unicode_minus"] = False
import matplotlib.patches as mpatches
import numpy as np

# ── 설정 ─────────────────────────────────────────────────────────────────────
DATASET_JSONL = Path("data/placement_ranker_dataset.jsonl")
RANKER_PKL    = Path("outputs/models/layout_ranker.pkl")
OUT_DIR       = Path("outputs/paper_figures")

PAPER_CATS = ["MONITOR", "KEYBOARD", "MOUSE", "MOUSEPAD", "SPEAKER", "DESK_LAMP", "CLOCK", "DECO"]

_PREFERRED_POS = {
    "MONITOR":      (0.50, 0.20),
    "DESK_SHELF":   (0.50, 0.20),
    "KEYBOARD":     (0.50, 0.62),
    "MOUSEPAD":     (0.50, 0.65),
    "MOUSE":        (0.70, 0.62),
    "SPEAKER":      (0.25, 0.25),
    "DESK_LAMP":    (0.12, 0.30),
    "DECO":         (0.75, 0.35),
    "CLOCK":        (0.80, 0.30),
    "LAPTOP_STAND": (0.50, 0.45),
}

CAT_COLORS = {
    "MONITOR":      "#e74c3c",
    "KEYBOARD":     "#3498db",
    "MOUSE":        "#2ecc71",
    "MOUSEPAD":     "#9b59b6",
    "SPEAKER":      "#f39c12",
    "DESK_LAMP":    "#1abc9c",
    "CLOCK":        "#e67e22",
    "DECO":         "#95a5a6",
    "DESK_SHELF":   "#34495e",
    "LAPTOP_STAND": "#d35400",
}

LABEL_KR = {
    "MONITOR": "모니터", "KEYBOARD": "키보드", "MOUSE": "마우스",
    "MOUSEPAD": "마우스패드", "SPEAKER": "스피커", "DESK_LAMP": "데스크 램프",
    "CLOCK": "시계", "DECO": "데코", "DESK_SHELF": "모니터 받침대",
    "LAPTOP_STAND": "노트북 스탠드",
}


#데이터 로드
def load_dataset(cats: list[str]) -> dict:
    rows = {}
    for cat in cats:
        rows[cat] = {"pos_rx": [], "pos_ry": [], "neg_rx": [], "neg_ry": []}

    for line in DATASET_JSONL.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        d   = json.loads(line)
        cat = d["category"]
        if cat not in rows:
            continue
        feat = d["features"]
        rx, ry = feat[1], feat[2]
        if d["label"] == 1:
            rows[cat]["pos_rx"].append(rx)
            rows[cat]["pos_ry"].append(ry)
        else:
            rows[cat]["neg_rx"].append(rx)
            rows[cat]["neg_ry"].append(ry)

    for cat in rows:
        for k in rows[cat]:
            rows[cat][k] = np.array(rows[cat][k])
    return rows


#1. 카테고리별 Positive / Negative 분포 scatter
def fig_placement_scatter(data: dict, cats: list[str]):
    n    = len(cats)
    cols = 4
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.5, rows * 3.5))
    axes = axes.flatten() if n > 1 else [axes]
    fig.suptitle("배치 학습 데이터: Positive / Negative 분포", fontsize=15, fontweight="bold", y=1.01)

    for i, cat in enumerate(cats):
        ax = axes[i]
        d  = data[cat]
        ax.set_facecolor("#f8f9fa")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.invert_yaxis()

        neg_rx, neg_ry = d["neg_rx"], d["neg_ry"]
        pos_rx, pos_ry = d["pos_rx"], d["pos_ry"]

        if len(neg_rx):
            ax.scatter(neg_rx, neg_ry, s=4, alpha=0.25, c="#e74c3c", label="Negative", rasterized=True)
        if len(pos_rx):
            ax.scatter(pos_rx, pos_ry, s=6, alpha=0.50, c="#2ecc71", label="Positive", rasterized=True)

        # 선호 위치 마킹
        if cat in _PREFERRED_POS:
            px, py = _PREFERRED_POS[cat]
            ax.plot(px, py, marker="*", markersize=14, color="#f39c12",
                    markeredgecolor="#c0392b", markeredgewidth=0.8, zorder=5,
                    label="선호 위치")

        ax.set_title(f"{LABEL_KR.get(cat, cat)}\n(+{len(pos_rx)} / −{len(neg_rx)})",
                     fontsize=9.5, fontweight="bold")
        ax.set_xlabel("rx (좌→우)", fontsize=8); ax.set_ylabel("ry (뒤→앞)", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.3)
        ax.axvline(0.5, color="gray", lw=0.5, ls="--", alpha=0.5)
        ax.axhline(0.5, color="gray", lw=0.5, ls="--", alpha=0.5)

    # 빈 subplot 숨기기
    for j in range(len(cats), len(axes)):
        axes[j].set_visible(False)

    handles = [
        mpatches.Patch(color="#2ecc71", label="Positive (좋은 배치)"),
        mpatches.Patch(color="#e74c3c", label="Negative (나쁜 배치)"),
        plt.Line2D([0], [0], marker="*", color="w", markerfacecolor="#f39c12",
                   markeredgecolor="#c0392b", markersize=11, label="선호 위치"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, -0.03), fontsize=9)
    fig.tight_layout()
    out = OUT_DIR / "placement_scatter.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out}")


#2. 배치 점수 히트맵 (rule-based score 시각화)
def _rule_score(cat: str, rx: float, ry: float) -> float:
    pref_rx, pref_ry = _PREFERRED_POS.get(cat, (0.5, 0.5))
    dist  = ((rx - pref_rx)**2 + (ry - pref_ry)**2) ** 0.5
    score = max(0.0, 1.0 - dist * 2.0)

    if cat == "MONITOR"  and ry > 0.40: score -= 2.0
    if cat == "KEYBOARD" and ry < 0.45: score -= 1.5
    if cat == "DESK_LAMP" and 0.20 <= rx <= 0.80: score -= 1.5
    if cat in ("DECO", "CLOCK") and 0.30 <= rx <= 0.70 and ry > 0.40: score -= 1.0
    if min(rx, 1 - rx, ry, 1 - ry) < 0.05: score -= 0.3
    return float(np.clip(score, -2.5, 1.5))


def fig_score_heatmap(cats: list[str], model=None, feature_names: list | None = None):
    n    = len(cats)
    cols = 4
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.5, rows * 3.5))
    axes = axes.flatten() if n > 1 else [axes]
    title = "배치 점수 히트맵 (Rule-based)"
    if model is not None:
        title = "배치 점수 히트맵 (Learned Ranker)"
    fig.suptitle(title, fontsize=15, fontweight="bold", y=1.01)

    grid_n = 40
    xs = np.linspace(0.05, 0.95, grid_n)
    ys = np.linspace(0.05, 0.95, grid_n)
    XX, YY = np.meshgrid(xs, ys)

    _CAT_ID = {"MONITOR":0,"KEYBOARD":1,"MOUSE":2,"MOUSEPAD":3,"SPEAKER":4,
               "DESK_LAMP":5,"DESK_SHELF":6,"LAPTOP_STAND":7,"DECO":8,"CLOCK":9}

    for i, cat in enumerate(cats):
        ax = axes[i]
        Z  = np.zeros((grid_n, grid_n))

        for gi, rx in enumerate(xs):
            for gj, ry in enumerate(ys):
                if model is not None and feature_names is not None:
                    # 13-dim feature 조립 (model feature_names 기준)
                    feat_map = {
                        "cat_id": _CAT_ID.get(cat, -1),
                        "rx": rx, "ry": ry,
                        "rw": 0.15, "rh": 0.15,
                        "dist_to_preferred": ((rx - _PREFERRED_POS.get(cat, (0.5,0.5))[0])**2 +
                                              (ry - _PREFERRED_POS.get(cat, (0.5,0.5))[1])**2)**0.5,
                        "edge_margin_x": min(rx, 1-rx),
                        "edge_margin_y": min(ry, 1-ry),
                        "monitor_rx":  0.5 if cat != "MONITOR" else -1.0,
                        "monitor_ry":  0.2 if cat != "MONITOR" else -1.0,
                        "keyboard_rx": 0.5 if cat == "MOUSE" else -1.0,
                        "keyboard_ry": 0.62 if cat == "MOUSE" else -1.0,
                        "n_other_objects": 2,
                    }
                    feat_vec = [feat_map.get(fn, 0.0) for fn in feature_names]
                    try:
                        prob = model.predict_proba([feat_vec])[0][1]
                        Z[gj, gi] = (prob - 0.5) * 3.0
                    except Exception:
                        Z[gj, gi] = _rule_score(cat, rx, ry)
                else:
                    Z[gj, gi] = _rule_score(cat, rx, ry)

        im = ax.imshow(Z, extent=[0, 1, 1, 0], origin="upper",
                       cmap="RdYlGn", vmin=-2.0, vmax=1.5, aspect="auto")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        if cat in _PREFERRED_POS:
            px, py = _PREFERRED_POS[cat]
            ax.plot(px, py, marker="*", markersize=14, color="white",
                    markeredgecolor="black", markeredgewidth=0.8, zorder=5)

        ax.set_title(LABEL_KR.get(cat, cat), fontsize=10, fontweight="bold")
        ax.set_xlabel("rx (좌→우)", fontsize=8)
        ax.set_ylabel("ry (뒤→앞)", fontsize=8)
        ax.tick_params(labelsize=7)

    for j in range(len(cats), len(axes)):
        axes[j].set_visible(False)

    fig.tight_layout()
    out = OUT_DIR / "score_heatmap.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out}")


#3. Feature Importance
def fig_feature_importance(ranker: dict):
    try:
        model = ranker["model"]
        fi    = model.feature_importances_
        names = ranker.get("feature_names") or [f"f{i}" for i in range(len(fi))]
        order = np.argsort(fi)[::-1]

        fig, ax = plt.subplots(figsize=(8, max(4, len(fi) * 0.4)))
        ax.barh([names[i] for i in order[::-1]], fi[order[::-1]], color="#3498db", alpha=0.8)
        ax.set_xlabel("Feature Importance (LightGBM gain)", fontsize=11)
        ax.set_title("배치 점수 예측 Feature Importance", fontsize=13, fontweight="bold")
        ax.tick_params(axis="y", labelsize=9)
        ax.grid(axis="x", alpha=0.3)
        fig.tight_layout()
        out = OUT_DIR / "feature_importance.png"
        fig.savefig(out, dpi=180, bbox_inches="tight")
        plt.close(fig)
        print(f"[saved] {out}")
    except Exception as e:
        print(f"[feature_importance] 스킵: {e}")


#4. 선호 위치 vs 실제 positive 비교
def fig_preferred_vs_actual(data: dict, cats: list[str]):
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.set_facecolor("#f0f4f8")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.invert_yaxis()
    ax.set_xlabel("rx (좌 → 우)", fontsize=11)
    ax.set_ylabel("ry (뒤 → 앞)", fontsize=11)
    ax.set_title("선호 위치 vs 실제 Positive", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.25)
    ax.axvline(0.5, color="gray", lw=0.5, ls="--", alpha=0.6)
    ax.axhline(0.5, color="gray", lw=0.5, ls="--", alpha=0.6)

    for cat in cats:
        color = CAT_COLORS.get(cat, "black")
        d = data.get(cat)
        if d is None:
            continue

        # 선호 위치
        if cat in _PREFERRED_POS:
            px, py = _PREFERRED_POS[cat]
            ax.plot(px, py, marker="D", markersize=9, color=color,
                    markeredgecolor="black", markeredgewidth=0.8, alpha=0.9, zorder=4)

        # 실제 positive
        if len(d["pos_rx"]) > 0:
            cx, cy = float(np.mean(d["pos_rx"])), float(np.mean(d["pos_ry"]))
            ax.plot(cx, cy, marker="o", markersize=9, color=color,
                    markeredgecolor="white", markeredgewidth=1.0, alpha=0.9, zorder=4)
            # 연결선
            if cat in _PREFERRED_POS:
                ax.plot([_PREFERRED_POS[cat][0], cx],
                        [_PREFERRED_POS[cat][1], cy],
                        "-", color=color, lw=1.2, alpha=0.5)
            ax.annotate(LABEL_KR.get(cat, cat), (cx, cy),
                        textcoords="offset points", xytext=(6, 4),
                        fontsize=8, color=color, fontweight="bold")

    handles = [
        plt.Line2D([0], [0], marker="D", color="w", markerfacecolor="gray",
                   markeredgecolor="black", markersize=9, label="선호 위치 (rule)"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="gray",
                   markeredgecolor="white", markersize=9, label="실제 positive"),
    ]
    ax.legend(handles=handles, loc="upper right", fontsize=9)
    fig.tight_layout()
    out = OUT_DIR / "preferred_vs_actual.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out}")


#5. 카테고리별 Positive 밀도 KDE 맵
def fig_kde_density(data: dict, cats: list[str]):
    try:
        from scipy.stats import gaussian_kde
    except ImportError:
        print("[kde] scipy 없음 — 스킵")
        return

    n    = len(cats)
    cols = 4
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.5, rows * 3.5))
    axes = axes.flatten() if n > 1 else [axes]
    fig.suptitle("카테고리별 Positive 배치 밀도 (KDE)", fontsize=15, fontweight="bold", y=1.01)

    grid_n = 60
    xs = np.linspace(0, 1, grid_n)
    ys = np.linspace(0, 1, grid_n)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.vstack([XX.ravel(), YY.ravel()])

    for i, cat in enumerate(cats):
        ax = axes[i]
        d  = data[cat]
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.invert_yaxis()

        if len(d["pos_rx"]) >= 5:
            kde = gaussian_kde(np.vstack([d["pos_rx"], d["pos_ry"]]))
            Z   = kde(pts).reshape(grid_n, grid_n)
            ax.imshow(Z, extent=[0, 1, 1, 0], origin="upper",
                      cmap="Blues", aspect="auto", alpha=0.9)
            ax.contour(XX, YY, Z, levels=5, colors="navy", linewidths=0.5, alpha=0.5)
        else:
            ax.text(0.5, 0.5, "데이터 부족", ha="center", va="center",
                    transform=ax.transAxes, fontsize=9, color="gray")

        if cat in _PREFERRED_POS:
            px, py = _PREFERRED_POS[cat]
            ax.plot(px, py, marker="*", markersize=14, color="#f39c12",
                    markeredgecolor="#c0392b", markeredgewidth=0.8, zorder=5)

        ax.set_title(LABEL_KR.get(cat, cat), fontsize=10, fontweight="bold")
        ax.set_xlabel("rx", fontsize=8); ax.set_ylabel("ry", fontsize=8)
        ax.tick_params(labelsize=7)

    for j in range(len(cats), len(axes)):
        axes[j].set_visible(False)

    fig.tight_layout()
    out = OUT_DIR / "kde_density.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out}")


# main
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cats", nargs="+", default=PAPER_CATS, help="시각화할 카테고리 목록")
    parser.add_argument("--no-model", action="store_true", help="모델 로드 없이 rule-based만 시각화")
    args = parser.parse_args()

    cats = [c.upper() for c in args.cats]
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if not DATASET_JSONL.exists():
        print(f"[error] {DATASET_JSONL} 없음 — build_ranker_dataset.py 먼저 실행")
        return

    print(f"[viz] 데이터 로드 중... cats={cats}")
    data = load_dataset(cats)
    total_pos = sum(len(data[c]["pos_rx"]) for c in cats)
    total_neg = sum(len(data[c]["neg_rx"]) for c in cats)
    print(f"[viz] positive={total_pos} negative={total_neg}")

    ranker = None
    model  = None
    feat_names = None
    if not args.no_model and RANKER_PKL.exists():
        try:
            with RANKER_PKL.open("rb") as f:
                ranker = pickle.load(f)
            model      = ranker["model"]
            feat_names = ranker.get("feature_names")
            print(f"[viz] ranker 로드 완료 AUC={ranker.get('val_auc')}")
        except Exception as e:
            print(f"[viz] ranker 로드 실패 (rule-based로 대체): {e}")

    print("[1/5] placement_scatter...")
    fig_placement_scatter(data, cats)

    print("[2/5] score_heatmap...")
    fig_score_heatmap(cats, model=model, feature_names=feat_names)

    print("[3/5] feature_importance...")
    if ranker is not None:
        fig_feature_importance(ranker)
    else:
        print("  → 스킵 (모델 없음)")

    print("[4/5] preferred_vs_actual...")
    fig_preferred_vs_actual(data, cats)

    print("[5/5] kde_density...")
    fig_kde_density(data, cats)

    print(f"\n[done] 모든 그래프 → {OUT_DIR}/")


if __name__ == "__main__":
    main()
