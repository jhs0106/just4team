import json
import pickle
import numpy as np
from pathlib import Path
from collections import Counter

IN_JSONL  = Path("data/placement_ranker_dataset.jsonl")
MODEL_OUT = Path("outputs/models/layout_ranker.pkl")


def main():
    if not IN_JSONL.exists():
        print(f"[error] {IN_JSONL} 없음 — build_ranker_dataset.py 먼저 실행")
        return

    rows = [json.loads(l) for l in IN_JSONL.read_text(encoding="utf-8").splitlines() if l.strip()]
    print(f"[train] {len(rows)}개 샘플 로드")

    feature_names = rows[0]["feature_names"]
    X = np.array([r["features"] for r in rows], dtype=np.float32)
    y = np.array([r["label"]    for r in rows], dtype=np.int32)
    cats = [r["category"] for r in rows]

    print(f"  X.shape={X.shape}  pos={y.sum()}  neg={len(y)-y.sum()}")
    print(f"  카테고리 분포(positive): {Counter(c for c, l in zip(cats, y) if l==1)}")

    from sklearn.model_selection import train_test_split
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )

    try:
        import lightgbm as lgb
        model = lgb.LGBMClassifier(
            n_estimators=400,
            max_depth=5,
            learning_rate=0.05,
            num_leaves=31,
            min_child_samples=10,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            callbacks=[
                lgb.early_stopping(40, verbose=False),
                lgb.log_evaluation(50),
            ],
        )
        model_type = "lightgbm"
        print(f"  [lightgbm] best_iteration={model.best_iteration_}")
    except ImportError:
        print("  [fallback] lightgbm 없음 → RandomForestClassifier 사용")
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(
            n_estimators=300,
            max_depth=8,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
        model.fit(X_tr, y_tr)
        model_type = "randomforest"

    from sklearn.metrics import roc_auc_score, classification_report
    y_prob = model.predict_proba(X_val)[:, 1]
    auc    = roc_auc_score(y_val, y_prob)
    print(f"\n[eval] val AUC = {auc:.4f}")
    print(classification_report(
        y_val, (y_prob > 0.5).astype(int),
        target_names=["negative", "positive"],
    ))

    # 카테고리별 AUC
    val_cats = [cats[i] for i in range(len(cats)) if i >= len(X_tr)]
    print("[eval] 카테고리별 AUC:")
    for cat in sorted(set(val_cats)):
        idx = [i for i, c in enumerate(val_cats) if c == cat]
        if len(idx) < 5:
            continue
        cat_auc = roc_auc_score(y_val[idx], y_prob[idx])
        print(f"  {cat:15s}: AUC={cat_auc:.4f}  n={len(idx)}")

    if hasattr(model, "feature_importances_"):
        fi = sorted(zip(feature_names, model.feature_importances_), key=lambda x: -x[1])
        print("\n[feature importance]")
        for name, imp in fi:
            print(f"  {name:25s}: {imp:.4f}")

    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    with MODEL_OUT.open("wb") as f:
        pickle.dump({
            "model":         model,
            "model_type":    model_type,
            "feature_names": feature_names,
            "val_auc":       round(auc, 4),
        }, f)
    print(f"\n[saved] {MODEL_OUT}")
    print("  → main.py score_region_for_product()에 learned_layout_score 반영 가능")


if __name__ == "__main__":
    main()
