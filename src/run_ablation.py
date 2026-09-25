"""
run_ablation.py
Retrains the selected Random Forest on progressively larger feature sets to
quantify what each data source contributes (RQ1, RQ2).

Everything except the feature set is held constant: the same spatial folds,
the same hyperparameters, the same class weighting, the same random seed.
Any difference in score is therefore attributable to the features added.

Results are written to model_metrics so the figures can be reproduced from the
database rather than from console output.

Run:  PYTHONPATH=src python src/run_ablation.py
"""

import json
import os
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_recall_fscore_support
from sklearn.preprocessing import LabelEncoder

from config import FEAT_MATRIX, ALL_FEATS, N_FOLDS, RANDOM_STATE, RESULTS_DIR
from db_loader import get_conn


def split_feature_groups(all_feats):
    """Derive the three groups by column name, so nothing is hard-coded wrongly."""
    chemistry = [f for f in all_feats if f in ("nitrate_mid", "phosphate_mid")]
    sewage = [f for f in all_feats if f in ("avg_spills", "n_overflows", "spills_per_pipe")]
    land = [f for f in all_feats if f.startswith("lc_")]

    unassigned = set(all_feats) - set(chemistry) - set(sewage) - set(land)
    if unassigned:
        print(f"  WARNING: unassigned features, check the grouping: {sorted(unassigned)}")
    return chemistry, sewage, land


def make_spatial_folds(df, n_folds=N_FOLDS):
    """Identical fold assignment to model_training.py, so results are comparable."""
    km = KMeans(n_clusters=n_folds, random_state=RANDOM_STATE, n_init=10)
    df = df.copy()
    df["fold"] = km.fit_predict(df[["easting", "northing"]].values)
    return df


def evaluate(df, feat_cols, label):
    """Train and evaluate on one feature set. Returns per-fold and pooled results."""
    le = LabelEncoder()
    y = le.fit_transform(df["wfd_status"])
    X = df[feat_cols].fillna(0).values

    fold_scores, all_true, all_pred = [], [], []

    for fold_id in sorted(df["fold"].unique()):
        tr, te = df["fold"] != fold_id, df["fold"] == fold_id

        model = RandomForestClassifier(
            n_estimators=200, max_depth=10, min_samples_leaf=5,
            class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1,
        )
        model.fit(X[tr], y[tr])
        pred = model.predict(X[te])

        fold_f1 = f1_score(y[te], pred, average="weighted")
        fold_scores.append({
            "fold": int(fold_id),
            "n_train": int(tr.sum()),
            "n_test": int(te.sum()),
            "weighted_f1": float(fold_f1),
        })
        all_true.extend(y[te].tolist())
        all_pred.extend(pred.tolist())
        print(f"    fold {fold_id}: {fold_f1:.4f}")

    mean_f1 = float(np.mean([f["weighted_f1"] for f in fold_scores]))
    pooled_f1 = float(f1_score(all_true, all_pred, average="weighted"))
    prec, rec, f1, _ = precision_recall_fscore_support(all_true, all_pred, zero_division=0)

    print(f"  {label}: mean {mean_f1:.4f} | pooled {pooled_f1:.4f}")

    return {
        "label": label,
        "n_features": len(feat_cols),
        "features": feat_cols,
        "fold_scores": fold_scores,
        "mean_weighted_f1": mean_f1,
        "pooled_weighted_f1": pooled_f1,
        "classes": le.classes_.tolist(),
        "per_class": {
            cls: {"precision": float(prec[i]), "recall": float(rec[i]), "f1": float(f1[i])}
            for i, cls in enumerate(le.classes_)
        },
    }


def log_to_db(result, feature_set_name):
    """Write per-fold rows to model_metrics, matching the existing schema."""
    conn = get_conn()
    mod = result["per_class"].get("Moderate", {})
    poor = result["per_class"].get("Poor", {})

    rows = [{
        "model_type": "Random Forest",
        "approach": "AI / Machine Learning",
        "feature_set": feature_set_name,
        "fold_id": f["fold"],
        "n_train": f["n_train"],
        "n_test": f["n_test"],
        "weighted_f1": f["weighted_f1"],
        "mod_precision": mod.get("precision"),
        "mod_recall": mod.get("recall"),
        "mod_f1": mod.get("f1"),
        "poor_precision": poor.get("precision"),
        "poor_recall": poor.get("recall"),
        "poor_f1": poor.get("f1"),
        "recorded_at": datetime.now().isoformat(),
    } for f in result["fold_scores"]]

    pd.DataFrame(rows).to_sql("model_metrics", conn, if_exists="append", index=False)
    conn.commit()
    conn.close()
    print(f"  logged {len(rows)} rows as feature_set='{feature_set_name}'")


if __name__ == "__main__":
    os.makedirs(RESULTS_DIR, exist_ok=True)

    df = pd.read_csv(FEAT_MATRIX)
    df = make_spatial_folds(df)
    print(f"Feature matrix: {len(df):,} rows\n")

    chemistry, sewage, land = split_feature_groups(ALL_FEATS)
    print(f"Chemistry  ({len(chemistry)}): {chemistry}")
    print(f"Sewage     ({len(sewage)}): {sewage}")
    print(f"Land cover ({len(land)}): {len(land)} features\n")

    stages = [
        ("ablation_chemistry", "Chemistry only", chemistry),
        ("ablation_chem_sewage", "Chemistry + sewage", chemistry + sewage),
        ("ablation_full", "Chemistry + sewage + land cover", chemistry + sewage + land),
    ]

    results = []
    for db_name, label, feats in stages:
        print(f"{label} ({len(feats)} features)")
        r = evaluate(df, feats, label)
        log_to_db(r, db_name)
        results.append(r)
        print()

    print("=" * 62)
    print("ABLATION SUMMARY (mean across five spatial folds)")
    print("=" * 62)
    prev = None
    for r in results:
        delta = "" if prev is None else f"   (+{r['mean_weighted_f1'] - prev:.4f})"
        print(f"  {r['label']:34s} {r['mean_weighted_f1']:.4f}{delta}")
        prev = r["mean_weighted_f1"]

    print("\nPooled figures, for comparison:")
    for r in results:
        print(f"  {r['label']:34s} {r['pooled_weighted_f1']:.4f}")

    out = os.path.join(RESULTS_DIR, "ablation_results.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved: {out}")
    print("\nNote: this appends to model_metrics. If you re-run it, delete the")
    print("previous ablation_* rows first to avoid duplicates.")