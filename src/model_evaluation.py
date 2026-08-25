"""
model_evaluation.py
One-time model selection script. Run before model_training.py.

Compares:
    - 4 traditional threshold rules (EA WFD regulatory standards)
    - 3 ML models via spatial 5-fold CV (LogReg, RF, XGBoost)

Ablation experiments show marginal contribution of each data source.
Results saved to DB and JSON. Never run again in production.

Run order:
    1. python src/model_evaluation.py   -- pick best model
    2. python src/model_training.py     -- train it on all data
    3. python src/shap_analysis.py      -- explain it
"""

import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import os
import json
from datetime import datetime
from config import (
    FEAT_MATRIX, RESULTS_DIR,
    ALL_FEATS, CHEM_FEATS, SEWAGE_FEATS,
    PO4_THRESHOLD, NO3_THRESHOLD,
    ABLATION_SETS
)
from model_training import train_model, make_spatial_folds
from db_loader import load_model_metrics


# ---- Traditional threshold rules --------------------------------------------

def eval_threshold(df, rule_name, rule_fn):
    """
    Apply a fixed threshold rule to every sample.
    Same metrics as ML models for fair comparison.
    rule_fn: takes a DataFrame row, returns "Poor" or "Moderate"
    """

    y_true = (df["wfd_status"] == "Poor").astype(int).values
    y_pred = df.apply(rule_fn, axis=1).map({"Poor": 1, "Moderate": 0}).values

    f1 = f1_score(y_true, y_pred, average="weighted")
    report = classification_report(
        y_true, y_pred,
        target_names=["Moderate", "Poor"],
        output_dict=True, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred)

    print(f"\n  {rule_name}")
    print(f"    Weighted F1:    {f1:.4f}")
    print(f"    Moderate F1:    {report['Moderate']['f1-score']:.4f}  "
          f"(P={report['Moderate']['precision']:.3f}, R={report['Moderate']['recall']:.3f})")
    print(f"    Poor F1:        {report['Poor']['f1-score']:.4f}  "
          f"(P={report['Poor']['precision']:.3f}, R={report['Poor']['recall']:.3f})")
    print(f"    Confusion matrix:\n    {cm}")

    return {
        "method": rule_name,
        "approach": "Traditional Rule-Based",
        "weighted_f1": round(f1, 4),
        "moderate_precision": round(report["Moderate"]["precision"], 4),
        "moderate_recall":    round(report["Moderate"]["recall"], 4),
        "moderate_f1":        round(report["Moderate"]["f1-score"], 4),
        "poor_precision":     round(report["Poor"]["precision"], 4),
        "poor_recall":        round(report["Poor"]["recall"], 4),
        "poor_f1":            round(report["Poor"]["f1-score"], 4),
        "confusion_matrix":   cm.tolist(),
    }


def run_traditional(df):
    """Run 4 threshold rules using EA UKTAG regulatory limits."""

    print("=" * 60)
    print("TRADITIONAL THRESHOLD METHODS")
    print(f"  PO4 threshold: > {PO4_THRESHOLD} mg/L -> Poor")
    print(f"  NO3 threshold: > {NO3_THRESHOLD} mg/L -> Poor")
    print("=" * 60)

    return [
        eval_threshold(
            df, f"Rule 1 - Phosphate only (PO4 > {PO4_THRESHOLD})",
            lambda r: "Poor" if r["phosphate_mid"] > PO4_THRESHOLD else "Moderate"
        ),
        eval_threshold(
            df, f"Rule 2 - Nitrate only (NO3 > {NO3_THRESHOLD})",
            lambda r: "Poor" if r["nitrate_mid"] > NO3_THRESHOLD else "Moderate"
        ),
        eval_threshold(
            df, "Rule 3 - OR rule (PO4 > 0.1 OR NO3 > 2.0)",
            lambda r: "Poor" if (
                r["phosphate_mid"] > PO4_THRESHOLD or
                r["nitrate_mid"] > NO3_THRESHOLD
            ) else "Moderate"
        ),
        eval_threshold(
            df, "Rule 4 - AND rule (PO4 > 0.1 AND NO3 > 2.0)",
            lambda r: "Poor" if (
                r["phosphate_mid"] > PO4_THRESHOLD and
                r["nitrate_mid"] > NO3_THRESHOLD
            ) else "Moderate"
        ),
    ]


# ---- ML model comparison ----------------------------------------------------

def run_ml_comparison(df):
    """
    Train and evaluate all three ML models using spatial 5-fold CV.
    Uses train_model() from model_training.py - same hyperparameters as production.
    """

    print("\n" + "=" * 60)
    print("ML MODEL COMPARISON (Spatial 5-Fold CV)")
    print("=" * 60)

    ml_results = {}
    for mtype, label in [
        ("logreg", "Logistic Regression"),
        ("rf",     "Random Forest"),
        ("xgb",    "XGBoost"),
    ]:
        print(f"\n  --- {label} ---")
        _, _, res = train_model(model_type=mtype, feat_cols=ALL_FEATS, df=df)
        ml_results[mtype] = res

    return ml_results


# ---- Ablation experiments ---------------------------------------------------

def run_ablation(df, model_type="rf"):
    """
    Test marginal contribution of each data source.
    Uses the best ML model type identified in run_ml_comparison().
    Directly answers RQ1 and RQ2.
    """

    print("\n" + "=" * 60)
    print(f"ABLATION EXPERIMENTS - {model_type.upper()}")
    print("=" * 60)

    ablation = {}
    prev_f1 = None

    for set_name, feat_cols in ABLATION_SETS.items():
        print(f"\n  Feature set: {set_name} ({len(feat_cols)} features)")
        _, _, res = train_model(model_type=model_type, feat_cols=feat_cols, df=df)
        ablation[set_name] = res

        f1 = res["overall_weighted_f1"]
        gain = f" (+{f1 - prev_f1:.4f})" if prev_f1 is not None else ""
        print(f"  F1 = {f1:.4f}{gain}")
        prev_f1 = f1

    print("\n  ABLATION SUMMARY:")
    for s, r in ablation.items():
        print(f"    {s:30s}: F1 = {r['overall_weighted_f1']:.4f}")

    return ablation


# ---- Final comparison table -------------------------------------------------

def print_comparison(trad_results, ml_results):
    """Print traditional vs ML side by side. Shows why AI beats rule-based."""

    print("\n" + "=" * 60)
    print("TRADITIONAL vs ML - FINAL COMPARISON")
    print("=" * 60)

    rows = []

    for r in trad_results:
        rows.append({
            "Method":        r["method"],
            "Approach":      "Traditional",
            "Weighted F1":   r["weighted_f1"],
            "Mod F1":        r["moderate_f1"],
            "Poor F1":       r["poor_f1"],
            "Poor Recall":   r["poor_recall"],
        })

    labels = {"logreg": "Logistic Regression", "rf": "Random Forest", "xgb": "XGBoost"}
    for key, res in ml_results.items():
        rep = res["classification_report"]
        rows.append({
            "Method":        labels[key],
            "Approach":      "AI / ML",
            "Weighted F1":   round(res["overall_weighted_f1"], 4),
            "Mod F1":        round(rep["Moderate"]["f1-score"], 4),
            "Poor F1":       round(rep["Poor"]["f1-score"], 4),
            "Poor Recall":   round(rep["Poor"]["recall"], 4),
        })

    df = pd.DataFrame(rows)
    print(df.to_string(index=False))

    # summary
    best_trad = max(trad_results, key=lambda x: x["weighted_f1"])
    best_ml_key = max(ml_results, key=lambda k: ml_results[k]["overall_weighted_f1"])
    best_ml_f1 = ml_results[best_ml_key]["overall_weighted_f1"]

    gain = best_ml_f1 - best_trad["weighted_f1"]
    gain_pct = (gain / best_trad["weighted_f1"]) * 100

    print(f"\n  Best traditional: {best_trad['method']} (F1={best_trad['weighted_f1']:.4f})")
    print(f"  Best ML:          {labels[best_ml_key]} (F1={best_ml_f1:.4f})")
    print(f"  AI improvement:   +{gain:.4f} ({gain_pct:.1f}% relative)")
    print(f"\n  Selected model for production: {labels[best_ml_key]}")

    return best_ml_key, df


# ---- Main -------------------------------------------------------------------

if __name__ == "__main__":

    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("=" * 60)
    print("Model evaluation - one-time run")
    print("=" * 60)

    # load data once
    df = pd.read_csv(FEAT_MATRIX)
    print(f"Feature matrix: {len(df)} rows")
    print(f"Labels:\n{df['wfd_status'].value_counts()}\n")

    # stage 1 - traditional methods
    trad = run_traditional(df)

    # stage 2 - ML comparison
    ml = run_ml_comparison(df)

    # stage 3 - pick best ML model
    best_model, comparison_df = print_comparison(trad, ml)

    # stage 4 - ablation on best model
    ablation = run_ablation(df, model_type=best_model)

    # save all results
    all_results = {
        "evaluated_at": datetime.now().isoformat(),
        "best_model": best_model,
        "traditional": trad,
        "ml_comparison": {k: v for k, v in ml.items()},
        "ablation": ablation,
    }

    out = os.path.join(RESULTS_DIR, "modelling_results.json")
    with open(out, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    comparison_df.to_csv(
        os.path.join(RESULTS_DIR, "traditional_vs_ml.csv"), index=False
    )
    load_model_metrics() #loads results into model_metrics table

    print(f"\nResults saved: {out}")
    print(f"\nNext step: python src/model_training.py")
