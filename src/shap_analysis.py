"""
shap_analysis.py
SHAP explainability analysis on the trained Random Forest model.
Run once after model_training.py. Re-run after retraining.

Outputs:
    results/shap_global_importance.json  - mean |SHAP| per feature
    results/shap_values.npy              - full SHAP matrix for dashboard
    results/shap_input_features.csv      - feature values aligned to SHAP matrix
    results/shap_fold_stability.csv      - SHAP rank correlation across folds
    results/shap_stability_summary.json  - mean Spearman correlation (RQ4 answer)

Addresses RQ4:
    Which features contribute most to predictions?
    Are those attributions stable across geographic regions?
"""

import pandas as pd
import numpy as np
import shap
import joblib
import json
import os
from datetime import datetime
from sklearn.cluster import KMeans
from config import (
    FEAT_MATRIX, MODELS_DIR, RESULTS_DIR,
    ALL_FEATS, N_FOLDS, RANDOM_STATE,
    ENCODER_PATH
)


# ---- Rename map for old column names ----------------------------------------
# Remove once feature_matrix.csv is regenerated with new short names
RENAME_MAP = {
    "edm_total_spill_count":    "spill_count",
    "edm_total_duration_hours": "spill_hrs",
    "edm_avg_long_term_spills": "avg_spills",
    "edm_n_overflows":          "n_overflows",
    "lc_pct_woodland_1km":      "lc_woodland_1km",
    "lc_pct_arable_1km":        "lc_arable_1km",
    "lc_pct_grassland_1km":     "lc_grass_1km",
    "lc_pct_wetland_1km":       "lc_wetland_1km",
    "lc_pct_urban_1km":         "lc_urban_1km",
    "lc_pct_freshwater_1km":    "lc_water_1km",
    "lc_pct_woodland_5km":      "lc_woodland_5km",
    "lc_pct_arable_5km":        "lc_arable_5km",
    "lc_pct_grassland_5km":     "lc_grass_5km",
    "lc_pct_wetland_5km":       "lc_wetland_5km",
    "lc_pct_urban_5km":         "lc_urban_5km",
    "lc_pct_freshwater_5km":    "lc_water_5km",
}


# ---- Load data and model ----------------------------------------------------

def load_data():
    df = pd.read_csv(FEAT_MATRIX)
    df = df.rename(columns={k: v for k, v in RENAME_MAP.items() if k in df.columns})
    print(f"Loaded feature matrix: {len(df)} rows")
    return df


def load_model():
    model_path = os.path.join(MODELS_DIR, "rf_model.pkl")
    model = joblib.load(model_path)
    le = joblib.load(ENCODER_PATH)
    print(f"Loaded model: {model_path}")
    return model, le


# ---- Global SHAP analysis ---------------------------------------------------

def global_shap(model, df, le):
    """
    Compute SHAP values on full dataset using trained model.
    Shows which features matter most overall across all predictions.
    """
    print("\nComputing global SHAP values...")
    X = df[ALL_FEATS].fillna(0)

    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(X)

    # Extract SHAP values for the "Poor" class specifically.
    # Class index must come from the label encoder, not assumed to be 1.
    classes = list(le.classes_)
    poor_idx = classes.index("Poor")
    print(f"  Label classes: {classes}, using index {poor_idx} for Poor")

    if isinstance(shap_vals, list):
        vals_poor = shap_vals[poor_idx]
    elif isinstance(shap_vals, np.ndarray) and shap_vals.ndim == 3:
        vals_poor = shap_vals[:, :, poor_idx]
    else:
        # binary case where SHAP returns a single 2D array - values already
        # represent the positive class (index 1). Flip sign if Poor is index 0.
        vals_poor = shap_vals if poor_idx == 1 else -shap_vals

    # sanity check - more sewage should push toward Poor (positive SHAP)
    if "spill_hrs" in X.columns:
        j = list(X.columns).index("spill_hrs")
        r = np.corrcoef(X["spill_hrs"].values, vals_poor[:, j])[0, 1]
        print(f"  Sanity check - corr(spill_hrs, its SHAP) = {r:+.3f} "
              f"({'OK' if r > 0 else 'WARNING: signs may be inverted'})")

    mean_abs = np.abs(vals_poor).mean(axis=0)
    importance = dict(sorted(
        zip(ALL_FEATS, mean_abs.tolist()),
        key=lambda x: x[1], reverse=True
    ))

    print("\nGlobal SHAP feature importance (mean |SHAP|):")
    for feat, val in importance.items():
        bar = "#" * int(val * 300)
        print(f"  {feat:30s}: {val:.4f}  {bar}")

    # save for dashboard use
    np.save(os.path.join(RESULTS_DIR, "shap_values.npy"), vals_poor)
    X.to_csv(os.path.join(RESULTS_DIR, "shap_input_features.csv"), index=False)

    return vals_poor, importance, X, explainer


# ---- SHAP stability across spatial folds ------------------------------------

def fold_stability(df, n_folds=N_FOLDS):
    """
    Train one RF per spatial fold, compute SHAP importance for each.
    Spearman rank correlation shows whether feature rankings are
    consistent across different geographic regions of England.
    High correlation = stable explanations = trustworthy SHAP (RQ4).
    """
    print("\nChecking SHAP stability across spatial folds...")

    # create spatial folds
    coords = df[["easting", "northing"]].values
    km = KMeans(n_clusters=n_folds, random_state=RANDOM_STATE, n_init=10)
    df = df.copy()
    df["fold"] = km.fit_predict(coords)

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import LabelEncoder

    le = LabelEncoder()
    y_all = le.fit_transform(df["wfd_status"])
    X_all = df[ALL_FEATS].fillna(0)

    fold_importances = {}

    for fold_id in sorted(df["fold"].unique()):
        tr = df["fold"] != fold_id
        X_tr = X_all[tr].values
        y_tr = y_all[tr]

        model = RandomForestClassifier(
            n_estimators=200, max_depth=10, min_samples_leaf=5,
            class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1
        )
        model.fit(X_tr, y_tr)

        explainer = shap.TreeExplainer(model)
        shap_vals = explainer.shap_values(X_all[tr].values)
        vals = shap_vals[1] if isinstance(shap_vals, list) else shap_vals

        mean_abs = np.abs(vals).mean(axis=0)
        fold_importances[f"fold_{fold_id}"] = dict(zip(ALL_FEATS, mean_abs.tolist()))
        print(f"  Fold {fold_id} done")

    # Spearman rank correlation across folds
    imp_df = pd.DataFrame(fold_importances)
    rank_corr = imp_df.rank(ascending=False).corr(method="spearman")

    # mean pairwise correlation (upper triangle only, no diagonal)
    upper = rank_corr.values[np.triu_indices_from(rank_corr.values, k=1)]
    mean_corr = float(upper.mean())

    print(f"\nSpearman rank correlation across folds:")
    print(rank_corr.round(3))
    print(f"\nMean pairwise Spearman correlation: {mean_corr:.4f}")
    print("(1.0 = perfectly stable, 0.0 = no consistency across regions)")

    return fold_importances, rank_corr, mean_corr


# ---- Main -------------------------------------------------------------------

if __name__ == "__main__":

    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("=" * 60)
    print("SHAP analysis")
    print("=" * 60)

    df = load_data()
    model, le = load_model()

    # global SHAP - which features matter most overall
    shap_vals, importance, X, explainer = global_shap(model, df, le)

    # fold stability - are explanations consistent across England
    fold_imp, rank_corr, mean_corr = fold_stability(df)

    # save all results
    with open(os.path.join(RESULTS_DIR, "shap_global_importance.json"), "w") as f:
        json.dump(importance, f, indent=2)

    rank_corr.to_csv(os.path.join(RESULTS_DIR, "shap_fold_stability.csv"))

    summary = {
        "analysed_at": datetime.now().isoformat(),
        "model": "rf_model.pkl",
        "n_samples": len(df),
        "n_features": len(ALL_FEATS),
        "top_5_features": list(importance.keys())[:5],
        "shap_fold_stability_mean_spearman": mean_corr,
        "interpretation": (
            "High stability (>0.7) means SHAP feature rankings are consistent "
            "across geographic regions of England - trustworthy explanations."
            if mean_corr > 0.7
            else "Moderate stability - feature importance varies across regions."
        ),
        "global_importance": importance,
        "fold_importances": fold_imp,
    }

    with open(os.path.join(RESULTS_DIR, "shap_stability_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print("\n" + "=" * 60)
    print("SHAP analysis complete")
    print("=" * 60)
    print(f"  Top 5 features: {list(importance.keys())[:5]}")
    print(f"  Fold stability (mean Spearman): {mean_corr:.4f}")
    print(f"  Outputs saved to: {RESULTS_DIR}/")
    print("\nNext step: python src/inference.py")