"""
model_training.py
Trains the selected model on all available historical data.
Reusable for retraining when new annual data arrives.

Usage:
    python src/model_training.py              -- train with default settings
    from model_training import train_model    -- import for reuse

Retraining trigger:
    - New EDM annual returns published (every March)
    - New UKCEH land cover map released (annual)
    - New WFD cycle published (~every 6 years)

Note: model evaluation (which model to use) is a separate one-time step.
      Run model_evaluation.py first, then this script.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import f1_score, classification_report, confusion_matrix
import xgboost as xgb
import joblib
import os
import json
import shutil
from datetime import datetime
from config import (
    FEAT_MATRIX, MODELS_DIR, RESULTS_DIR,
    ALL_FEATS, N_FOLDS, RANDOM_STATE,
    MODEL_PATH, ENCODER_PATH
)


# ---- Spatial folds ----------------------------------------------------------

def make_spatial_folds(df, n_folds=N_FOLDS):
    """
    Split England into n geographic clusters using KMeans on coordinates.
    Keeps nearby points in the same fold - prevents spatial data leakage.
    """
    coords = df[["easting", "northing"]].values
    km = KMeans(n_clusters=n_folds, random_state=RANDOM_STATE, n_init=10)
    df = df.copy()
    df["fold"] = km.fit_predict(coords)
    return df


# ---- Core training function (reusable) -------------------------------------

def train_model(model_type="rf", feat_cols=ALL_FEATS, df=None, retrain=False):
    """
    Train a model using spatial k-fold CV.
    Returns trained model, label encoder, and per-fold metrics.

    model_type: "rf" | "xgb" | "logreg"
    feat_cols:  list of feature column names to use
    df:         feature matrix DataFrame. Loads from DB if None.
    retrain:    if True, backs up existing model.pkl before saving new one.

    Reusable - call this directly for retraining:
        from model_training import train_model
        train_model(model_type="rf", retrain=True)
    """

    if df is None:
        print("Loading feature matrix...")
        df = pd.read_csv(FEAT_MATRIX)

    rename_map = {"edm_total_spill_count": "spill_count", "edm_total_duration_hours": "spill_hrs", "edm_avg_long_term_spills": "avg_spills", "edm_n_overflows": "n_overflows", "lc_pct_woodland_1km": "lc_woodland_1km", "lc_pct_arable_1km": "lc_arable_1km", "lc_pct_grassland_1km": "lc_grass_1km", "lc_pct_wetland_1km": "lc_wetland_1km", "lc_pct_urban_1km": "lc_urban_1km", "lc_pct_freshwater_1km": "lc_water_1km", "lc_pct_woodland_5km": "lc_woodland_5km", "lc_pct_arable_5km": "lc_arable_5km", "lc_pct_grassland_5km": "lc_grass_5km", "lc_pct_wetland_5km": "lc_wetland_5km", "lc_pct_urban_5km": "lc_urban_5km", "lc_pct_freshwater_5km": "lc_water_5km"}
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
    df = make_spatial_folds(df)

    le = LabelEncoder()
    y = le.fit_transform(df["wfd_status"])
    X = df[feat_cols].fillna(0)

    fold_results = []
    all_true, all_pred = [], []
    importances = []

    for fold_id in sorted(df["fold"].unique()):
        tr = df["fold"] != fold_id  # train mask
        te = df["fold"] == fold_id  # test mask

        X_tr, X_te = X[tr].values, X[te].values
        y_tr, y_te = y[tr], y[te]

        model = _build_model(model_type, y_tr)

        if model_type == "logreg":
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_tr)
            X_te = scaler.transform(X_te)

        model.fit(X_tr, y_tr)
        pred = model.predict(X_te)

        fold_f1 = f1_score(y_te, pred, average="weighted")
        fold_results.append({
            "fold": int(fold_id),
            "n_train": int(tr.sum()),
            "n_test": int(te.sum()),
            "weighted_f1": float(fold_f1),
        })
        all_true.extend(y_te.tolist())
        all_pred.extend(pred.tolist())

        # collect feature importances per fold
        if hasattr(model, "feature_importances_"):
            importances.append(model.feature_importances_)
        elif hasattr(model, "coef_"):
            importances.append(np.abs(model.coef_[0]))

        print(f"  Fold {fold_id}: train={tr.sum()}, test={te.sum()}, F1={fold_f1:.4f}")

    # overall metrics across all folds
    overall_f1 = f1_score(all_true, all_pred, average="weighted")
    cm = confusion_matrix(all_true, all_pred)
    report = classification_report(
        all_true, all_pred,
        target_names=le.classes_, output_dict=True
    )

    print(f"\n  Overall weighted F1: {overall_f1:.4f}")
    print(f"  Confusion matrix:\n{cm}")
    print(f"\n{classification_report(all_true, all_pred, target_names=le.classes_)}")

    # mean feature importance across folds
    mean_imp = np.mean(importances, axis=0) if importances else []
    feat_importance = dict(zip(feat_cols, mean_imp.tolist())) if len(mean_imp) else {}

    results = {
        "model_type": model_type,
        "feat_cols": feat_cols,
        "n_folds": N_FOLDS,
        "trained_at": datetime.now().isoformat(),
        "fold_results": fold_results,
        "overall_weighted_f1": float(overall_f1),
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
        "label_classes": le.classes_.tolist(),
        "feat_importance": feat_importance,
    }

    # train final model on ALL data for saving
    final_model = _build_model(model_type, y)
    if model_type == "logreg":
        scaler = StandardScaler()
        final_model.fit(scaler.fit_transform(X.values), y)
    else:
        final_model.fit(X.values, y)

    _save_model(final_model, le, model_type, retrain)

    return final_model, le, results


# ---- Model builder ----------------------------------------------------------

def _build_model(model_type, y_train):
    """Build model instance for a given type."""

    if model_type == "rf":
        return RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_leaf=5,
            class_weight="balanced",  # handles Moderate/Poor imbalance
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
    elif model_type == "xgb":
        # scale_pos_weight handles class imbalance for binary XGBoost
        ratio = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
        return xgb.XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            scale_pos_weight=ratio,
            random_state=RANDOM_STATE,
            eval_metric="logloss",
        )
    elif model_type == "logreg":
        return LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=RANDOM_STATE,
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


# ---- Save model -------------------------------------------------------------

def _save_model(model, le, model_type, retrain=False):
    """
    Save model and label encoder to disk.
    If retrain=True, backs up existing model.pkl with timestamp first.
    """
    os.makedirs(MODELS_DIR, exist_ok=True)

    model_out = os.path.join(MODELS_DIR, f"{model_type}_model.pkl")

    # backup existing model before overwriting on retrain
    if retrain and os.path.exists(model_out):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup = os.path.join(MODELS_DIR, f"{model_type}_model_{ts}.pkl")
        shutil.copy2(model_out, backup)
        print(f"  Existing model backed up: {backup}")

    joblib.dump(model, model_out)
    joblib.dump(le, ENCODER_PATH)
    print(f"  Model saved: {model_out}")
    print(f"  Encoder saved: {ENCODER_PATH}")


# ---- Load model for inference -----------------------------------------------

def load_model(model_type="rf"):
    """
    Load saved model and label encoder for inference.
    Called by inference.py - never calls train_model().
    """
    model_path = os.path.join(MODELS_DIR, f"{model_type}_model.pkl")
    model = joblib.load(model_path)
    le = joblib.load(ENCODER_PATH)
    print(f"  Model loaded: {model_path}")
    return model, le


# ---- Main -------------------------------------------------------------------

if __name__ == "__main__":

    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("=" * 60)
    print("Model training - Random Forest (full feature set)")
    print("=" * 60)

    # load data once, pass to train_model
    df = pd.read_csv(FEAT_MATRIX)
    print(f"Feature matrix: {len(df)} rows, {len(df.columns)} cols")
    print(f"Labels:\n{df['wfd_status'].value_counts()}\n")

    # train with all features - RF is the selected model from model_evaluation.py
    model, le, results = train_model(
        model_type="rf",
        feat_cols=ALL_FEATS,
        df=df,
        retrain=False,  # set True when retraining on new annual data
    )

    # save results
    out = os.path.join(RESULTS_DIR, "training_results.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved: {out}")

    print("\n" + "=" * 60)
    print(f"Training complete")
    print(f"  Model:       Random Forest")
    print(f"  Features:    {len(ALL_FEATS)}")
    print(f"  Weighted F1: {results['overall_weighted_f1']:.4f}")
    print(f"  Saved to:    {MODELS_DIR}/rf_model.pkl")
    print("=" * 60)
