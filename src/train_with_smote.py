"""
train_with_smote.py
Trains Random Forest with SMOTE oversampling on the current 17-feature set.

SMOTE (Synthetic Minority Oversampling Technique, Chawla et al. 2002) is a
well-established, peer-reviewed method for handling class imbalance. It
generates synthetic feature vectors for the minority class (Poor, ~28% of
data) by interpolating between real Poor samples and their nearest
neighbours in feature space.

This is fundamentally different from asking an LLM to invent plausible-
looking data: SMOTE's synthetic points are mathematically constrained to
sit between real observed points, using only the real data's own
statistical structure - it cannot invent patterns that are not already
present in your real dataset.

IMPORTANT: SMOTE is applied only inside each training fold, never to the
test fold - this keeps evaluation metrics honest and prevents synthetic
points from leaking into what the model is scored against.

Compares directly against your existing class_weight='balanced' approach
(F1 = 0.6584) to see whether SMOTE performs better, worse, or the same.

Run:  python src/train_with_smote.py
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import f1_score, classification_report, confusion_matrix
import json
import os
from datetime import datetime
from config import FEAT_MATRIX, ALL_FEATS, N_FOLDS, RANDOM_STATE, RESULTS_DIR

try:
    from imblearn.over_sampling import SMOTE
except ImportError:
    print("Missing dependency. Run: pip install imbalanced-learn")
    raise


def make_spatial_folds(df, n_folds=N_FOLDS):
    """Same spatial fold logic as model_training.py - keeps comparison fair."""
    coords = df[["easting", "northing"]].values
    km = KMeans(n_clusters=n_folds, random_state=RANDOM_STATE, n_init=10)
    df = df.copy()
    df["fold"] = km.fit_predict(coords)
    return df


def train_with_smote(df, feat_cols=ALL_FEATS):
    """
    Train Random Forest with SMOTE applied only to each training fold.
    Test folds always use real, unmodified data.
    """
    print("=" * 60)
    print("Training with SMOTE oversampling")
    print(f"Features: {len(feat_cols)}")
    print("=" * 60)

    df = make_spatial_folds(df)

    le = LabelEncoder()
    y = le.fit_transform(df["wfd_status"])
    X = df[feat_cols].fillna(0).values

    print(f"Original class distribution: "
          f"{dict(zip(le.classes_, np.bincount(y)))}")

    fold_results = []
    all_true, all_pred = [], []

    for fold_id in sorted(df["fold"].unique()):
        tr = df["fold"] != fold_id
        te = df["fold"] == fold_id

        X_tr, X_te = X[tr], X[te]
        y_tr, y_te = y[tr], y[te]

        # SMOTE applied ONLY to training data
        smote = SMOTE(random_state=RANDOM_STATE)
        X_tr_res, y_tr_res = smote.fit_resample(X_tr, y_tr)

        print(f"  Fold {fold_id}: train {len(y_tr)} -> {len(y_tr_res)} after SMOTE "
              f"(balanced to {dict(zip(*np.unique(y_tr_res, return_counts=True)))})")

        model = RandomForestClassifier(
            n_estimators=200, max_depth=10, min_samples_leaf=5,
            random_state=RANDOM_STATE, n_jobs=-1
            # no class_weight needed - SMOTE already balances the classes
        )
        model.fit(X_tr_res, y_tr_res)
        pred = model.predict(X_te)  # predict on REAL test data only

        fold_f1 = f1_score(y_te, pred, average="weighted")
        fold_results.append({
            "fold": int(fold_id),
            "n_train_original": int(len(y_tr)),
            "n_train_smote": int(len(y_tr_res)),
            "n_test": int(te.sum()),
            "weighted_f1": float(fold_f1),
        })
        all_true.extend(y_te.tolist())
        all_pred.extend(pred.tolist())

        print(f"    F1 = {fold_f1:.4f}")

    overall_f1 = f1_score(all_true, all_pred, average="weighted")
    cm = confusion_matrix(all_true, all_pred)
    report = classification_report(
        all_true, all_pred, target_names=le.classes_, output_dict=True
    )

    print(f"\nOverall weighted F1 (SMOTE): {overall_f1:.4f}")
    print(f"Confusion matrix:\n{cm}")
    print(f"\n{classification_report(all_true, all_pred, target_names=le.classes_)}")

    return {
        "method": "Random Forest with SMOTE",
        "feat_cols": feat_cols,
        "trained_at": datetime.now().isoformat(),
        "fold_results": fold_results,
        "overall_weighted_f1": float(overall_f1),
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
        "label_classes": le.classes_.tolist(),
    }


def compare_to_baseline(smote_results, baseline_f1=0.6584):
    """Compare against your current production model (class_weight='balanced')."""
    print("\n" + "=" * 60)
    print("SMOTE vs CURRENT PRODUCTION MODEL")
    print("=" * 60)

    smote_f1 = smote_results["overall_weighted_f1"]
    diff = smote_f1 - baseline_f1

    print(f"  Current model (class_weight='balanced'): F1 = {baseline_f1:.4f}")
    print(f"  SMOTE-trained model:                     F1 = {smote_f1:.4f}")
    print(f"  Difference:                              {diff:+.4f}")

    if diff > 0.01:
        print("\n  SMOTE improves performance meaningfully - consider adopting it.")
    elif diff < -0.01:
        print("\n  SMOTE performs worse - stick with class_weight='balanced'.")
    else:
        print("\n  No meaningful difference - either is defensible; class_weight")
        print("  is simpler and already in production, so no change needed.")

    return diff


if __name__ == "__main__":
    os.makedirs(RESULTS_DIR, exist_ok=True)

    df = pd.read_csv(FEAT_MATRIX)
    print(f"Feature matrix: {len(df)} rows")
    print(f"Labels:\n{df['wfd_status'].value_counts()}\n")

    results = train_with_smote(df)
    diff = compare_to_baseline(results)
    results["comparison_vs_baseline"] = diff

    out = os.path.join(RESULTS_DIR, "smote_results.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nResults saved: {out}")
    print("\nNote: this is a comparison experiment. It does NOT overwrite")
    print("your production model.pkl. Only adopt SMOTE by updating")
    print("model_training.py if it genuinely improves F1.")