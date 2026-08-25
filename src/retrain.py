'''retrain.py
Retrains the model on whatever is currently in feat_matrix table.

Correct workflow:
    1. Pull/download latest data (scheduler or manual)
    2. python src/feature_engineering.py --mode append  (or full)
    3. python src/retrain.py

Retraining trigger events:
    - New EDM annual returns published (every March)
    - New UKCEH land cover map released (annual)
    - New WFD cycle published (~every 6 years)
    - Significant FWW data accumulated via weekly scheduler

Backs up old model.pkl with timestamp before saving new one.
Re-runs SHAP after retraining to update explanations.
'''

import pandas as pd
import os
import json
import subprocess
import argparse
from datetime import datetime
from config import (
    MODELS_DIR, RESULTS_DIR, FEAT_MATRIX,
    ALL_FEATS, DB_PATH
)
from db_loader import get_feat_matrix, log_run


# ---- Load feature matrix from DB --------------------------------------------

def load_matrix(source="db"):
    """
    Load feature matrix for retraining.
    source="db"  - read from feat_matrix table (includes all appended data)
    source="csv" - fallback to FEAT_MATRIX csv file
    """
    if source == "db":
        print("Loading feature matrix from DB...")
        df = get_feat_matrix()
    else:
        print(f"Loading feature matrix from CSV: {FEAT_MATRIX}")
        df = pd.read_csv(FEAT_MATRIX)

    print(f"  {len(df)} rows")
    print(f"  Date range: {df['sample_date'].min()} to {df['sample_date'].max()}")
    print(f"  Labels:\n{df['wfd_status'].value_counts()}")
    return df


# ---- Run retrain ------------------------------------------------------------

def run_retrain(model_type="rf", source="db"):
    """
    Retrain model on current feat_matrix.
    Backs up old model, trains new one, updates SHAP.
    """
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("=" * 60)
    print(f"Retraining - {model_type.upper()}")
    print("=" * 60)

    # load latest feature matrix
    matrix = load_matrix(source=source)

    if matrix.empty:
        print("Feature matrix is empty - run feature_engineering.py first")
        log_run("retrain", 0, 0, "failed", "Empty feature matrix")
        return

    # retrain - backs up old model.pkl automatically
    from model_training import train_model

    model, le, results = train_model(
        model_type=model_type,
        feat_cols=ALL_FEATS,
        df=matrix,
        retrain=True,  # triggers backup of old model before saving
    )

    # save training results
    out = os.path.join(RESULTS_DIR, "training_results.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2, default=str)

    # update SHAP explanations
    print("\nUpdating SHAP analysis...")
    subprocess.run(["python", "src/shap_analysis.py"], check=True)

    log_run("retrain", len(matrix), len(matrix), "success")

    print("\n" + "=" * 60)
    print("Retraining complete")
    print(f"  Model:       {model_type}")
    print(f"  Rows used:   {len(matrix)}")
    print(f"  Weighted F1: {results['overall_weighted_f1']:.4f}")
    print(f"  Results:     {out}")
    print("=" * 60)

    return results


# ---- Main -------------------------------------------------------------------

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Retrain model on latest feat_matrix")
    parser.add_argument("--model",  type=str, default="rf",  help="rf / xgb / logreg")
    parser.add_argument("--source", type=str, default="db",  help="db or csv")
    args = parser.parse_args()

    run_retrain(model_type=args.model, source=args.source)
