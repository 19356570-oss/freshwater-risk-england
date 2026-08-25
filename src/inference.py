"""
inference.py
Generates predictions using the saved trained model.
Called by scheduler.py after each data load - never retrains.

Two modes:
    historical - predict on full feature matrix (run once after training)
    live       - predict on latest staging data (called by APScheduler)

Outputs saved to predictions table in SQLite.
"""

import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime
from config import (
    MODELS_DIR, RESULTS_DIR, ENCODER_PATH,
    ALL_FEATS, DATA_MODE, DB_PATH
)
from db_loader import (
    get_conn, get_feat_matrix,
    save_prediction, log_run
)


# ---- Rename map for old column names ----------------------------------------
# remove once feature_matrix.csv regenerated with new short names
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
    "waterbody_id":             "wb_id",
    "nearest_wfd_dist_m":       "wfd_dist_m",
    "wfd_match_quality":        "match_q",
    "river_basin_district":     "rbd",
}


# ---- Load model -------------------------------------------------------------

def load_model(model_type="rf"):
    """Load saved model and label encoder from disk."""
    path = os.path.join(MODELS_DIR, f"{model_type}_model.pkl")
    model = joblib.load(path)
    le = joblib.load(ENCODER_PATH)
    print(f"  Model loaded: {path}")
    return model, le


# ---- Build live feature row -------------------------------------------------

def build_live_features(db_path=DB_PATH):
    """
    Build feature rows from latest staging data for live inference.
    Uses same aggregation logic as feature_engineering.py to ensure
    consistency between training features and live inference features.

    Called by APScheduler after each EDM live or FWW update.
    TODO: complete implementation in Sprint 4 (dashboard build).
    """
    conn = get_conn(db_path)

    # get latest FWW records (last 7 days)
    fww = pd.read_sql("""
        SELECT fww_id, site_name, easting, northing,
               nitrate_mid, phosphate_mid, county, rbd
        FROM staging_fww
        WHERE received_at >= datetime("now", "-7 days")
    """, conn)

    # get latest EDM live aggregated per waterbody
    edm_live = pd.read_sql("""
        SELECT wb_id,
               SUM(duration_hrs) as spill_hrs,
               COUNT(*)          as n_overflows
        FROM staging_edm_live
        WHERE received_at >= datetime("now", "-1 day")
        GROUP BY wb_id
    """, conn)

    # get current land cover features
    lc = pd.read_sql("""
        SELECT fww_id,
               lc_woodland_1km, lc_arable_1km, lc_grass_1km,
               lc_wetland_1km,  lc_urban_1km,  lc_water_1km,
               lc_woodland_5km, lc_arable_5km, lc_grass_5km,
               lc_wetland_5km,  lc_urban_5km,  lc_water_5km
        FROM staging_lc
    """, conn)

    conn.close()

    print(f"  Live features: {len(fww)} FWW records, {len(edm_live)} EDM waterbodies")
    return fww, edm_live, lc


# ---- Run inference ----------------------------------------------------------

def run_inference(model, le, df, data_source="historical", model_version="rf_model"):
    """
    Run predictions on a feature DataFrame.
    Same function used for historical batch and live inference.
    Saves each prediction to the predictions table.
    """
    print(f"Running inference ({data_source}, {len(df)} rows)...")

    df = df.rename(columns={k: v for k, v in RENAME_MAP.items() if k in df.columns})

    # keep only features the model was trained on
    feat_cols = [c for c in ALL_FEATS if c in df.columns]
    X = df[feat_cols].fillna(0).values

    pred_encoded = model.predict(X)
    pred_proba = model.predict_proba(X)  # [prob_moderate, prob_poor]
    pred_labels = le.inverse_transform(pred_encoded)

    # class order from label encoder
    classes = list(le.classes_)
    mod_idx = classes.index("Moderate") if "Moderate" in classes else 0
    poor_idx = classes.index("Poor") if "Poor" in classes else 1

    results = []
    for i, row in df.iterrows():
        pred_row = {
            "fww_id":           row.get("fww_id"),
            "site_name":        row.get("site_name"),
            "easting":          row.get("easting"),
            "northing":         row.get("northing"),
            "wb_id":            row.get("wb_id"),
            "predicted_status": pred_labels[len(results)],
            "prob_moderate":    round(float(pred_proba[len(results)][mod_idx]), 4),
            "prob_poor":        round(float(pred_proba[len(results)][poor_idx]), 4),
            "model_version":    model_version,
            "data_source":      data_source,
            "predicted_at":     datetime.now().isoformat(),
        }
        results.append(pred_row)
        save_prediction(pred_row)

    print(f"  {len(results)} predictions saved to DB")
    print(f"  Poor: {sum(1 for r in results if r['predicted_status'] == 'Poor')}")
    print(f"  Moderate: {sum(1 for r in results if r['predicted_status'] == 'Moderate')}")

    log_run(f"inference_{data_source}", len(results), len(results), "success")
    return pd.DataFrame(results)


# ---- Historical batch inference ---------------------------------------------

def run_historical_inference(model_type="rf"):
    """
    Predict on full historical feature matrix.
    Run once after model_training.py to populate predictions table.
    """
    print("=" * 60)
    print("Historical batch inference")
    print("=" * 60)

    model, le = load_model(model_type)
    df = get_feat_matrix()  # reads from feat_matrix table in SQLite

    return run_inference(
        model, le, df,
        data_source="historical",
        model_version=f"{model_type}_model"
    )


# ---- Live inference (called by scheduler) -----------------------------------

def run_live_inference(model_type="rf"):
    """
    Predict on latest live data from staging tables.
    Called by APScheduler after each FWW/EDM update.
    TODO: complete in Sprint 4 once build_live_features() is implemented.
    """
    print("Live inference...")
    model, le = load_model(model_type)

    # fww, edm_live, lc = build_live_features()
    # df = merge_live_features(fww, edm_live, lc)
    # return run_inference(model, le, df, data_source="live")

    print("  Live inference not yet implemented - Sprint 4")
    log_run("inference_live", 0, 0, "skipped")


# ---- Main -------------------------------------------------------------------

if __name__ == "__main__":

    # run historical batch predictions
    # change to run_live_inference() when dashboard is deployed
    results = run_historical_inference(model_type="rf")

    print("\n" + "=" * 60)
    print("Inference complete")
    print(f"  Total predictions: {len(results)}")
    print(f"  Poor:     {(results['predicted_status'] == 'Poor').sum()}")
    print(f"  Moderate: {(results['predicted_status'] == 'Moderate').sum()}")
    print("=" * 60)
