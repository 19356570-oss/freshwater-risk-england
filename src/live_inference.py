"""
live_inference.py
Scheduled job: re-predicts locations affected by fresh POOPy sewage data.

Does NOT retrain the model or touch chemistry/land-cover features -
those only change on manual retrain (see PIPELINE_RUN_ORDER.md).
Only recomputes spills_per_pipe etc for waterbodies with new live EDM
data, then re-runs the saved model on just those affected points.

Called by scheduler.py after each POOPy pull.
"""

import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta
from config import MODELS_DIR, ENCODER_PATH, ALL_FEATS
from db_loader import get_conn, save_prediction, log_run


def get_waterbodies_with_new_live_data(since_minutes=None):
    """
    Which waterbodies got new POOPy records since the last successful
    live_inference run - not a fixed rolling window, since Codespaces
    stops between sessions and a fixed 20-min window misses data that
    arrived days ago while the scheduler was not running.

    since_minutes: if given, overrides auto-detection with a fixed window
                   (kept for manual testing). If None, uses the timestamp
                   of the last successful live_inference run from
                   ingestion_log, or all available data if never run before.
    """
    conn = get_conn()

    if since_minutes is not None:
        cutoff = (datetime.now() - timedelta(minutes=since_minutes)).isoformat()
    else:
        # only count runs that actually processed data - "no_new_data" runs
        # do not advance the cutoff, otherwise an empty run permanently
        # hides genuinely new data behind an ever-later timestamp
        last_run = pd.read_sql("""
            SELECT MAX(run_at) as last_run
            FROM ingestion_log
            WHERE source = "live_inference"
              AND status = "success"
              AND records_new > 0
        """, conn).iloc[0]["last_run"]

        if last_run:
            cutoff = last_run
            print(f"  Checking for data since last successful run: {cutoff}")
        else:
            cutoff = "2000-01-01T00:00:00"  # no prior run - take everything
            print("  No prior successful run found - checking all available live data")

    wb_ids = pd.read_sql(f"""
        SELECT DISTINCT wb_id
        FROM staging_edm_live
        WHERE received_at >= "{cutoff}"
          AND wb_id IS NOT NULL
    """, conn)
    conn.close()

    return wb_ids["wb_id"].tolist()


def recompute_live_edm_features(wb_ids, window_days=365):
    """
    Aggregate live POOPy data into the same spills_per_pipe schema used
    in training, for just the affected waterbodies. Matches the logic
    in feature_engineering.py exactly, so live and historical features
    stay comparable.
    """
    if not wb_ids:
        return pd.DataFrame()

    conn = get_conn()
    placeholders = ",".join("?" * len(wb_ids))
    cutoff = (datetime.now() - timedelta(days=window_days)).isoformat()

    live = pd.read_sql(f"""
        SELECT wb_id, outlet_ngr, status, duration_hrs
        FROM staging_edm_live
        WHERE wb_id IN ({placeholders})
          AND received_at >= ?
    """, conn, params=wb_ids + [cutoff])
    conn.close()

    if live.empty:
        return pd.DataFrame()

    agg = live.groupby("wb_id").agg(
        spill_count=("status", lambda x: x.str.contains("Discharge", case=False, na=False).sum()),
        n_overflows=("outlet_ngr", "nunique"),
    ).reset_index()

    agg["spills_per_pipe"] = agg["spill_count"] / agg["n_overflows"].clip(lower=1)
    # avg_spills kept from historical - live window too short for a stable long-term average
    return agg[["wb_id", "spills_per_pipe", "n_overflows"]]


def run_live_inference():
    """
    Main entry point. Finds affected waterbodies, recomputes their
    sewage features, re-predicts just those points, saves as data_source='live'.
    """
    print("Checking for waterbodies with new live EDM data...")
    wb_ids = get_waterbodies_with_new_live_data()

    if not wb_ids:
        print("  No new live data since last check - nothing to update.")
        # status "no_new_data" (not "success") - keeps the cutoff from
        # advancing, so the next run still checks from the same point
        log_run("live_inference", 0, 0, "no_new_data")
        return

    print(f"  {len(wb_ids)} waterbodies have new live data")

    live_features = recompute_live_edm_features(wb_ids)
    if live_features.empty:
        print("  Could not aggregate live features - skipping.")
        log_run("live_inference", len(wb_ids), 0, "skipped")
        return

    # pull the affected points' other features (chemistry, land cover) from
    # the existing feature matrix - only sewage numbers change, nothing else
    conn = get_conn()
    placeholders = ",".join("?" * len(wb_ids))
    points = pd.read_sql(f"""
        SELECT * FROM feat_matrix WHERE wb_id IN ({placeholders})
    """, conn, params=wb_ids)
    conn.close()

    if points.empty:
        print("  No matching FWW points for these waterbodies - nothing to update.")
        log_run("live_inference", len(wb_ids), 0, "success")
        return

    # overwrite sewage columns with fresh live values
    points = points.drop(columns=["spills_per_pipe"], errors="ignore")
    points = points.merge(live_features, on="wb_id", how="left")
    points["spills_per_pipe"] = points["spills_per_pipe"].fillna(0)
    points["n_overflows"] = points["n_overflows_y"].fillna(points.get("n_overflows_x", 0)) \
        if "n_overflows_y" in points.columns else points.get("n_overflows", 0)

    # load model and predict
    model = joblib.load(f"{MODELS_DIR}/rf_model.pkl")
    le = joblib.load(ENCODER_PATH)

    feat_cols = [c for c in ALL_FEATS if c in points.columns]
    X = points[feat_cols].fillna(0).values

    pred_encoded = model.predict(X)
    pred_proba = model.predict_proba(X)
    pred_labels = le.inverse_transform(pred_encoded)

    classes = list(le.classes_)
    mod_idx = classes.index("Moderate") if "Moderate" in classes else 0
    poor_idx = classes.index("Poor") if "Poor" in classes else 1

    print(f"  Re-predicting {len(points)} points...")
    points = points.reset_index(drop=True)  # ensures position matches pred arrays exactly

    for pos, row in points.iterrows():
        save_prediction({
            "fww_id":           row.get("fww_id"),
            "site_name":        row.get("site_name"),
            "easting":          row.get("easting"),
            "northing":         row.get("northing"),
            "wb_id":            row.get("wb_id"),
            "predicted_status": pred_labels[pos],
            "prob_moderate":    round(float(pred_proba[pos][mod_idx]), 4),
            "prob_poor":        round(float(pred_proba[pos][poor_idx]), 4),
            "model_version":    "rf_model_live",
            "data_source":      "live",
            "predicted_at":     datetime.now().isoformat(),
        })

    n_poor_now = int((pred_labels == "Poor").sum())
    print(f"  Done. {n_poor_now} of {len(points)} now predicted Poor.")
    log_run("live_inference", len(points), len(points), "success")


if __name__ == "__main__":
    run_live_inference()