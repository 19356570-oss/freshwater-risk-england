"""
db_loader.py
Load step of the ETL pipeline - writes all layers to SQLite.

Table strategy:
    Append          - staging_fww, staging_edm_live, predictions, ingestion_log
    Truncate+load   - staging_wfd, staging_edm_historical, staging_lc, feat_matrix
    Snapshot backup - staging_wfd_snapshot, staging_edm_historical_snapshot, staging_lc_snapshot
    Write once      - model_metrics (after training only)

Land cover note:
    Raw raster is not stored in SQLite - too large.
    Only the extracted % features per FWW point are stored in staging_lc.
    When UKCEH releases a new annual raster, snapshot current and reload.
"""

import sqlite3
import pandas as pd
import json
import os
import shutil
from datetime import datetime
from config import DB_PATH, FEAT_MATRIX, RESULTS_JSON


# ---- Connection helper ------------------------------------------------------

def get_conn(db_path=DB_PATH):
    """Open SQLite connection. Creates db file if it does not exist."""
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    return sqlite3.connect(db_path)


# ---- Snapshot helper --------------------------------------------------------

def snapshot_before_truncate(conn, src_table, snap_table):
    """
    Copy current rows from src_table into snap_table before truncating.
    Adds snapshot_at timestamp so you can tell when each snapshot was taken.
    Called before every truncate+load operation.
    """
    # check source table has rows worth snapshotting
    n = pd.read_sql(f"SELECT COUNT(*) as n FROM {src_table}", conn).iloc[0]["n"]
    if n == 0:
        print(f"  {src_table} is empty - nothing to snapshot")
        return

    conn.execute(f"""
        INSERT INTO {snap_table}
        SELECT *, datetime('now') as snapshot_at
        FROM {src_table}
    """)
    print(f"  Snapshot: {n} rows from {src_table} -> {snap_table}")


# ---- Create all tables ------------------------------------------------------

def create_tables(db_path=DB_PATH):
    """
    Create all tables if they do not exist.
    Safe to run multiple times - uses IF NOT EXISTS.
    """

    print(f"Setting up database: {db_path}")
    conn = get_conn(db_path)
    cur = conn.cursor()

    # ---- staging_fww (append) -----------------------------------------------
    # All FWW readings ever received - keeps full history
    cur.execute("""
        CREATE TABLE IF NOT EXISTS staging_fww (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            fww_id        TEXT,
            site_name     TEXT,
            sample_date   TEXT,
            lon           REAL,
            lat           REAL,
            easting       REAL,
            northing      REAL,
            nitrate_mid   REAL,
            phosphate_mid REAL,
            turbidity     REAL,
            ph            REAL,
            conductivity  REAL,
            dissolved_o2  REAL,
            water_temp    REAL,
            fww_rating    TEXT,
            county        TEXT,
            rbd           TEXT,
            received_at   TEXT DEFAULT (datetime('now'))  -- when we ingested this record
        )
    """)

    # ---- staging_wfd (truncate+load) ----------------------------------------
    # Current WFD classifications only - replaced on each WFD cycle (~6 years)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS staging_wfd (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            wb_id      TEXT,
            wb_name    TEXT,
            easting    REAL,
            northing   REAL,
            rbd        TEXT,
            wfd_status TEXT,
            loaded_at  TEXT DEFAULT (datetime('now'))
        )
    """)

    # ---- staging_wfd_snapshot (append before truncate) ----------------------
    # Backup of previous WFD cycles before overwriting
    cur.execute("""
        CREATE TABLE IF NOT EXISTS staging_wfd_snapshot (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            wb_id       TEXT,
            wb_name     TEXT,
            easting     REAL,
            northing    REAL,
            rbd         TEXT,
            wfd_status  TEXT,
            loaded_at   TEXT,
            snapshot_at TEXT  -- when this version was archived
        )
    """)

    # ---- staging_edm_historical (truncate+load) -----------------------------
    # Annual EDM returns - replaced when new year published (every March)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS staging_edm_historical (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            wb_id       TEXT,
            outlet_ngr  TEXT,
            spill_count REAL,
            spill_hrs   REAL,
            avg_spills  REAL,
            n_overflows INTEGER,
            spills_per_pipe REAL,
            hrs_per_pipe    REAL,
            edm_pct     REAL,
            years       TEXT,  -- which years covered e.g. "2024,2025"
            loaded_at   TEXT DEFAULT (datetime('now'))
        )
    """)

    # ---- staging_edm_historical_snapshot (append before truncate) -----------
    cur.execute("""
        CREATE TABLE IF NOT EXISTS staging_edm_historical_snapshot (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            wb_id       TEXT,
            outlet_ngr  TEXT,
            spill_count REAL,
            spill_hrs   REAL,
            avg_spills  REAL,
            n_overflows INTEGER,
            edm_pct     REAL,
            years       TEXT,
            loaded_at   TEXT,
            snapshot_at TEXT
        )
    """)

    # ---- staging_edm_live (append) ------------------------------------------
    # POOPy 15-min live data - append only, keeps full event history
    cur.execute("""
        CREATE TABLE IF NOT EXISTS staging_edm_live (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            wb_id        TEXT,
            outlet_ngr   TEXT,
            company      TEXT,
            status        TEXT,   -- "discharging" / "not discharging"
            duration_hrs REAL,
            received_at  TEXT DEFAULT (datetime('now'))  -- when scheduler pulled this
        )
    """)

    # ---- staging_lc (truncate+load) -----------------------------------------
    # Extracted land cover % per FWW point for current raster year
    # Raw raster not stored here - only the extracted features
    cur.execute("""
        CREATE TABLE IF NOT EXISTS staging_lc (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            fww_id          TEXT,
            easting         REAL,
            northing        REAL,
            lc_woodland_1km REAL,
            lc_arable_1km   REAL,
            lc_grass_1km    REAL,
            lc_wetland_1km  REAL,
            lc_urban_1km    REAL,
            lc_water_1km    REAL,
            lc_woodland_5km REAL,
            lc_arable_5km   REAL,
            lc_grass_5km    REAL,
            lc_wetland_5km  REAL,
            lc_urban_5km    REAL,
            lc_water_5km    REAL,
            raster_year     INTEGER,  -- which UKCEH LCM year this came from
            loaded_at       TEXT DEFAULT (datetime('now'))
        )
    """)

    # ---- staging_lc_snapshot (append before truncate) -----------------------
    cur.execute("""
        CREATE TABLE IF NOT EXISTS staging_lc_snapshot (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            fww_id          TEXT,
            easting         REAL,
            northing        REAL,
            lc_woodland_1km REAL,
            lc_arable_1km   REAL,
            lc_grass_1km    REAL,
            lc_wetland_1km  REAL,
            lc_urban_1km    REAL,
            lc_water_1km    REAL,
            lc_woodland_5km REAL,
            lc_arable_5km   REAL,
            lc_grass_5km    REAL,
            lc_wetland_5km  REAL,
            lc_urban_5km    REAL,
            lc_water_5km    REAL,
            raster_year     INTEGER,
            loaded_at       TEXT,
            snapshot_at     TEXT
        )
    """)

    # ---- feat_matrix (truncate+load) ----------------------------------------
    # Current training-ready feature matrix - rebuilt after any source updates
    cur.execute("""
        CREATE TABLE IF NOT EXISTS feat_matrix (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            fww_id          TEXT,
            site_name       TEXT,
            sample_date     TEXT,
            easting         REAL,
            northing        REAL,
            wb_id           TEXT,
            nitrate_mid     REAL,
            phosphate_mid   REAL,
            spill_count     REAL,
            spill_hrs       REAL,
            spills_per_pipe REAL,
            hrs_per_pipe    REAL,
            avg_spills      REAL,
            n_overflows     INTEGER,
            lc_woodland_1km REAL,
            lc_arable_1km   REAL,
            lc_grass_1km    REAL,
            lc_wetland_1km  REAL,
            lc_urban_1km    REAL,
            lc_water_1km    REAL,
            lc_woodland_5km REAL,
            lc_arable_5km   REAL,
            lc_grass_5km    REAL,
            lc_wetland_5km  REAL,
            lc_urban_5km    REAL,
            lc_water_5km    REAL,
            county          TEXT,
            rbd             TEXT,
            wfd_dist_m      REAL,
            match_q         TEXT,
            wfd_status      TEXT,
            loaded_at       TEXT DEFAULT (datetime('now'))
        )
    """)

    # ---- predictions (append) -----------------------------------------------
    # All predictions ever made - historical batch and live dashboard
    cur.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            fww_id           TEXT,
            site_name        TEXT,
            easting          REAL,
            northing         REAL,
            wb_id            TEXT,
            predicted_status TEXT,
            prob_moderate    REAL,
            prob_poor        REAL,
            model_version    TEXT,
            data_source      TEXT,   -- "historical" or "live"
            predicted_at     TEXT DEFAULT (datetime('now'))
        )
    """)

    # ---- model_metrics (write once after training) --------------------------
    # Static after training - not updated in production
    cur.execute("""
        CREATE TABLE IF NOT EXISTS model_metrics (
            id             INTEGER PRIMARY KEY AUTOINCREMENT,
            model_type     TEXT,
            approach       TEXT,   -- "Traditional Rule-Based" or "AI / Machine Learning"
            feature_set    TEXT,
            fold_id        INTEGER,
            n_train        INTEGER,
            n_test         INTEGER,
            weighted_f1    REAL,
            mod_precision  REAL,
            mod_recall     REAL,
            mod_f1         REAL,
            poor_precision REAL,
            poor_recall    REAL,
            poor_f1        REAL,
            recorded_at    TEXT DEFAULT (datetime('now'))
        )
    """)

    # ---- ingestion_log (append) ---------------------------------------------
    # Every ETL run logged - shows scheduled pipeline is working
    cur.execute("""
        CREATE TABLE IF NOT EXISTS ingestion_log (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            source          TEXT,      -- fww / edm_historical / edm_live / wfd / lc
            records_fetched INTEGER,
            records_new     INTEGER,
            status          TEXT,      -- success / failed
            error_msg       TEXT,
            run_at          TEXT DEFAULT (datetime('now'))
        )
    """)

    conn.commit()
    conn.close()
    print("  All tables ready.")


# ---- Staging loads ----------------------------------------------------------

def load_staging_fww(fww_df, db_path=DB_PATH):
    """Append new FWW records to staging_fww. Keeps full history."""
    print("Loading staging_fww (append)...")
    conn = get_conn(db_path)
    fww_df["received_at"] = datetime.now().isoformat()
    fww_df.to_sql("staging_fww", conn, if_exists="append", index=False)
    n = pd.read_sql("SELECT COUNT(*) as n FROM staging_fww", conn).iloc[0]["n"]
    conn.commit()
    conn.close()
    log_run("fww", len(fww_df), len(fww_df), "success", db_path=db_path)
    print(f"  staging_fww total: {n} rows")


def load_staging_wfd(wfd_df, db_path=DB_PATH):
    """
    Truncate+load staging_wfd.
    Snapshots current data before truncating so history is not lost.
    """
    print("Loading staging_wfd (truncate+load with snapshot)...")
    conn = get_conn(db_path)
    snapshot_before_truncate(conn, "staging_wfd", "staging_wfd_snapshot")
    conn.execute("DELETE FROM staging_wfd")
    wfd_df["loaded_at"] = datetime.now().isoformat()
    wfd_df.to_sql("staging_wfd", conn, if_exists="append", index=False)
    n = pd.read_sql("SELECT COUNT(*) as n FROM staging_wfd", conn).iloc[0]["n"]
    conn.commit()
    conn.close()
    log_run("wfd", len(wfd_df), len(wfd_df), "success", db_path=db_path)
    print(f"  staging_wfd: {n} rows")


def load_staging_edm_historical(edm_df, db_path=DB_PATH):
    """
    Truncate+load staging_edm_historical.
    Snapshots current data first - previous annual returns are preserved.
    """
    print("Loading staging_edm_historical (truncate+load with snapshot)...")
    conn = get_conn(db_path)
    snapshot_before_truncate(conn, "staging_edm_historical", "staging_edm_historical_snapshot")
    conn.execute("DELETE FROM staging_edm_historical")
    edm_df["loaded_at"] = datetime.now().isoformat()
    edm_df.to_sql("staging_edm_historical", conn, if_exists="append", index=False)
    n = pd.read_sql("SELECT COUNT(*) as n FROM staging_edm_historical", conn).iloc[0]["n"]
    conn.commit()
    conn.close()
    log_run("edm_historical", len(edm_df), len(edm_df), "success", db_path=db_path)
    print(f"  staging_edm_historical: {n} rows")


def load_staging_edm_live(edm_live_df, db_path=DB_PATH):
    """
    Append live POOPy data to staging_edm_live.
    Called by APScheduler every 15 minutes in dashboard mode.
    """
    conn = get_conn(db_path)
    edm_live_df["received_at"] = datetime.now().isoformat()
    edm_live_df.to_sql("staging_edm_live", conn, if_exists="append", index=False)
    n = pd.read_sql("SELECT COUNT(*) as n FROM staging_edm_live", conn).iloc[0]["n"]
    conn.commit()
    conn.close()
    log_run("edm_live", len(edm_live_df), len(edm_live_df), "success", db_path=db_path)
    print(f"  staging_edm_live total: {n} rows")


def load_staging_lc(lc_df, raster_year=2024, db_path=DB_PATH):
    """
    Truncate+load staging_lc with extracted land cover features.
    Snapshots current before truncating - previous raster year preserved.
    lc_df must have fww_id, easting, northing, and all lc_* columns.
    """
    print(f"Loading staging_lc (truncate+load with snapshot, raster year={raster_year})...")
    conn = get_conn(db_path)
    snapshot_before_truncate(conn, "staging_lc", "staging_lc_snapshot")
    conn.execute("DELETE FROM staging_lc")
    lc_df["raster_year"] = raster_year
    lc_df["loaded_at"] = datetime.now().isoformat()
    lc_df.to_sql("staging_lc", conn, if_exists="append", index=False)
    n = pd.read_sql("SELECT COUNT(*) as n FROM staging_lc", conn).iloc[0]["n"]
    conn.commit()
    conn.close()
    log_run("land_cover", len(lc_df), len(lc_df), "success", db_path=db_path)
    print(f"  staging_lc: {n} rows (LCM {raster_year})")


# ---- Curated layer loads ----------------------------------------------------

def load_feat_matrix(matrix_df=None, csv_path=FEAT_MATRIX, db_path=DB_PATH):
    """
    Truncate+load feat_matrix with current training-ready features.
    Rebuilt whenever any source table is updated.
    No snapshot needed - can always be regenerated from staging tables.
    """
    print("Loading feat_matrix (truncate+load)...")
    if matrix_df is None:
        matrix_df = pd.read_csv(csv_path)  # fallback to CSV if no df passed

    # rename old long column names to short names if needed
    rename_map = {
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
    matrix_df = matrix_df.rename(columns={k: v for k, v in rename_map.items() if k in matrix_df.columns})

    conn = get_conn(db_path)
    conn.execute("DELETE FROM feat_matrix")
    matrix_df["loaded_at"] = datetime.now().isoformat()
    matrix_df.to_sql("feat_matrix", conn, if_exists="append", index=False)
    n = pd.read_sql("SELECT COUNT(*) as n FROM feat_matrix", conn).iloc[0]["n"]
    conn.commit()
    conn.close()
    log_run("feat_matrix", len(matrix_df), len(matrix_df), "success", db_path=db_path)
    print(f"  feat_matrix: {n} rows")


def load_model_metrics(results_path=RESULTS_JSON, db_path=DB_PATH):
    """
    Write model evaluation metrics to model_metrics.
    Called once after training - not updated in production.
    """
    print("Loading model_metrics (write once)...")

    with open(results_path, "r") as f:
        results = json.load(f)

    conn = get_conn(db_path)
    conn.execute("DELETE FROM model_metrics")  # clear if rerunning after retraining

    rows = []

    # traditional threshold results
    if "traditional" in results:
        for r in results["traditional"]:
            rows.append({
                "model_type":    r["method"],
                "approach":      "Traditional Rule-Based",
                "feature_set":   "phosphate + nitrate thresholds",
                "fold_id":       None,
                "n_train":       None,
                "n_test":        None,
                "weighted_f1":   r["weighted_f1"],
                "mod_precision": r["moderate_precision"],
                "mod_recall":    r["moderate_recall"],
                "mod_f1":        r["moderate_f1"],
                "poor_precision":r["poor_precision"],
                "poor_recall":   r["poor_recall"],
                "poor_f1":       r["poor_f1"],
                "recorded_at":   datetime.now().isoformat(),
            })

    # ML model per-fold results - nested under "ml_comparison" key
    ml_keys = {
        "logreg": "Logistic Regression",
        "rf":     "Random Forest",
        "xgb":    "XGBoost",
    }
    ml_results = results.get("ml_comparison", {})
    for key, name in ml_keys.items():
        if key not in ml_results:
            continue
        res = ml_results[key]
        report = res.get("classification_report", {})
        for fold in res.get("fold_results", []):
            rows.append({
                "model_type":    name,
                "approach":      "AI / Machine Learning",
                "feature_set":   "full_feature_set",
                "fold_id":       fold["fold"],
                "n_train":       fold["n_train"],
                "n_test":        fold["n_test"],
                "weighted_f1":   fold["weighted_f1"],
                "mod_precision": report.get("Moderate", {}).get("precision"),
                "mod_recall":    report.get("Moderate", {}).get("recall"),
                "mod_f1":        report.get("Moderate", {}).get("f1-score"),
                "poor_precision":report.get("Poor", {}).get("precision"),
                "poor_recall":   report.get("Poor", {}).get("recall"),
                "poor_f1":       report.get("Poor", {}).get("f1-score"),
                "recorded_at":   datetime.now().isoformat(),
            })

    pd.DataFrame(rows).to_sql("model_metrics", conn, if_exists="append", index=False)
    n = pd.read_sql("SELECT COUNT(*) as n FROM model_metrics", conn).iloc[0]["n"]
    conn.commit()
    conn.close()
    print(f"  model_metrics: {n} rows")


# ---- Audit helpers ----------------------------------------------------------

def log_run(source, fetched, new_recs, status, error=None, db_path=DB_PATH):
    """Log every ETL run. Called at the end of each load function."""
    conn = get_conn(db_path)
    conn.execute("""
        INSERT INTO ingestion_log
        (source, records_fetched, records_new, status, error_msg, run_at)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (source, fetched, new_recs, status, error, datetime.now().isoformat()))
    conn.commit()
    conn.close()


def save_prediction(row, db_path=DB_PATH):
    """Append one prediction to predictions table. Used by dashboard."""
    conn = get_conn(db_path)
    pd.DataFrame([row]).to_sql("predictions", conn, if_exists="append", index=False)
    conn.commit()
    conn.close()


# ---- Query helpers for dashboard --------------------------------------------

def get_feat_matrix(db_path=DB_PATH):
    """Read current feature matrix. Used by dashboard for inference."""
    conn = get_conn(db_path)
    df = pd.read_sql("SELECT * FROM feat_matrix", conn)
    conn.close()
    return df


def get_predictions(source=None, db_path=DB_PATH):
    """Read predictions, optionally filtered by source."""
    conn = get_conn(db_path)
    if source:
        df = pd.read_sql(
            "SELECT * FROM predictions WHERE data_source = ?",
            conn, params=(source,)
        )
    else:
        df = pd.read_sql("SELECT * FROM predictions", conn)
    conn.close()
    return df


def get_metrics_summary(db_path=DB_PATH):
    """Summary of model metrics grouped by approach and model type."""
    conn = get_conn(db_path)
    df = pd.read_sql("""
        SELECT
            approach,
            model_type,
            feature_set,
            COUNT(*)                   as n_folds,
            ROUND(AVG(weighted_f1), 4) as mean_f1,
            ROUND(MIN(weighted_f1), 4) as min_f1,
            ROUND(MAX(weighted_f1), 4) as max_f1
        FROM model_metrics
        GROUP BY approach, model_type, feature_set
        ORDER BY mean_f1 DESC
    """, conn)
    conn.close()
    return df


# ---- Weekly database snapshot -----------------------------------------------

def db_snapshot(db_path=DB_PATH, snap_dir="data/snapshots"):
    """
    Copy the entire database file to a timestamped backup.
    Called weekly by APScheduler per the data management plan.
    Different from table snapshots - this is a full file-level backup.
    """
    os.makedirs(snap_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    dest = os.path.join(snap_dir, f"freshwater_risk_{ts}.db")
    shutil.copy2(db_path, dest)
    log_run("db_snapshot", 0, 0, "success", db_path=db_path)
    print(f"  DB snapshot saved: {dest}")
    return dest


# ---- Main - full ETL load ---------------------------------------------------

if __name__ == "__main__":

    print("=" * 60)
    print("ETL - Load step")
    print("=" * 60)

    create_tables()

    # load feature matrix from CSV (staging tables loaded during pipeline run)
    if os.path.exists(FEAT_MATRIX):
        load_feat_matrix()
    else:
        print(f"feat_matrix CSV not found - run feature_engineering.py first")

    # load model metrics if training is complete
    if os.path.exists(RESULTS_JSON):
        load_model_metrics()
    else:
        print(f"modelling_results.json not found - run model_training.py first")

    # print summary
    conn = get_conn()
    print("\nDatabase summary:")
    tables = [
        "staging_fww", "staging_wfd", "staging_wfd_snapshot",
        "staging_edm_historical", "staging_edm_historical_snapshot",
        "staging_edm_live", "staging_lc", "staging_lc_snapshot",
        "feat_matrix", "predictions", "model_metrics", "ingestion_log"
    ]
    for t in tables:
        try:
            n = pd.read_sql(f"SELECT COUNT(*) as n FROM {t}", conn).iloc[0]["n"]
            print(f"  {t:40s}: {n} rows")
        except Exception:
            print(f"  {t:40s}: table missing")
    conn.close()

    print("\nMetrics summary:")
    print(get_metrics_summary())

    print(f"\nDatabase ready: {DB_PATH}")