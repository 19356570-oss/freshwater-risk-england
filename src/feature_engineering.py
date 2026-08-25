"""
feature_engineering.py
Builds and maintains the feature matrix from staging tables.

Two modes:
    full    - rebuild entire feat_matrix from scratch (first run)
    append  - process only new records since last_run_date, append to feat_matrix

Column names are standardised here - all downstream scripts use short names.
Same functions used for both historical and live data.
Never change aggregation logic after model is trained.
"""

import pandas as pd
import numpy as np
import rasterio
from rasterio.mask import mask
from shapely.geometry import Point
from scipy.spatial import cKDTree
from pyproj import Transformer
import os
from datetime import datetime
from config import (
    LC_FILE, FEAT_MATRIX, FWW_JOINED,
    BUF_1KM, BUF_5KM, LC_GROUPS,
    PROCESSED_DIR, CRS_WGS84, CRS_BNG,
    DB_PATH
)


# ---- Load from staging tables -----------------------------------------------

def load_fww_from_staging(since=None, db_path=DB_PATH):
    """
    Load FWW records from staging_fww.
    since: ISO datetime string - only load records received after this date.
           None = load all records (full rebuild mode).
    """
    from db_loader import get_conn
    conn = get_conn(db_path)

    if since:
        query = f"""
            SELECT * FROM staging_fww
            WHERE received_at > "{since}"
              AND nitrate_mid IS NOT NULL
              AND phosphate_mid IS NOT NULL
        """
        print(f"Loading FWW from staging (since {since})...")
    else:
        query = """
            SELECT * FROM staging_fww
            WHERE nitrate_mid IS NOT NULL
              AND phosphate_mid IS NOT NULL
        """
        print("Loading FWW from staging (full)...")

    df = pd.read_sql(query, conn)
    conn.close()

    # convert coords to BNG if not already done
    if "easting" not in df.columns or df["easting"].isnull().all():
        proj = Transformer.from_crs(CRS_WGS84, CRS_BNG, always_xy=True)
        df["easting"], df["northing"] = proj.transform(
            df["lon"].values, df["lat"].values
        )

    df["sample_date"] = pd.to_datetime(df["sample_date"], errors="coerce")
    print(f"  {len(df)} FWW records")
    return df


def load_wfd_from_staging(db_path=DB_PATH):
    """Load current WFD classifications from staging_wfd."""
    from db_loader import get_conn
    conn = get_conn(db_path)
    df = pd.read_sql("SELECT wb_id, wb_name, easting, northing, rbd, wfd_status FROM staging_wfd", conn)
    conn.close()
    print(f"  WFD: {len(df)} waterbodies")
    return df


def load_edm_from_staging(db_path=DB_PATH):
    """
    Load aggregated EDM from staging_edm_historical.
    For live retraining, run poopy_loader.py first to populate staging_edm_live,
    then use load_combined_edm() from retrain.py instead.
    """
    from db_loader import get_conn
    conn = get_conn(db_path)
    df = pd.read_sql("""
        SELECT wb_id, spill_count, spill_hrs,
               avg_spills, n_overflows, edm_pct
        FROM staging_edm_historical
    """, conn)
    conn.close()
    print(f"  EDM: {len(df)} waterbodies")
    return df


def load_lc_from_staging(db_path=DB_PATH):
    """Load land cover features from staging_lc."""
    from db_loader import get_conn
    conn = get_conn(db_path)
    df = pd.read_sql("""
        SELECT fww_id,
               lc_woodland_1km, lc_arable_1km, lc_grass_1km,
               lc_wetland_1km,  lc_urban_1km,  lc_water_1km,
               lc_woodland_5km, lc_arable_5km, lc_grass_5km,
               lc_wetland_5km,  lc_urban_5km,  lc_water_5km
        FROM staging_lc
    """, conn)
    conn.close()
    print(f"  Land cover: {len(df)} records")
    return df


def get_last_run_date(db_path=DB_PATH):
    """Get the date of the last successful feature engineering run."""
    from db_loader import get_conn
    conn = get_conn(db_path)
    try:
        df = pd.read_sql("""
            SELECT MAX(run_at) as last_run
            FROM ingestion_log
            WHERE source = "feature_engineering"
              AND status = "success"
        """, conn)
        last = df.iloc[0]["last_run"]
    except Exception:
        last = None
    conn.close()
    return last


# ---- Transform steps --------------------------------------------------------

def join_fww_wfd(fww, wfd):
    """Match each FWW point to nearest WFD waterbody centroid via KD-tree."""
    print("Spatial join: FWW -> WFD...")
    wfd_coords = np.array(list(zip(wfd["easting"], wfd["northing"])))
    fww_coords = np.array(list(zip(fww["easting"], fww["northing"])))
    tree = cKDTree(wfd_coords)
    dists, idx = tree.query(fww_coords, k=1)

    fww = fww.copy()
    fww["wb_id"]      = wfd["wb_id"].iloc[idx].values
    fww["wb_name"]    = wfd["wb_name"].iloc[idx].values
    fww["wfd_status"] = wfd["wfd_status"].iloc[idx].values
    fww["wfd_dist_m"] = dists
    fww["match_q"]    = np.where(dists <= 1000, "good",
                        np.where(dists <= 5000, "acceptable", "poor"))

    print(f"  Match quality:\n{fww['match_q'].value_counts()}")
    return fww


def join_edm(fww, edm):
    """Join EDM sewage features via waterbody ID. Missing = 0."""
    print("Joining EDM features...")
    fww = fww.merge(edm, on="wb_id", how="left")
    edm_cols = ["spill_count", "spill_hrs", "avg_spills", "n_overflows", "edm_pct"]
    fww[edm_cols] = fww[edm_cols].fillna(0)

    # Normalise by number of monitored outlets.
    # Raw spill_count is inflated in urban areas simply because more EDM monitors
    # are installed there - it partly measures monitoring density, not pollution.
    # Dividing by n_overflows gives discharge intensity per outlet instead.
    pipes = fww["n_overflows"].replace(0, np.nan)
    fww["spills_per_pipe"] = (fww["spill_count"] / pipes).fillna(0)
    fww["hrs_per_pipe"]    = (fww["spill_hrs"] / pipes).fillna(0)

    print(f"  Records with spill data: {(fww['spill_count'] > 0).sum()}")
    return fww


def join_lc(fww, lc):
    """Join land cover features via fww_id."""
    print("Joining land cover features...")
    fww = fww.merge(lc, on="fww_id", how="left")
    return fww


def extract_lc_features(fww, lc_path=LC_FILE):
    """
    Extract land cover % within 1km and 5km buffers from UKCEH raster.
    Used when staging_lc does not yet have data for new FWW points.
    Slow - run once per new batch of FWW points.
    """
    print("Extracting land cover from raster (slow)...")

    for buf_lbl in ["1km", "5km"]:
        for grp in LC_GROUPS.keys():
            fww[f"{grp}_{buf_lbl}"] = np.nan

    with rasterio.open(lc_path) as src:
        for idx, row in fww.iterrows():
            pt = Point(row["easting"], row["northing"])
            for buf_m, buf_lbl in [(BUF_1KM, "1km"), (BUF_5KM, "5km")]:
                buf = pt.buffer(buf_m)
                try:
                    out, _ = mask(src, [buf], crop=True)
                    pixels = out[0].flatten()
                    pixels = pixels[pixels != src.nodata]
                    total = len(pixels)
                    if total == 0:
                        continue
                    for grp, codes in LC_GROUPS.items():
                        n = np.isin(pixels, codes).sum()
                        fww.at[idx, f"{grp}_{buf_lbl}"] = (n / total) * 100
                except Exception:
                    pass
            if idx % 1000 == 0:
                print(f"  {idx}/{len(fww)} done...")

    print("  Land cover extraction done.")
    return fww


def build_matrix(fww):
    """
    Select final ML features. Canonical feature set - do not change after training.
    Drops sparse chemistry columns (90-100% missing in FWW).
    """
    print("Building feature matrix...")

    keep = [
        "fww_id", "site_name", "sample_date",
        "easting", "northing", "wb_id",
        "nitrate_mid", "phosphate_mid",
        "spill_count", "spill_hrs", "avg_spills", "n_overflows",
        "spills_per_pipe", "hrs_per_pipe",
        "lc_woodland_1km", "lc_arable_1km", "lc_grass_1km",
        "lc_wetland_1km",  "lc_urban_1km",  "lc_water_1km",
        "lc_woodland_5km", "lc_arable_5km", "lc_grass_5km",
        "lc_wetland_5km",  "lc_urban_5km",  "lc_water_5km",
        "county", "rbd", "wfd_dist_m", "match_q", "wfd_status",
    ]

    available = [c for c in keep if c in fww.columns]
    df = fww[available].copy()

    df = df.dropna(subset=["wfd_status"])
    df = df[df["match_q"] != "poor"]
    df = df.dropna(subset=["nitrate_mid", "phosphate_mid"])
    df["county"] = df["county"].fillna("Unknown")
    df["rbd"]    = df["rbd"].fillna("Unknown")

    print(f"  Feature matrix: {len(df)} rows x {len(df.columns)} cols")
    print(f"  Labels:\n{df['wfd_status'].value_counts()}")
    return df


# ---- Pipeline modes ---------------------------------------------------------

def run_full(use_staging=True, skip_lc=False):
    """
    Full rebuild of feat_matrix from all staging tables.
    use_staging=True  - read from DB staging tables (recommended)
    use_staging=False - read from raw CSV files (first run only)
    skip_lc=True      - skip land cover extraction (fast testing)
    """
    from db_loader import load_feat_matrix, log_run

    os.makedirs(PROCESSED_DIR, exist_ok=True)
    print("=" * 60)
    print("Feature engineering - full rebuild")
    print("=" * 60)

    if use_staging:
        fww = load_fww_from_staging()
        wfd = load_wfd_from_staging()
        edm = load_edm_from_staging()
        lc  = load_lc_from_staging()
    else:
        # fallback - read from raw files (first run before staging is populated)
        from data_loader import load_fww as _load_fww, load_wfd as _load_wfd, load_edm as _load_edm
        fww = _load_fww()
        wfd = _load_wfd()
        edm = _load_edm()
        lc  = pd.DataFrame()

    fww = join_fww_wfd(fww, wfd)
    fww = join_edm(fww, edm)

    if lc.empty or skip_lc:
        if not skip_lc:
            fww = extract_lc_features(fww)  # extract from raster
    else:
        fww = join_lc(fww, lc)              # join from staging_lc

    matrix = build_matrix(fww)
    matrix.to_csv(FEAT_MATRIX, index=False)
    load_feat_matrix(matrix_df=matrix)

    log_run("feature_engineering", len(matrix), len(matrix), "success")
    print(f"\nSaved: {FEAT_MATRIX}")
    return matrix


def run_append(since=None):
    """
    Process only new FWW records since last run, append to feat_matrix.
    since: ISO datetime string. None = auto-detect from ingestion_log.

    Used by scheduler after each weekly FWW update.
    New rows get same features + labels as historical - consistent schema.
    """
    from db_loader import get_conn, load_feat_matrix, log_run

    print("=" * 60)
    print("Feature engineering - append mode")
    print("=" * 60)

    # auto-detect last run date if not provided
    if since is None:
        since = get_last_run_date()
        if since:
            print(f"Processing records since: {since}")
        else:
            print("No previous run found - switching to full rebuild")
            return run_full()

    # load only new FWW records
    new_fww = load_fww_from_staging(since=since)
    if new_fww.empty:
        print("No new FWW records since last run - nothing to append")
        log_run("feature_engineering", 0, 0, "success")
        return

    # load reference tables (unchanged)
    wfd = load_wfd_from_staging()
    edm = load_edm_from_staging()
    lc  = load_lc_from_staging()

    # transform new records
    new_fww = join_fww_wfd(new_fww, wfd)
    new_fww = join_edm(new_fww, edm)

    # join land cover - new points may not be in staging_lc yet
    new_ids = set(new_fww["fww_id"].astype(str))
    lc_ids  = set(lc["fww_id"].astype(str)) if not lc.empty else set()
    missing_lc = new_ids - lc_ids

    if missing_lc:
        print(f"  {len(missing_lc)} new points need land cover extraction...")
        new_needs_lc = new_fww[new_fww["fww_id"].astype(str).isin(missing_lc)].copy()
        new_has_lc   = new_fww[~new_fww["fww_id"].astype(str).isin(missing_lc)].copy()

        new_needs_lc = extract_lc_features(new_needs_lc)
        new_has_lc   = join_lc(new_has_lc, lc)
        new_fww = pd.concat([new_has_lc, new_needs_lc], ignore_index=True)
    else:
        new_fww = join_lc(new_fww, lc)

    new_matrix = build_matrix(new_fww)

    if new_matrix.empty:
        print("No valid rows after filtering - nothing appended")
        log_run("feature_engineering", 0, 0, "success")
        return

    # append to existing feat_matrix in DB
    conn = get_conn()
    new_matrix["loaded_at"] = datetime.now().isoformat()
    new_matrix.to_sql("feat_matrix", conn, if_exists="append", index=False)
    total = pd.read_sql("SELECT COUNT(*) as n FROM feat_matrix", conn).iloc[0]["n"]
    conn.commit()
    conn.close()

    # also append to CSV
    new_matrix.to_csv(FEAT_MATRIX, mode="a", header=False, index=False)

    log_run("feature_engineering", len(new_matrix), len(new_matrix), "success")
    print(f"\nAppended {len(new_matrix)} new rows. Total feat_matrix: {total}")
    return new_matrix


# ---- Main -------------------------------------------------------------------

if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",     default="full",  help="full / append")
    parser.add_argument("--since",    default=None,    help="ISO date for append mode")
    parser.add_argument("--no-staging", action="store_true", help="use raw files instead of staging")
    parser.add_argument("--skip-lc",  action="store_true",   help="skip land cover (fast test)")
    args = parser.parse_args()

    if args.mode == "append":
        run_append(since=args.since)
    else:
        run_full(
            use_staging=not args.no_staging,
            skip_lc=args.skip_lc
        )