"""
pull_poopy_now.py
Pulls current sewage discharge status from all water companies via POOPy.

Each monitor's x_coord/y_coord (British National Grid) is spatially
joined to the nearest EA WFD waterbody, giving a proper wb_id that
matches the training feature matrix - not a name-based guess.

Used for:
    1. Scheduled live ingestion (every 15 min via scheduler.py)
    2. live_inference.py, which re-predicts affected FWW points
"""

import pandas as pd
import numpy as np
from datetime import datetime
from scipy.spatial import cKDTree
from db_loader import load_staging_edm_live, get_conn, log_run

COMPANIES = [
    "ThamesWater", "SouthernWater", "WessexWater", "SouthWestWater",
    "AnglianWater", "NorthumbrianWater", "UnitedUtilities",
    "YorkshireWater", "WelshWater",
    # SevernTrent excluded - not a valid attribute in poopy.companies
    # as of this version; class name differs, needs checking separately
]


def get_wfd_lookup():
    """
    Load waterbody coordinates from feat_matrix for spatial matching.
    Uses the same easting/northing + wb_id pairs already used in training,
    so live points join to the SAME waterbody codes the model was trained on.
    """
    conn = get_conn()
    wfd = pd.read_sql("""
        SELECT DISTINCT wb_id, easting, northing
        FROM feat_matrix
        WHERE easting IS NOT NULL AND northing IS NOT NULL
    """, conn)
    conn.close()
    return wfd


def pull_one_company(name, wfd_tree, wfd_ids):
    """
    Pull all monitor statuses from one water company.
    Spatially joins each monitor's x_coord/y_coord to nearest wb_id.
    """
    try:
        mod = __import__("poopy.companies", fromlist=[name])
        cls = getattr(mod, name)
        company = cls()

        rows = []
        for key, m in company.active_monitors.items():
            x, y = getattr(m, "x_coord", None), getattr(m, "y_coord", None)

            wb_id = None
            if x is not None and y is not None and x != 0 and y != 0:
                # nearest WFD waterbody by real coordinates - same method
                # used to link FWW points to waterbodies during training
                dist, idx = wfd_tree.query([x, y], k=1)
                if dist <= 5000:  # 5km max match distance, same as training
                    wb_id = wfd_ids[idx]

            rows.append({
                "wb_id":        wb_id,
                "outlet_ngr":   getattr(m, "site_name", key),
                "company":      name,
                "status":       str(getattr(m, "current_status", "Unknown")),
                "duration_hrs": 0.0,
            })

        n_matched = sum(1 for r in rows if r["wb_id"] is not None)
        print(f"  {name}: {len(rows)} monitors, {n_matched} matched to a waterbody")
        return rows

    except Exception as e:
        print(f"  {name}: failed - {e}")
        return []


def main():
    print("Pulling live EDM data from all WaSCs via POOPy...")

    wfd = get_wfd_lookup()
    if wfd.empty:
        print("  No waterbody coordinates found in feat_matrix - cannot spatially join.")
        log_run("edm_live", 0, 0, "failed", "feat_matrix has no coordinates")
        return

    wfd_tree = cKDTree(wfd[["easting", "northing"]].values)
    wfd_ids = wfd["wb_id"].values
    print(f"  Loaded {len(wfd)} waterbody locations for spatial matching")

    all_rows = []
    for company in COMPANIES:
        all_rows.extend(pull_one_company(company, wfd_tree, wfd_ids))

    if not all_rows:
        print("No data pulled from any company.")
        log_run("edm_live", 0, 0, "failed", "No data from any WaSC")
        return

    df = pd.DataFrame(all_rows)
    n_matched = df["wb_id"].notna().sum()
    print(f"\nTotal monitors pulled: {len(df)}")
    print(f"Matched to a waterbody: {n_matched} ({n_matched/len(df)*100:.0f}%)")
    print(df["company"].value_counts())
    print(df["status"].value_counts())

    load_staging_edm_live(df)
    print("\nSaved to staging_edm_live table.")
    log_run("edm_live", len(df), n_matched, "success")


if __name__ == "__main__":
    main()