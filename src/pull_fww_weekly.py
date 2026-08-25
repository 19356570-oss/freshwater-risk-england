"""
pull_poopy_now.py
One-off script to pull current POOPy status and append to staging_edm_live.
Run this manually now to get real incremental data for testing retrain.
 
Later, scheduler.py will call this same logic every 15 minutes automatically.
"""
 
import pandas as pd
from datetime import datetime
from db_loader import load_staging_edm_live, get_conn
 
# All 10 WaSCs POOPy supports - not all may have working APIs right now
COMPANIES = [
    "ThamesWater",
    "SouthernWater",
    "WessexWater",
    "SouthWestWater",
    "AnglianWater",
    "NorthumbrianWater",
    "UnitedUtilities",
    "YorkshireWater",
    "SevernTrent",
    "WelshWater",
]
 
 
def pull_one_company(name):
    """Try pulling one company's monitors. Returns empty list on failure."""
    try:
        module = __import__("poopy.companies", fromlist=[name])
        cls = getattr(module, name)
        company = cls()
        monitors = company.active_monitors
 
        rows = []
        for key, m in monitors.items():
            rows.append({
                "wb_id":        getattr(m, "receiving_watercourse", None) or key,
                "outlet_ngr":   key,
                "company":      name,
                "status":       str(getattr(m, "current_status", "Unknown")),
                "duration_hrs": 0.0,   # current status snapshot, not duration yet
            })
        print(f"  {name}: {len(rows)} monitors")
        return rows
 
    except Exception as e:
        print(f"  {name}: failed - {e}")
        return []
 
 
def main():
    print("Pulling current POOPy status from all companies...")
    all_rows = []
 
    for company in COMPANIES:
        rows = pull_one_company(company)
        all_rows.extend(rows)
 
    if not all_rows:
        print("No data pulled from any company.")
        return
 
    df = pd.DataFrame(all_rows)
    print(f"\nTotal monitors pulled: {len(df)}")
    print(df["company"].value_counts())
    print(df["status"].value_counts())
 
    # append to staging_edm_live
    load_staging_edm_live(df)
    print("\nSaved to staging_edm_live table.")
 
 
if __name__ == "__main__":
    main()