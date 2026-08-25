#data_loader.py

#Loads and cleans raw data from all four sources.
#Switch DATA_MODE in config.py to swap historical for live POOPy data.


import pandas as pd
import numpy as np
from pyproj import Transformer
from config import (
    FWW_FILE, WFD_FILE, EDM_2024_FILE, EDM_2025_FILE,
    CRS_WGS84, CRS_BNG, STATUS_MAP, DATA_MODE
)


# FreshWater Watch

def load_fww(path=FWW_FILE):
    #Load FWW CSV, filter to England, convert coords to BNG.

    print("Loading FreshWater Watch...")
    df = pd.read_csv(path, low_memory=False)

    df = df[df["Country"] == "England"].copy()  # Filter England only
    df = df.dropna(subset=["x", "y"]) #drop rows with no coordinates
    df = df.dropna(subset=["Nitrate (mg/L) MID", "Phosphate (mg/L) MID"]) #drop rows with no chemistry data
    print(f"  {len(df)} records after filtering")

    #keeping only what we need
    df = df[[
        "ObjectID", "Site Name", "Sample Date",
        "x", "y", #longitude, latitude WGS84
        "Nitrate (mg/L) MID", "Phosphate (mg/L) MID",
        "Turbidity (NTU)", "pH",
        "Conductivity (µS cm-1) Europe",
        "Dissolved oxygen (mg/L)",
        "Water temperature (°C)",
        "Feedback Rating" , #fww's own ecological rating 
        "County", "RBD_NAME", #river basin district
    ]].copy()
    #rename columns to match other datasets
    df.columns = [
        "fww_id", "site_name", "sample_date",
        "lon", "lat",
        "nitrate_mid", "phosphate_mid",
        "turbidity", "ph", "conductivity",
        "dissolved_o2", "water_temp",
        "fww_rating", "county", "rbd",
    ]
    #convert sample_date to datetime, coerce errors to NaT
    df["sample_date"] = pd.to_datetime(df["sample_date"], errors="coerce")

    #convert lat/lon to British National Grid
    proj = Transformer.from_crs(CRS_WGS84, CRS_BNG, always_xy=True)
    df["easting"], df["northing"] = proj.transform(df["lon"].values, df["lat"].values)

    print(f"  FWW loaded: {len(df)} records")
    return df


# EA WFD Classifications

def load_wfd(path=WFD_FILE):
    #Load WFD CSV, filter to Cycle 3 rivers, map to 3-class label.

    print("Loading EA WFD classifications...")
    df = pd.read_csv(path, low_memory=False)
    print(f"  Total rows: {len(df)}")

    df = df[df["Cycle"] == 3]                                    #2022 cycle only
    df = df[df["Classification Level"] == "Overall Waterbody"]   #one row per waterbody
    df = df[df["Water Body Type"] == "River"]                    #rivers only
    df = df[df["Status"].isin(STATUS_MAP.keys())]                #drop unknown statuses

    df["wfd_status"] = df["Status"].map(STATUS_MAP)  # 5 classes -> 3

    df = df[[
        "Water Body ID", "Water Body",
        "Easting", "Northing",
        "River Basin District",
        "Management Catchment",
        "Operational Catchment",
        "wfd_status",
    ]].copy()

    df.columns = [
        "wb_id", "wb_name",
        "easting", "northing",
        "rbd", "mgmt_catchment",
        "op_catchment", "wfd_status",
    ]

    df = df.drop_duplicates(subset=["wb_id"])  #drop duplicates, one row per waterbody

    print(f"  Waterbodies (rivers, Cycle 3): {len(df)}")
    print(f"  Status:\n{df['wfd_status'].value_counts()}")
    return df


# EA EDM Storm Overflow

def load_edm(path_2024=EDM_2024_FILE, path_2025=EDM_2025_FILE):
    #Load EDM 2024 and 2025, handle structural differences, aggregate per waterbody.
    #2024: header row 2, timedelta duration.
    #2025: some sheets offset headers, duration as string, All WaSC sheet skipped.

    print("Loading EA EDM storm overflow data...")

    def clean_cols(df):
        #strip newlines from column headers (common in EA Excel files)
        df.columns = [str(c).replace("\n", " ").strip() for c in df.columns]
        return df

    dur_col = "Total Duration (hh:mm:ss) all spills prior to processing through 12-24h count method"

    #2024
    xl = pd.ExcelFile(path_2024)
    dfs = []
    for sheet in xl.sheet_names:
        try:
            df = pd.read_excel(path_2024, sheet_name=sheet, header=1)
            df = clean_cols(df)
            df["data_year"] = 2024
            dfs.append(df)
        except Exception as e:
            print(f"  Skipping 2024 sheet '{sheet}': {e}")
    edm_2024 = pd.concat(dfs, ignore_index=True)
    edm_2024["spill_hrs"] = pd.to_timedelta(
        edm_2024[dur_col], errors="coerce"
    ).dt.total_seconds() / 3600  # convert timedelta to float hours

    #2025
    xl = pd.ExcelFile(path_2025)
    dfs = []
    for sheet in xl.sheet_names:
        try:
            df = pd.read_excel(path_2025, sheet_name=sheet, header=1)
            df = clean_cols(df)
            if len(df) < 5:
                print(f"  Skipping 2025 sheet '{sheet}' (summary sheet)")
                continue
            #some 2025 sheets have company name as first column instead of header
            if df.columns[0] not in ["Unique ID", "Water Company Name"]:
                df2 = pd.read_excel(path_2025, sheet_name=sheet, header=None)
                df2 = clean_cols(df2)
                for i, row in df2.iterrows():
                    if "Unique ID" in str(row.values):
                        df2.columns = [str(v).replace("\n", " ").strip() for v in df2.iloc[i]]
                        df = df2.iloc[i+1:].reset_index(drop=True)
                        break
            df["data_year"] = 2025
            dfs.append(df)
        except Exception as e:
            print(f"  Skipping 2025 sheet '{sheet}': {e}")
    edm_2025 = pd.concat(dfs, ignore_index=True)
    edm_2025 = edm_2025.drop_duplicates(subset=["Unique ID"])  #remove All WaSC duplicates
    edm_2025["spill_hrs"] = edm_2025[dur_col].apply(
        lambda s: pd.to_timedelta(str(s)).total_seconds() / 3600
        if pd.notna(s) else 0.0
    )

    #common columns in both data files (2024 and 2025)
    keep_cols = [
        "Unique ID", "Water Company Name",
        "WFD Waterbody ID (Cycle 3) (discharge outlet)",
        "Outlet Discharge NGR (EA Consents Database)",
        "Counted spills using 12-24h count method",
        "Long-term average spill count",
        "EDM Operation - % of reporting period EDM operational",
        "spill_hrs", "data_year",
    ]
    s24 = edm_2024[[c for c in keep_cols if c in edm_2024.columns]].copy()
    s25 = edm_2025[[c for c in keep_cols if c in edm_2025.columns]].copy()
    combined = pd.concat([s24, s25], ignore_index=True)

    combined = combined.rename(columns={
        "WFD Waterbody ID (Cycle 3) (discharge outlet)":        "wb_id",
        "Outlet Discharge NGR (EA Consents Database)":           "outlet_ngr",
        "Counted spills using 12-24h count method":              "spill_count",
        "Long-term average spill count":                         "avg_spills",
        "EDM Operation - % of reporting period EDM operational": "edm_pct",
    })

    if "wb_id" not in combined.columns:
        raise ValueError(f"wb_id missing. Available: {list(combined.columns)}")

    for col in ["spill_count", "spill_hrs", "avg_spills"]:
        if col in combined.columns:
            combined[col] = pd.to_numeric(combined[col], errors="coerce").fillna(0)

    print(f"  Combined EDM: {len(combined)} rows (2024: {len(s24)}, 2025: {len(s25)})")

    #aggregate to one row per waterbody
    agg = combined.groupby("wb_id").agg(
        spill_count=("spill_count", "sum"),       #total spills across all overflows
        spill_hrs=("spill_hrs", "sum"),            #total discharge hours
        avg_spills=("avg_spills", "mean"),         #long-term average
        n_overflows=("spill_count", "count"),      #number of overflow pipes
        edm_pct=("edm_pct", "mean"),               #% of year EDM was operational
        years=("data_year", lambda x: ",".join(map(str, sorted(x.unique())))),
    ).reset_index()

    print(f"  EDM aggregated: {len(agg)} waterbodies")
    return agg


# EDM Live data mode (POOPy)

def load_edm_live():
    #Load live EDM data via POOPy.
    #Called by APScheduler every 15 minutes in dashboard mode.
    #Aggregation matches load_edm() exactly for inference consistency.

    from poopy.companies import ThamesWater  # import here to avoid hard dependency
    raise NotImplementedError("Live EDM loading via POOPy - implement in Sprint 4")


def get_edm_data():
    #Route to historical or live EDM based on DATA_MODE in config.
    if DATA_MODE == "live":
        return load_edm_live()
    return load_edm()
