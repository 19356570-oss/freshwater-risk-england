"""
Feature Engineering Pipeline
Freshwater Ecological Risk Prediction — England
COMP7039 MSc Dissertation
Student: Surumimol Madathiparambil Shajahan (19356570)

This script builds the feature matrix from 4 data sources:
1. FreshWater Watch (FWW) — water chemistry
2. EA WFD Classifications — ground truth labels
3. EA EDM Storm Overflow — sewage pressure features
4. UKCEH Land Cover 2024 — land use buffer features

IMPORTANT: This same build_features() function must be used
for both historical training AND live dashboard inference.
Never change aggregation logic after model training.
"""

import pandas as pd
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.mask import mask
from pyproj import Transformer
from shapely.geometry import Point
from scipy.spatial import cKDTree
import os
import warnings
warnings.filterwarnings('ignore')

# ── CONFIG ────────────────────────────────────────────────────────────────────
RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"

FWW_FILE = os.path.join(RAW_DIR, "Global_Data_Set_XvsX_0.csv") #FreshWater Watch data
WFD_FILE = os.path.join(RAW_DIR, "EA WFD England_classifications.csv") #EA WFD classifications
EDM_2024_FILE = os.path.join(RAW_DIR, "EDM 2024 Storm Overflow Annual Return - all water and sewerage companies.xlsx") #EDM 2024 Storm Overflow data
EDM_2025_FILE = os.path.join(RAW_DIR, "EDM 2025 Storm Overflow Annual Return - all water and sewerage companies.xlsx") #EDM 2025 Storm Overflow data
LANDCOVER_FILE = os.path.join(RAW_DIR, "gblcm2024_25m.tif") #UKCEH Land Cover data

BUFFER_1KM = 1000   # metres
BUFFER_5KM = 5000   # metres

# UKCEH Land Cover class codes → names
# Full 21-class legend for LCM 2024
LANDCOVER_CLASSES = {
    1:  "broadleaved_woodland",
    2:  "coniferous_woodland",
    3:  "arable",
    4:  "improved_grassland",
    5:  "neutral_grassland",
    6:  "calcareous_grassland",
    7:  "acid_grassland",
    8:  "fen_marsh_swamp",
    9:  "heather",
    10: "heather_grassland",
    11: "bog",
    12: "inland_rock",
    13: "saltwater",
    14: "freshwater",
    15: "supralittoral_rock",
    16: "supralittoral_sediment",
    17: "littoral_rock",
    18: "littoral_sediment",
    19: "saltmarsh",
    20: "urban",
    21: "suburban",
}

# Grouped classes for ML features
LANDCOVER_GROUPS = {
    "pct_woodland":   [1, 2],
    "pct_arable":     [3],
    "pct_grassland":  [4, 5, 6, 7],
    "pct_wetland":    [8, 11, 19],
    "pct_urban":      [20, 21],
    "pct_freshwater": [14],
}

# ── STEP 1: LOAD AND CLEAN FWW ────────────────────────────────────────────────

def load_fww(filepath=FWW_FILE):
    """
    Load FreshWater Watch data, filter to England,
    select relevant columns, convert coordinates to BNG.
    """
    print("Loading FreshWater Watch data...")
    df = pd.read_csv(filepath, low_memory=False)

    # Filter to England only
    df = df[df['Country'] == 'England'].copy()
    print(f"  England records: {len(df)}")

    # Drop rows with no coordinates
    df = df.dropna(subset=['x', 'y'])

    # Drop rows with no chemistry readings
    df = df.dropna(subset=['Nitrate (mg/L) MID', 'Phosphate (mg/L) MID'])

    print(f"  Records after dropping nulls: {len(df)}")

    # Select and rename columns
    fww = df[[
        'ObjectID',
        'Site Name',
        'Sample Date',
        'x',                        # longitude (WGS84)
        'y',                        # latitude (WGS84)
        'Nitrate (mg/L) MID',
        'Phosphate (mg/L) MID',
        'Turbidity (NTU)',
        'pH',
        'Conductivity (µS cm-1) Europe',
        'Dissolved oxygen (mg/L)',
        'Water temperature (°C)',
        'Feedback Rating',          # FWW's own ecological rating
        'County',
        'RBD_NAME',                 # River Basin District
    ]].copy()

    fww.columns = [
        'fww_id', 'site_name', 'sample_date',
        'lon', 'lat',
        'nitrate_mid', 'phosphate_mid', 'turbidity_ntu',
        'ph', 'conductivity', 'dissolved_oxygen', 'water_temp',
        'fww_feedback_rating',
        'county', 'river_basin_district',
    ]

    # Parse date
    fww['sample_date'] = pd.to_datetime(fww['sample_date'], errors='coerce')

    # Convert lat/lon (WGS84) → Easting/Northing (BNG EPSG:27700)
    print("  Converting coordinates to British National Grid...")
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:27700", always_xy=True)
    fww['easting'], fww['northing'] = transformer.transform(
        fww['lon'].values, fww['lat'].values
    )

    print(f"  FWW loaded: {len(fww)} records")
    return fww


# ── STEP 2: LOAD AND CLEAN WFD ───────────────────────────────────────────────

def load_wfd(filepath=WFD_FILE):
    """
    Load EA WFD Classifications.
    Filter to Cycle 3 (2022), Overall Waterbody status.
    Returns one row per waterbody with Good/Moderate/Poor label.
    """
    print("Loading EA WFD Classifications...")
    df = pd.read_csv(filepath, low_memory=False)
    print(f"  Total rows: {len(df)}")

    # Filter to Cycle 3 (most recent — 2022)
    df = df[df['Cycle'] == 3]

    # Filter to Overall Waterbody classification only
    df = df[df['Classification Level'] == 'Overall Waterbody']

    # Filter to River waterbodies only (your scope)
    df = df[df['Water Body Type'] == 'River']

    # Keep only relevant status categories
    valid_status = ['Good', 'Moderate', 'Poor', 'Bad', 'High']
    df = df[df['Status'].isin(valid_status)]

    # Simplify to 3 classes: Good, Moderate, Poor
    # Bad → Poor, High → Good
    status_map = {
        'High': 'Good',
        'Good': 'Good',
        'Moderate': 'Moderate',
        'Poor': 'Poor',
        'Bad': 'Poor',
    }
    df['wfd_status'] = df['Status'].map(status_map)

    # Select columns
    wfd = df[[
        'Water Body ID',
        'Water Body',
        'Easting',
        'Northing',
        'River Basin District',
        'Management Catchment',
        'Operational Catchment',
        'wfd_status',
    ]].copy()

    wfd.columns = [
        'waterbody_id', 'waterbody_name',
        'easting', 'northing',
        'river_basin_district', 'management_catchment',
        'operational_catchment',
        'wfd_status',
    ]

    # Drop duplicates — keep one record per waterbody
    wfd = wfd.drop_duplicates(subset=['waterbody_id'])

    print(f"  WFD waterbodies (rivers, Cycle 3): {len(wfd)}")
    print(f"  Status distribution:\n{wfd['wfd_status'].value_counts()}")
    return wfd


# ── STEP 3: LOAD AND AGGREGATE EDM ───────────────────────────────────────────
def load_edm(edm_2024_path=EDM_2024_FILE, edm_2025_path=EDM_2025_FILE):
    print("Loading EA EDM Storm Overflow data...")

    # ── Helper: normalise column names (strip newlines and whitespace) ────────
    def clean_cols(df):
        df.columns = [str(c).replace('\n', ' ').strip() for c in df.columns]
        return df

    # ── 2024 ──────────────────────────────────────────────────────────────────
    xl_2024 = pd.ExcelFile(edm_2024_path)
    dfs_2024 = []
    for sheet in xl_2024.sheet_names:
        try:
            df = pd.read_excel(edm_2024_path, sheet_name=sheet, header=1)
            df = clean_cols(df)
            df['data_year'] = 2024
            dfs_2024.append(df)
        except Exception as e:
            print(f"  Warning 2024 sheet '{sheet}': {e}")
    edm_2024 = pd.concat(dfs_2024, ignore_index=True)

    # Duration: 2024 stores as timedelta-compatible string
    dur_col = 'Total Duration (hh:mm:ss) all spills prior to processing through 12-24h count method'
    edm_2024['duration_hours'] = pd.to_timedelta(
        edm_2024[dur_col], errors='coerce'
    ).dt.total_seconds() / 3600

    # ── 2025 ──────────────────────────────────────────────────────────────────
    xl_2025 = pd.ExcelFile(edm_2025_path)
    dfs_2025 = []
    for sheet in xl_2025.sheet_names:
        try:
            df = pd.read_excel(edm_2025_path, sheet_name=sheet, header=1)
            df = clean_cols(df)

            # Skip sheets that are clearly not data
            # (e.g. All WaSC summary sheet with only 4 rows)
            if len(df) < 5:
                print(f"  Skipping 2025 sheet '{sheet}' (only {len(df)} rows)")
                continue

            # Fix offset headers — if first col is not a known header
            if df.columns[0] not in ['Unique ID', 'Water Company Name']:
                df2 = pd.read_excel(edm_2025_path, sheet_name=sheet, header=None)
                df2 = clean_cols(df2)
                for i, row in df2.iterrows():
                    if 'Unique ID' in str(row.values):
                        df2.columns = [str(v).replace('\n', ' ').strip()
                                       for v in df2.iloc[i]]
                        df = df2.iloc[i+1:].reset_index(drop=True)
                        break

            df['data_year'] = 2025
            dfs_2025.append(df)
        except Exception as e:
            print(f"  Warning 2025 sheet '{sheet}': {e}")

    edm_2025 = pd.concat(dfs_2025, ignore_index=True)
    edm_2025 = edm_2025.drop_duplicates(subset=['Unique ID'])

    # Duration: 2025 stores as string
    def parse_dur(s):
        try:
            return pd.to_timedelta(str(s)).total_seconds() / 3600
        except:
            return 0.0

    edm_2025['duration_hours'] = edm_2025[dur_col].apply(parse_dur)

    # ── COMBINE ───────────────────────────────────────────────────────────────
    # Note: use cleaned column names (newlines already stripped above)
    common_cols = [
        'Unique ID',
        'Water Company Name',
        'WFD Waterbody ID (Cycle 3) (discharge outlet)',
        'Outlet Discharge NGR (EA Consents Database)',
        'Counted spills using 12-24h count method',
        'Long-term average spill count',
        'EDM Operation - % of reporting period EDM operational',
        'duration_hours',
        'data_year',
    ]

    s2024 = edm_2024[[c for c in common_cols if c in edm_2024.columns]].copy()
    s2025 = edm_2025[[c for c in common_cols if c in edm_2025.columns]].copy()

    print(f"  2024 cols matched: {list(s2024.columns)}")
    print(f"  2025 cols matched: {list(s2025.columns)}")

    edm_combined = pd.concat([s2024, s2025], ignore_index=True)

    edm_combined = edm_combined.rename(columns={
        'WFD Waterbody ID (Cycle 3) (discharge outlet)': 'waterbody_id',
        'Outlet Discharge NGR (EA Consents Database)':   'outlet_ngr',
        'Counted spills using 12-24h count method':      'spill_count',
        'Long-term average spill count':                 'long_term_avg_spill_count',
        'EDM Operation - % of reporting period EDM operational': 'edm_operational_pct',
    })

    if 'waterbody_id' not in edm_combined.columns:
        raise ValueError(
            f"'waterbody_id' still missing. Available: {list(edm_combined.columns)}"
        )

    for col in ['spill_count', 'duration_hours', 'long_term_avg_spill_count']:
        if col in edm_combined.columns:
            edm_combined[col] = pd.to_numeric(
                edm_combined[col], errors='coerce'
            ).fillna(0)

    print(f"  EDM combined: {len(edm_combined)} records "
          f"(2024: {len(s2024)}, 2025: {len(s2025)})")

    # ── AGGREGATE PER WATERBODY ───────────────────────────────────────────────
    edm_agg = edm_combined.groupby('waterbody_id').agg(
        edm_total_spill_count    = ('spill_count', 'sum'),
        edm_total_duration_hours = ('duration_hours', 'sum'),
        edm_avg_long_term_spills = ('long_term_avg_spill_count', 'mean'),
        edm_n_overflows          = ('spill_count', 'count'),
        edm_mean_operational_pct = ('edm_operational_pct', 'mean'),
        edm_years_covered        = ('data_year', lambda x: ','.join(
                                       map(str, sorted(x.unique())))),
    ).reset_index()

    print(f"  EDM aggregated: {len(edm_agg)} waterbodies")
    return edm_agg

def load_edm1(edm_2024_path=EDM_2024_FILE, edm_2025_path=EDM_2025_FILE):
    """
    Load EDM 2024 and 2025 files, handle structural differences, combine and aggregate.
    2024: 28 cols, separate sheet per company, timedelta duration
    2025: 26 cols, has All WaSC combined sheet + individual sheets (some offset headers),
          duration stored as string
    """
    print("Loading EA EDM Storm Overflow data...")

    # ── 2024 ─────────────────────────────────────────────────────────────────
    xl_2024 = pd.ExcelFile(edm_2024_path)
    dfs_2024 = []
    for sheet in xl_2024.sheet_names:
        try:
            df = pd.read_excel(edm_2024_path, sheet_name=sheet,header=1)
            df['data_year'] = 2024
            dfs_2024.append(df)
        except Exception as e:
            print(f"  Warning 2024 sheet {sheet}: {e}")
    edm_2024 = pd.concat(dfs_2024, ignore_index=True)
    edm_2024['duration_hours'] = pd.to_timedelta(
        edm_2024['Total Duration (hh:mm:ss) all spills prior to processing through 12-24h count method'],
        errors='coerce'
    ).dt.total_seconds() / 3600

    # ── 2025 ─────────────────────────────────────────────────────────────────
    xl_2025 = pd.ExcelFile(edm_2025_path)
    dfs_2025 = []
    for sheet in xl_2025.sheet_names:
        try:
            df = pd.read_excel(edm_2025_path, sheet_name=sheet,header=1)
            # Fix offset headers (some 2025 sheets have company name as first row)
            if df.columns[0] not in ['Unique ID', 'Water Company Name'] and 'Unnamed' not in str(df.columns[0]):
                df2 = pd.read_excel(edm_2025_path, sheet_name=sheet, header=None)
                for i, row in df2.iterrows():
                    if 'Unique ID' in str(row.values):
                        df2.columns = df2.iloc[i]
                        df = df2.iloc[i+1:].reset_index(drop=True)
                        break
            df['data_year'] = 2025
            dfs_2025.append(df)
        except Exception as e:
            print(f"  Warning 2025 sheet {sheet}: {e}")
    edm_2025 = pd.concat(dfs_2025, ignore_index=True)
    # Deduplicate — All WaSC sheet duplicates individual company sheets
    edm_2025 = edm_2025.drop_duplicates(subset=['Unique ID'])

    def parse_dur(s):
        try:
            return pd.to_timedelta(str(s)).total_seconds() / 3600
        except:
            return 0.0

    edm_2025['duration_hours'] = edm_2025[
        'Total Duration (hh:mm:ss) all spills prior to processing through 12-24h count method'
    ].apply(parse_dur)

    # ── COMBINE ───────────────────────────────────────────────────────────────────
    common_cols = [
    'Unique ID', 'Water Company Name',
    'WFD Waterbody ID (Cycle 3) (discharge outlet)',
    'Outlet Discharge NGR (EA Consents Database)',
    'Counted spills using 12-24h count method',
    'Long-term average spill count',
    'EDM Operation - % of reporting period EDM operational',
    'duration_hours', 'data_year',
]
    s2024 = edm_2024[[c for c in common_cols if c in edm_2024.columns]].copy()
    s2025 = edm_2025[[c for c in common_cols if c in edm_2025.columns]].copy()
    edm_combined = pd.concat([s2024, s2025], ignore_index=True)

    # ── DEBUG: print columns to confirm rename targets exist ──────────────────────
    print(f"  EDM combined columns: {list(edm_combined.columns)}")

    # ── RENAME only if column exists ──────────────────────────────────────────────
    rename_map = {
    'WFD Waterbody ID (Cycle 3) (discharge outlet)': 'waterbody_id',
    'Outlet Discharge NGR (EA Consents Database)':   'outlet_ngr',
    'Counted spills using 12-24h count method':      'spill_count',
    'Long-term average spill count':                 'long_term_avg_spill_count',
    'EDM Operation - % of reporting period EDM operational': 'edm_operational_pct',
    }
    edm_combined = edm_combined.rename(columns=rename_map)

    # ── Confirm waterbody_id exists before groupby ────────────────────────────────
    if 'waterbody_id' not in edm_combined.columns:
     raise ValueError(
         "Column 'waterbody_id' missing after rename. "
         f"Available columns: {list(edm_combined.columns)}"
     )

    # ── Convert numeric columns ───────────────────────────────────────────────────
    for col in ['spill_count', 'duration_hours', 'long_term_avg_spill_count']:
      if col in edm_combined.columns:
           edm_combined[col] = pd.to_numeric(edm_combined[col], errors='coerce').fillna(0)

    # ── AGGREGATE PER WATERBODY ───────────────────────────────────────────────────
    edm_agg = edm_combined.groupby('waterbody_id').agg(
    edm_total_spill_count    = ('spill_count', 'sum'),
    edm_total_duration_hours = ('duration_hours', 'sum'),
    edm_avg_long_term_spills = ('long_term_avg_spill_count', 'mean'),
    edm_n_overflows          = ('spill_count', 'count'),
    edm_mean_operational_pct = ('edm_operational_pct', 'mean'),
    edm_years_covered        = ('data_year', lambda x: ','.join(map(str, sorted(x.unique())))),
    ).reset_index()

    print(f"  EDM aggregated: {len(edm_agg)} waterbodies")
    return edm_agg



# ── STEP 4: SPATIAL JOIN FWW → WFD ──────────────────────────────────────────

def join_fww_to_wfd(fww, wfd):
    """
    Nearest-neighbour spatial join.
    Match each FWW sampling point to its nearest WFD waterbody centroid.
    Uses BNG Easting/Northing for both.
    """
    print("Spatial join: FWW → WFD (nearest neighbour)...")

    # Build KD-tree from WFD centroids
    wfd_coords = np.array(list(zip(wfd['easting'], wfd['northing'])))
    tree = cKDTree(wfd_coords)

    # Query nearest WFD centroid for each FWW point
    fww_coords = np.array(list(zip(fww['easting'], fww['northing'])))
    distances, indices = tree.query(fww_coords, k=1)

    fww = fww.copy()
    fww['waterbody_id'] = wfd['waterbody_id'].iloc[indices].values
    fww['waterbody_name'] = wfd['waterbody_name'].iloc[indices].values
    fww['wfd_status'] = wfd['wfd_status'].iloc[indices].values
    fww['nearest_wfd_dist_m'] = distances

    # Flag points far from any waterbody (>5km — likely noise)
    fww['wfd_match_quality'] = np.where(
        distances <= 1000, 'good',
        np.where(distances <= 5000, 'acceptable', 'poor')
    )

    print(f"  Match quality:\n{fww['wfd_match_quality'].value_counts()}")
    return fww


# ── STEP 5: JOIN EDM FEATURES ────────────────────────────────────────────────

def join_edm_features(fww, edm_agg):
    """
    Join aggregated EDM sewage features to FWW points via waterbody_id.
    Fill missing with 0 (no recorded overflows = no pressure).
    """
    print("Joining EDM sewage features...")
    fww = fww.merge(edm_agg, on='waterbody_id', how='left')

    # Fill nulls — waterbodies with no EDM record had no recorded spills
    edm_cols = [
        'edm_total_spill_count', 'edm_total_duration_hours',
        'edm_avg_long_term_spills', 'edm_n_overflows',
        'edm_mean_operational_pct'
    ]
    fww[edm_cols] = fww[edm_cols].fillna(0)

    print(f"  EDM joined. Records with spill data: {(fww['edm_total_spill_count'] > 0).sum()}")
    return fww


# ── STEP 6: EXTRACT LAND COVER FEATURES ─────────────────────────────────────

def extract_landcover_features(fww, landcover_path=LANDCOVER_FILE):
    """
    For each FWW point, extract % of each land cover group
    within 1km and 5km buffer zones.
    Uses rasterio to read UKCEH LCM 2024 25m raster.
    """
    print("Extracting land cover buffer features...")
    print("  This may take several minutes for large datasets...")

    # Initialise output columns
    for buf_label in ['1km', '5km']:
        for group_name in LANDCOVER_GROUPS.keys():
            fww[f"lc_{group_name}_{buf_label}"] = np.nan

    with rasterio.open(landcover_path) as src:
        print(f"  Raster CRS: {src.crs}")
        print(f"  Raster shape: {src.shape}")

        for idx, row in fww.iterrows():
            point = Point(row['easting'], row['northing'])

            for buf_m, buf_label in [(BUFFER_1KM, '1km'), (BUFFER_5KM, '5km')]:
                buffer_geom = point.buffer(buf_m)

                try:
                    # Mask raster to buffer polygon
                    out_image, _ = mask(src, [buffer_geom], crop=True)
                    pixels = out_image[0].flatten()

                    # Remove nodata values
                    pixels = pixels[pixels != src.nodata]
                    total = len(pixels)

                    if total == 0:
                        continue

                    # Calculate % per group
                    for group_name, class_codes in LANDCOVER_GROUPS.items():
                        count = np.isin(pixels, class_codes).sum()
                        fww.at[idx, f"lc_{group_name}_{buf_label}"] = (count / total) * 100

                except Exception:
                    pass

            if idx % 100 == 0:
                print(f"  Processed {idx}/{len(fww)} points...")

    print("  Land cover extraction complete.")
    return fww


# ── STEP 7: FINAL FEATURE MATRIX ─────────────────────────────────────────────

def build_feature_matrix(fww):
    """
    Select final features for ML model.
    Drop rows with missing labels or key features.
    This is the CANONICAL feature set — do not change after training.
    """
    print("Building final feature matrix...")

    feature_cols = [
        # Identity
        'fww_id', 'site_name', 'sample_date',
        'easting', 'northing', 'waterbody_id',

        # Water chemistry features (FWW)
        'nitrate_mid', 'phosphate_mid', 'turbidity_ntu',
        'ph', 'conductivity', 'dissolved_oxygen', 'water_temp',

        # Sewage pressure features (EDM)
        'edm_total_spill_count', 'edm_total_duration_hours',
        'edm_avg_long_term_spills', 'edm_n_overflows',

        # Land cover features — 1km buffer
        'lc_pct_woodland_1km', 'lc_pct_arable_1km',
        'lc_pct_grassland_1km', 'lc_pct_wetland_1km',
        'lc_pct_urban_1km', 'lc_pct_freshwater_1km',

        # Land cover features — 5km buffer
        'lc_pct_woodland_5km', 'lc_pct_arable_5km',
        'lc_pct_grassland_5km', 'lc_pct_wetland_5km',
        'lc_pct_urban_5km', 'lc_pct_freshwater_5km',

        # Metadata
        'county', 'river_basin_district', 'nearest_wfd_dist_m',
        'wfd_match_quality',

        # Target label
        'wfd_status',
    ]

    # Keep only columns that exist
    available_cols = [c for c in feature_cols if c in fww.columns]
    matrix = fww[available_cols].copy()

    # Drop rows with no label
    matrix = matrix.dropna(subset=['wfd_status'])

    # Drop poor quality spatial matches
    matrix = matrix[matrix['wfd_match_quality'] != 'poor']

    print(f"  Final feature matrix: {len(matrix)} rows x {len(matrix.columns)} columns")
    print(f"  Label distribution:\n{matrix['wfd_status'].value_counts()}")

    return matrix


# ── MAIN PIPELINE ─────────────────────────────────────────────────────────────

def run_pipeline(skip_landcover=False):
    """
    Run full feature engineering pipeline.
    Set skip_landcover=True to run without land cover (faster for testing).
    """
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    # Step 1: Load data
    fww = load_fww()
    wfd = load_wfd()
    edm_agg = load_edm(EDM_2024_FILE, EDM_2025_FILE)

    # Step 2: Spatial join FWW → WFD
    fww = join_fww_to_wfd(fww, wfd)

    # Step 3: Join EDM features
    fww = join_edm_features(fww, edm_agg)

    # Step 4: Extract land cover features
    if not skip_landcover:
        fww = extract_landcover_features(fww)
    else:
        print("Skipping land cover extraction (skip_landcover=True)")

    # Step 5: Build final feature matrix
    matrix = build_feature_matrix(fww)

    # Save outputs
    output_path = os.path.join(PROCESSED_DIR, "feature_matrix.csv")
    matrix.to_csv(output_path, index=False)
    print(f"\nFeature matrix saved to: {output_path}")

    # Save intermediate FWW with all joins (useful for debugging)
    fww.to_csv(os.path.join(PROCESSED_DIR, "fww_joined.csv"), index=False)
    print(output_path)
    return matrix


if __name__ == "__main__":
    # Run without land cover first to test the pipeline quickly
    # Change to skip_landcover=False when ready for full run
    matrix = run_pipeline()#(skip_landcover=True)
    print("\nPipeline complete.")
    print(matrix.head())
#if __name__ == "__main__":
    # Quick test on 100 rows first
    #fww_test = load_fww()
    #wfd_test = load_wfd()
    #edm_test = load_edm(EDM_2024_FILE, EDM_2025_FILE)
    #fww_test = join_fww_to_wfd(fww_test, wfd_test)
    #fww_test = join_edm_features(fww_test, edm_test)
    #fww_test = fww_test.head(100)  # sample only
    #fww_test = extract_landcover_features(fww_test)
    #print(fww_test[[c for c in fww_test.columns if 'lc_' in c]].head())


  
