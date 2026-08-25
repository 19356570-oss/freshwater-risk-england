"""
config.py
Central configuration for freshwater risk prediction pipeline.
Change paths and thresholds here - nowhere else.
"""

import os

# -- Directories
RAW_DIR       = "data/raw"
PROCESSED_DIR = "data/processed"
MODELS_DIR    = "models"
RESULTS_DIR   = "results"
SNAPSHOT_DIR  = "data/snapshots"
DB_PATH       = "data/freshwater_risk.db"

# -- Raw data files
FWW_FILE      = os.path.join(RAW_DIR, "Global_Data_Set_XvsX_0.csv")
WFD_FILE      = os.path.join(RAW_DIR, "EA WFD England_classifications.csv")
EDM_2024_FILE = os.path.join(RAW_DIR, "EDM 2024 Storm Overflow Annual Return - all water and sewerage companies.xlsx")
EDM_2025_FILE = os.path.join(RAW_DIR, "EDM 2025 Storm Overflow Annual Return - all water and sewerage companies.xlsx")
LC_FILE       = os.path.join(RAW_DIR, "gblcm2024_25m.tif")

# -- Processed outputs
FEAT_MATRIX   = os.path.join(PROCESSED_DIR, "feature_matrix.csv")
FWW_JOINED    = os.path.join(PROCESSED_DIR, "fww_joined.csv")

# -- Model outputs
MODEL_PATH    = os.path.join(MODELS_DIR, "rf_model.pkl")      # updated after training
ENCODER_PATH  = os.path.join(MODELS_DIR, "label_encoder.pkl")
SHAP_PATH     = os.path.join(RESULTS_DIR, "shap_values.npy")
SHAP_FEATS    = os.path.join(RESULTS_DIR, "shap_input_features.csv")
RESULTS_JSON  = os.path.join(RESULTS_DIR, "modelling_results.json")

# -- Coordinate systems
CRS_WGS84 = "EPSG:4326"   # lat/lon - used by FreshWater Watch
CRS_BNG   = "EPSG:27700"  # British National Grid - used by EA/WFD

# -- Buffer sizes for land cover extraction (metres)
BUF_1KM = 1000
BUF_5KM = 5000

# -- WFD regulatory thresholds (EA UKTAG standards)
# Rivers exceeding these are classified as Poor ecological status
PO4_THRESHOLD = 0.1   # phosphate mg/L
NO3_THRESHOLD = 2.0   # nitrate mg/L

# -- Model settings
N_FOLDS      = 5     # spatial k-fold CV
RANDOM_STATE = 42
N_TREES      = 200   # RF and XGBoost estimators

# -- WFD status mapping (5 classes → 3 classes)
# High and Good both map to Good
# Bad maps to Poor
STATUS_MAP = {
    "High":     "Good",
    "Good":     "Good",
    "Moderate": "Moderate",
    "Poor":     "Poor",
    "Bad":      "Poor",
}

# -- Feature groups (used in training and ablation experiments)
CHEM_FEATS = [
    "nitrate_mid",
    "phosphate_mid",
]

SEWAGE_FEATS = [
    "avg_spills",
    "n_overflows",
    "spills_per_pipe",   # spill_count / n_overflows - removes monitor-density bias
]
# NOTE: raw spill_count and spill_hrs deliberately excluded from ALL_FEATS.
# Verified empirically (13 Aug) that they correlate NEGATIVELY with Poor status
# (corr = -0.493) because EDM monitor density is higher in urban catchments,
# not because more spills indicate better water quality. spills_per_pipe
# corrects for this (corr = +0.392, verified on final 18-feature model) and
# is used instead. Raw columns remain in feature_matrix.csv for transparency
# but are not fed to the model.
#
# hrs_per_pipe also excluded (17 Aug). Tested via test_hrs_per_pipe.py:
# F1 with = 0.6583, F1 without = 0.6584 - no measurable contribution to
# accuracy, and its SHAP correlation was near-zero and inconsistent
# (-0.042). Discharge frequency (spills_per_pipe) appears to be the more
# informative signal than discharge duration for this task.

LC_FEATS = [
    "lc_woodland_1km", "lc_arable_1km", "lc_grass_1km",
    "lc_wetland_1km",  "lc_urban_1km",  "lc_water_1km",
    "lc_woodland_5km", "lc_arable_5km", "lc_grass_5km",
    "lc_wetland_5km",  "lc_urban_5km",  "lc_water_5km",
]

ALL_FEATS = CHEM_FEATS + SEWAGE_FEATS + LC_FEATS

# -- Ablation sets (keys used in results JSON)
ABLATION_SETS = {
    "chemistry_only":        CHEM_FEATS,
    "chemistry_plus_sewage": CHEM_FEATS + SEWAGE_FEATS,
    "full_feature_set":      ALL_FEATS,
}

# -- UKCEH LCM 2024 class codes grouped into 6 categories
LC_GROUPS = {
    "lc_woodland": [1, 2],        # broadleaved + coniferous
    "lc_arable":   [3],           # arable farmland
    "lc_grass":    [4, 5, 6, 7],  # improved/neutral/calcareous/acid grassland
    "lc_wetland":  [8, 11, 19],   # fen/marsh/swamp, bog, saltmarsh
    "lc_urban":    [20, 21],      # urban + suburban
    "lc_water":    [14],          # freshwater
}

# -- Data source modes
# Switch from "historical" to "live" when deploying dashboard with POOPy
DATA_MODE = "historical"   # options: "historical" | "live"

# -- Scheduling intervals (seconds) - used by APScheduler in dashboard
SCHEDULE_FWW_SECS  = 604800   # weekly (7 days)
SCHEDULE_EDM_SECS  = 900      # every 15 minutes (POOPy update frequency)
SCHEDULE_SNAP_SECS = 604800   # weekly SQLite snapshot