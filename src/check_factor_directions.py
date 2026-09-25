"""
check_factor_directions.py
Verifies whether specific SHAP factor directions reflect genuine patterns
in the real data, or need further investigation - same diagnostic
approach used to find and fix the sewage monitoring-density bias.

Checks:
    1. SHAP correlation direction for each feature (population-level)
    2. Actual median values by ground-truth status (does the real data
       support the direction SHAP is showing, or contradict it?)

Run:  python src/check_factor_directions.py
"""

import numpy as np
import pandas as pd
from db_loader import get_conn

shap_vals = np.load("results/shap_values.npy")
shap_feats = pd.read_csv("results/shap_input_features.csv")

FEATURES_TO_CHECK = [
    "lc_water_1km", "lc_water_5km",
    "lc_woodland_1km", "lc_woodland_5km",
    "lc_urban_1km", "lc_urban_5km",
    "lc_grass_1km", "lc_grass_5km",
]

print("=" * 70)
print("STEP 1: SHAP correlation direction (population-level, all 36,280 points)")
print("=" * 70)
print("Positive = pushes toward Poor.")
print()

for feat in FEATURES_TO_CHECK:
    if feat not in shap_feats.columns:
        continue
    idx = shap_feats.columns.get_loc(feat)
    corr = np.corrcoef(shap_feats[feat].values, shap_vals[:, idx])[0, 1]
    print(f"  {feat:20s}  SHAP corr = {corr:+.3f}")

print()
print("=" * 70)
print("STEP 2: Real median values by ACTUAL ground-truth status")
print("=" * 70)
print("If SHAP says 'more of X pushes toward Poor', Poor's median for X")
print("should genuinely be higher than Moderate's median, in real data.")
print()

conn = get_conn()
fm = pd.read_sql(f"""
    SELECT wfd_status, {', '.join(FEATURES_TO_CHECK)}
    FROM feat_matrix
""", conn)
conn.close()

medians = fm.groupby("wfd_status")[FEATURES_TO_CHECK].median().T
medians["Poor_higher_than_Moderate"] = medians["Poor"] > medians["Moderate"]
print(medians.round(2))

print()
print("=" * 70)
print("INTERPRETATION")
print("=" * 70)
print("If a feature's SHAP corr is POSITIVE (pushes toward Poor) AND")
print("Poor's real median IS higher than Moderate's - the direction is")
print("genuinely supported by real data, not a bug.")
print()
print("If SHAP corr is positive but Poor's median is LOWER than Moderate's")
print("(or vice versa) - that's a real inconsistency worth investigating")
print("further, similar to the sewage monitoring-density bias found earlier.")
