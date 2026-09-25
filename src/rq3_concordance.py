"""
rq3_concordance.py

RQ3: How concordant are FreshWater Watch's own citizen-science ratings
with the official Environment Agency WFD classifications?

This is a standalone analysis script - it does not modify feat_matrix,
the trained model, or any part of the existing pipeline. It just loads
the raw FWW rating fresh and compares it against the wfd_status already
sitting in your database.

Run:  python src/rq3_concordance.py
"""

import pandas as pd
from db_loader import get_conn

RAW_FWW_PATH = "data/raw/Global_Data_Set_XvsX_0.csv"

print("=" * 70)
print("RQ3: FreshWater Watch vs Environment Agency concordance")
print("=" * 70)

# ---- Load FWW's own rating, keyed by fww_id (== ObjectID) ------------------
print("\nLoading raw FWW ratings...")
fww_raw = pd.read_csv(RAW_FWW_PATH, usecols=["ObjectID", "Feedback Rating"], low_memory=False)
fww_raw.columns = ["fww_id", "fww_rating"]

# "No water available" isn't a real ecological rating - exclude it, same as
# the model itself does at the feature-engineering stage.
fww_raw = fww_raw[fww_raw["fww_rating"] != "No water available"]

# ---- Load the official WFD status already in feat_matrix ------------------
print("Loading official WFD status from feat_matrix...")
conn = get_conn()
wfd = pd.read_sql("SELECT fww_id, wfd_status FROM feat_matrix", conn)
conn.close()

# ---- Join on fww_id (== ObjectID, confirmed via data_loader.py) -----------
fww_raw["fww_id"] = fww_raw["fww_id"].astype(str)
wfd["fww_id"] = wfd["fww_id"].astype(str)
merged = fww_raw.merge(wfd, on="fww_id", how="inner")
print(f"\nMatched {len(merged):,} locations with both a FWW rating and an official WFD status.")

merged_mod_poor = merged[merged['fww_rating'] != 'Good']
agree = (merged_mod_poor['fww_rating'] == merged_mod_poor['wfd_status']).mean() * 100
print(f'Agreement among only Moderate/Poor FWW ratings: {agree:.1f}%')
print()
print(pd.crosstab(merged_mod_poor['fww_rating'], merged_mod_poor['wfd_status'], normalize='index') * 100)

# ---- Confusion matrix -------------------------------------------------------
print("\n" + "=" * 70)
print("CONFUSION MATRIX (rows = FWW's own rating, columns = official EA status)")
print("=" * 70)
confusion = pd.crosstab(merged["fww_rating"], merged["wfd_status"], margins=True, margins_name="Total")
print(confusion)

# ---- % exact agreement ------------------------------------------------------
# FWW rating and WFD status use the same three labels (Good/Moderate/Poor),
# so a direct string match is a fair, simple measure of exact agreement.
exact_match = (merged["fww_rating"] == merged["wfd_status"]).mean() * 100
print(f"\nExact agreement: {exact_match:.1f}% of locations have FWW's rating "
      f"exactly matching the official EA rating.")

# ---- Cohen's kappa -----------------------------------------------------------
# Kappa corrects for the agreement you'd expect by pure chance, given how
# common each category is - a fairer measure than raw % agreement alone.
try:
    from sklearn.metrics import cohen_kappa_score
    kappa = cohen_kappa_score(merged["fww_rating"], merged["wfd_status"])
    print(f"Cohen's kappa: {kappa:.3f}")
    print()
    if kappa < 0.20:
        strength = "slight"
    elif kappa < 0.40:
        strength = "fair"
    elif kappa < 0.60:
        strength = "moderate"
    elif kappa < 0.80:
        strength = "substantial"
    else:
        strength = "almost perfect"
    print(f"By the standard Landis & Koch (1977) benchmark, this represents "
          f"'{strength}' agreement.")
except ImportError:
    print("scikit-learn not available - install it to compute Cohen's kappa.")

print("\n" + "=" * 70)
print("Done. Use the confusion matrix and these two statistics directly in")
print("your RQ3 results section.")
print("=" * 70)