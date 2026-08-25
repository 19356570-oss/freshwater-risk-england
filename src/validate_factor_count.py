"""
validate_factor_counts.py
Standalone validation - NOT part of the dashboard.

Checks whether the number of SHAP factors pointing toward Poor actually
relates to the real predicted status, across the whole dataset. This is
a sanity check on the dashboard's "X of 18 factors" logic, independent
of any single example.

If the logic is meaningful, Poor-status locations should have MORE
factors pointing toward Polluted (on average) than Moderate locations,
and Moderate should have more than Good (once Good exists in the data).

Run once, standalone:  python src/validate_factor_counts.py
"""

import numpy as np
import pandas as pd
from db_loader import get_conn

# ---- Load data ---------------------------------------------------------------

shap_vals = np.load("results/shap_values.npy")
feats = pd.read_csv("results/shap_input_features.csv")

conn = get_conn()
preds = pd.read_sql(
    "SELECT fww_id, predicted_status FROM predictions WHERE data_source='historical' ORDER BY id",
    conn
)
conn.close()

assert len(preds) == len(shap_vals) == len(feats), (
    f"Row count mismatch: predictions={len(preds)}, "
    f"shap_vals={len(shap_vals)}, feats={len(feats)}"
)

n_features = shap_vals.shape[1]
print(f"Dataset: {len(preds)} locations, {n_features} features")
print(f"Predicted status counts:\n{preds['predicted_status'].value_counts()}\n")

# ---- Count factors pointing toward Poor, per location -------------------------

n_toward_poor = (shap_vals > 0).sum(axis=1)      # count of positive SHAP values per row
n_toward_mod  = (shap_vals < 0).sum(axis=1)       # count of negative SHAP values per row

preds = preds.copy()
preds["n_toward_poor"] = n_toward_poor
preds["n_toward_moderate"] = n_toward_mod
preds["pct_toward_poor"] = (n_toward_poor / n_features * 100).round(1)

# ---- Validation: does factor count relate to actual status? -------------------

print("=" * 60)
print("VALIDATION: factor count vs predicted status")
print("=" * 60)

summary = preds.groupby("predicted_status").agg(
    mean_factors_toward_poor=("n_toward_poor", "mean"),
    median_factors_toward_poor=("n_toward_poor", "median"),
    mean_pct_toward_poor=("pct_toward_poor", "mean"),
    n_locations=("fww_id", "count"),
)
print(summary.round(2))
print()

# ---- Statistical test: correlation between factor count and Poor probability --

conn = get_conn()
prob_df = pd.read_sql(
    "SELECT fww_id, prob_poor FROM predictions WHERE data_source='historical' ORDER BY id",
    conn
)
conn.close()

merged = preds.merge(prob_df, on="fww_id")
corr = np.corrcoef(merged["n_toward_poor"], merged["prob_poor"])[0, 1]

print("=" * 60)
print("CORRELATION CHECK")
print("=" * 60)
print(f"Correlation between (count of factors pointing toward Poor) and")
print(f"(model's actual predicted probability of Poor): {corr:+.3f}")
print()

if corr > 0.3:
    print("STRONG positive relationship - the factor count is a meaningful,")
    print("valid signal that reflects the model's actual confidence.")
elif corr > 0.1:
    print("WEAK positive relationship - factor count is somewhat meaningful")
    print("but should not be over-interpreted on its own.")
else:
    print("WARNING: little to no relationship. Simply counting how many")
    print("factors point each way may not be a reliable indicator - the")
    print("MAGNITUDE of each factor matters more than the raw count.")

print()
print("=" * 60)
print("EXAMPLE: distribution for a random sample of 5 Poor locations")
print("=" * 60)
poor_sample = merged[merged["predicted_status"] == "Poor"].sample(
    min(5, (merged["predicted_status"] == "Poor").sum()), random_state=42
)
print(poor_sample[["fww_id", "predicted_status", "prob_poor",
                    "n_toward_poor", "n_toward_moderate"]].to_string(index=False))

print()
print("=" * 60)
print("EXAMPLE: distribution for a random sample of 5 Moderate locations")
print("=" * 60)
mod_sample = merged[merged["predicted_status"] == "Moderate"].sample(5, random_state=42)
print(mod_sample[["fww_id", "predicted_status", "prob_poor",
                   "n_toward_poor", "n_toward_moderate"]].to_string(index=False))