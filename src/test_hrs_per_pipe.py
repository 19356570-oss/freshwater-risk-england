"""
test_hrs_per_pipe.py
One-off check: does hrs_per_pipe meaningfully help model performance,
or is it just noise given its near-zero SHAP correlation?

Compares weighted F1 with vs without the feature using the same
spatial CV setup as the main pipeline.

Run once, delete after.
"""

import pandas as pd
from config import FEAT_MATRIX, ALL_FEATS
from model_training import train_model

df = pd.read_csv(FEAT_MATRIX)

feats_with = ALL_FEATS
feats_without = [f for f in ALL_FEATS if f != "hrs_per_pipe"]

print("=" * 60)
print(f"WITH hrs_per_pipe ({len(feats_with)} features)")
print("=" * 60)
_, _, res_with = train_model(model_type="rf", feat_cols=feats_with, df=df.copy())

print("\n" + "=" * 60)
print(f"WITHOUT hrs_per_pipe ({len(feats_without)} features)")
print("=" * 60)
_, _, res_without = train_model(model_type="rf", feat_cols=feats_without, df=df.copy())

print("\n" + "=" * 60)
print("COMPARISON")
print("=" * 60)
f1_with = res_with["overall_weighted_f1"]
f1_without = res_without["overall_weighted_f1"]
print(f"  With hrs_per_pipe:    F1 = {f1_with:.4f}")
print(f"  Without hrs_per_pipe: F1 = {f1_without:.4f}")
print(f"  Difference: {f1_with - f1_without:+.4f}")

if f1_without >= f1_with - 0.001:
    print("\n  hrs_per_pipe adds no meaningful value - safe to remove.")
else:
    print("\n  hrs_per_pipe contributes to accuracy - investigate the sign issue instead of removing.")