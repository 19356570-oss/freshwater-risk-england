"""
check_overflow_outlier.py
One-off check: is 44 overflow pipes at one location genuinely unusual,
or a normal value within the real distribution?

Run:  python src/check_overflow_outlier.py
"""

import pandas as pd
from config import FEAT_MATRIX

df = pd.read_csv(FEAT_MATRIX)

print("n_overflows distribution:")
print(df["n_overflows"].describe())
print()

print("Percentile of value=44:")
pct = (df["n_overflows"] <= 44).mean() * 100
print(f"  {pct:.1f}% of locations have 44 or fewer overflow pipes nearby")
print(f"  So this location is in the top {100-pct:.1f}% by overflow count")
print()

print("Top 10 highest n_overflows values in the dataset:")
print(df.nlargest(10, "n_overflows")[["site_name", "n_overflows", "spills_per_pipe", "wfd_status"]])
print()

print("How many locations have n_overflows >= 44:")
print((df["n_overflows"] >= 44).sum(), "out of", len(df))