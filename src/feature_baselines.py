"""
feature_baselines.py
Prints the low/typical/high threshold for every feature, based on real
data across all 36,280 locations. Use this as a permanent reference for
any location's explanation - not just one example.

Run:  python src/feature_baselines.py
"""

import pandas as pd
from db_loader import get_conn
from config import ALL_FEATS

conn = get_conn()
fm = pd.read_sql(f"SELECT {', '.join(ALL_FEATS)} FROM feat_matrix", conn)
conn.close()

print(f"{'Feature':20s} {'Low if below':>14s} {'Typical range':>20s} {'High if above':>14s}")
print("-" * 72)

for col in ALL_FEATS:
    p25, p75 = fm[col].quantile(0.25), fm[col].quantile(0.75)
    print(f"{col:20s} {p25:>14.2f} {f'{p25:.2f} - {p75:.2f}':>20s} {p75:>14.2f}")