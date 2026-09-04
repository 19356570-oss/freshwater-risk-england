"""
demo_live_prediction.py
Live demonstration script - proves the trained model genuinely predicts,
not just displays cached dashboard results.

Two modes:
    1. Look up a real, existing location by name and show its live prediction
    2. Enter completely custom feature values and get a live prediction
       for a hypothetical scenario

Both call model.predict() live, in front of whoever is watching - nothing
here is pre-computed or staged.

Run:  python src/demo_live_prediction.py
"""

import pandas as pd
import joblib
from config import FEAT_MATRIX, ALL_FEATS, MODELS_DIR, ENCODER_PATH

model = joblib.load(f"{MODELS_DIR}/rf_model.pkl")
le = joblib.load(ENCODER_PATH)

print("=" * 60)
print("LIVE PREDICTION DEMONSTRATION")
print("=" * 60)
print(f"Loaded trained model: {MODELS_DIR}/rf_model.pkl")
print()
print("Choose a mode:")
print("  1. Look up a real location by name")
print("  2. Enter custom feature values (hypothetical scenario)")
mode = input("Enter 1 or 2: ").strip()

if mode == "1":
    df = pd.read_csv(FEAT_MATRIX)
    search = input("Type part of a site name to search: ").strip()
    matches = df[df["site_name"].str.contains(search, case=False, na=False)]

    if matches.empty:
        print("No matches found.")
    else:
        print(f"\nFound {len(matches)} match(es):")
        for i, (_, row) in enumerate(matches.head(10).iterrows()):
            wb = row.get("wb_id", "unknown waterbody")
            print(f"  {i}: {row['site_name']} ({row['county']}) - official waterbody: {wb}")

        idx = int(input("\nEnter the number to select: ").strip())
        chosen = matches.iloc[idx]

        print(f"\nSelected: {chosen['site_name']}, {chosen['county']}")
        print(f"Official WFD status (ground truth): {chosen['wfd_status']}")

        X = chosen[ALL_FEATS].fillna(0).values.reshape(1, -1)
        pred = model.predict(X)
        proba = model.predict_proba(X)

        print()
        print(">> Running model.predict() live now...")
        print(f"   Predicted status: {le.inverse_transform(pred)[0]}")
        classes = list(le.classes_)
        for cls, p in zip(classes, proba[0]):
            print(f"   P({cls}) = {p:.3f}")

elif mode == "2":
    print("\nEnter values for a hypothetical location.")
    print("(Press Enter to use a typical/median value)\n")

    df = pd.read_csv(FEAT_MATRIX)
    values = {}
    for feat in ALL_FEATS:
        default = df[feat].median()
        raw = input(f"  {feat} [default {default:.2f}]: ").strip()
        values[feat] = float(raw) if raw else default

    X = pd.DataFrame([values])[ALL_FEATS].values
    pred = model.predict(X)
    proba = model.predict_proba(X)

    print()
    print(">> Running model.predict() live now, on values you just entered...")
    print(f"   Predicted status: {le.inverse_transform(pred)[0]}")
    classes = list(le.classes_)
    for cls, p in zip(classes, proba[0]):
        print(f"   P({cls}) = {p:.3f}")

else:
    print("Invalid mode.")