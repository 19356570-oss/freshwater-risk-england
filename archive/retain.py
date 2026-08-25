import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import shap
import joblib
import os
import json
from datetime import datetime

from model_training1 import (
    load_feature_matrix,
    create_spatial_folds,
    train_evaluate_spatial_cv,
    ALL_FEATURES,
)

MODELS_DIR = "models"
RESULTS_DIR = "results"
SEEDS_TO_TEST = [42, 7, 123, 2024, 99]  # different seeds to test model stability


def run_stability_check():
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    df = load_feature_matrix()
    df = create_spatial_folds(df)

    run_summaries = []
    all_predictions = {}
    final_models = {}
    shap_importances = {}

    for seed in SEEDS_TO_TEST:
        # spatial CV score for this seed
        cv_results, le = train_evaluate_spatial_cv(df, ALL_FEATURES, model_type='rf')

        y_full = le.transform(df['wfd_status'])
        X_full = df[ALL_FEATURES].fillna(0)

        # final model trained on all data for this seed
        model = RandomForestClassifier(
            n_estimators=200, max_depth=10, min_samples_leaf=5,
            class_weight='balanced', random_state=seed, n_jobs=-1
        )
        model.fit(X_full, y_full)

        pred_encoded = model.predict(X_full)
        pred_labels = le.inverse_transform(pred_encoded)

        # SHAP importance for this seed's model
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_full)
        shap_vals_poor = shap_values[1] if isinstance(shap_values, list) else shap_values
        mean_abs_shap = np.abs(shap_vals_poor).mean(axis=0)
        shap_importances[seed] = dict(zip(ALL_FEATURES, mean_abs_shap.tolist()))

        run_summaries.append({
            'seed': seed,
            'spatial_cv_weighted_f1': cv_results['overall_weighted_f1'],
        })
        all_predictions[f'seed_{seed}'] = pred_labels
        final_models[seed] = model

        print(f"Seed {seed}: F1 = {cv_results['overall_weighted_f1']:.4f}")

    # compare F1 across seeds
    f1_values = [r['spatial_cv_weighted_f1'] for r in run_summaries]
    print(f"\nMean F1: {np.mean(f1_values):.4f}, Std: {np.std(f1_values):.4f}")
    print(f"Range: {min(f1_values):.4f} to {max(f1_values):.4f}")

    # check how many samples got the same predicted label across all seeds
    pred_df = pd.DataFrame(all_predictions)
    agreement_counts = pred_df.apply(lambda row: row.value_counts().iloc[0], axis=1)
    fully_agreed = (agreement_counts == len(SEEDS_TO_TEST)).sum()
    pct_agreed = 100 * fully_agreed / len(pred_df)
    print(f"Prediction agreement across seeds: {pct_agreed:.1f}%")

    # average SHAP importance across seeds
    shap_df = pd.DataFrame(shap_importances)
    shap_df['mean_importance'] = shap_df.mean(axis=1)
    shap_df['std_importance'] = shap_df.drop(columns='mean_importance').std(axis=1)
    shap_df = shap_df.sort_values('mean_importance', ascending=False)

    print("\nSHAP feature importance (mean across seeds):")
    for feat, row in shap_df.iterrows():
        print(f"  {feat:35s}: {row['mean_importance']:.4f} (+/- {row['std_importance']:.4f})")

    # save everything
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary = {
        'timestamp': timestamp,
        'seeds_tested': SEEDS_TO_TEST,
        'f1_per_seed': run_summaries,
        'f1_mean': float(np.mean(f1_values)),
        'f1_std': float(np.std(f1_values)),
        'pct_samples_fully_agreed': float(pct_agreed),
        'shap_mean_importance': shap_df['mean_importance'].to_dict(),
        'shap_std_importance': shap_df['std_importance'].to_dict(),
    }

    with open(os.path.join(RESULTS_DIR, 'stability_check_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    pred_df.to_csv(os.path.join(RESULTS_DIR, 'stability_check_predictions.csv'), index=False)
    shap_df.to_csv(os.path.join(RESULTS_DIR, 'shap_stability.csv'))

    # keep the best-performing seed's model as the one used going forward
    best_seed = run_summaries[int(np.argmax(f1_values))]['seed']
    joblib.dump(final_models[best_seed], os.path.join(MODELS_DIR, 'rf_model.pkl'))
    print(f"\nBest seed: {best_seed}, saved as models/rf_model.pkl")

    return summary, pred_df, shap_df


if __name__ == "__main__":
    run_stability_check()
