# freshwater-risk-england and 

Geospatial machine learning for freshwater ecological status in England.
MSc dissertation, COMP7039, Oxford Brookes University.

*Live dashboard: RiverWatch* https://freshwater-risk-england-ml.streamlit.app

---

## What this does

Predicts Water Framework Directive ecological status for 36,280 river monitoring
locations in England, and explains each prediction in terms of the environmental
pressures behind it.

Four public data sources are integrated into a single spatial dataset:

| Source | Provides |
|---|---|
| FreshWater Watch (Earthwatch Europe) | Volunteer nitrate and phosphate readings |
| EA Event Duration Monitoring | Storm overflow discharge records |
| UKCEH Land Cover Map 2024 | 25 m land classification raster |
| EA WFD classifications | Official ecological status — the training label |

The first three produce 17 engineered features. The fourth is the target.

## Results

| | Weighted F1 |
|---|---|
| Random Forest (selected) | 0.700 |
| XGBoost | 0.693 |
| Logistic Regression | 0.636 |
| Best EA threshold rule | 0.630 |

Figures are the mean across five spatial cross-validation folds. Pooling all
36,280 predictions gives 0.658 for the same model.

Ablation: chemistry alone 0.501, plus sewage 0.671, plus land cover 0.699.
Sewage discharge contributes the larger share of the gain.

SHAP attributions are stable across spatial folds (Spearman rho = 0.878).

---

## Setup

Python 3.12.

    python -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt

Two dependencies need care:

**GDAL** requires the system library, not just the Python package:

    sudo apt-get install -y libgdal-dev gdal-bin
    pip install --no-cache-dir "gdal==$(gdal-config --version)"

**POOPy** must be installed from source. The package on PyPI bundles an
installer containing Python 2 syntax, which cannot be parsed under Python 3:

    pip install "git+https://github.com/AlexLipp/POOPy.git@6837ea2c75116703ae9f919917ba6acc07d6569e"

---

## Pipeline

First run, in order:

    python src/db_loader.py             # create tables
    python src/feature_engineering.py   # build feat_matrix
    python src/model_evaluation.py      # compare 3 ML models and 4 rule baselines
    python src/model_training.py        # train the selected model, save rf_model.pkl
    python src/shap_analysis.py         # compute SHAP attributions
    python src/inference.py             # write predictions to the database

Retraining, when new EDM, UKCEH or WFD data arrives:

    python src/feature_engineering.py   # rebuild with new data
    python src/db_loader.py             # reload feat_matrix
    python src/model_training.py        # retrain with retrain=True
    python src/shap_analysis.py         # update explanations

Run the dashboard locally:

    streamlit run src/dashboard.py

---

## Analysis scripts

Each of these produces a figure reported in the dissertation.

| Script | Produces |
|---|---|
| `run_ablation.py` | Ablation results, Section 4.1.2 |
| `rq3_concordance.py` | Citizen-official concordance, Cohen's kappa, Section 4.1.5 |
| `train_with_smote.py` | SMOTE comparison, Section 4.1.1 |
| `validate_all_factors.py` | SHAP direction validation against ground truth |
| `check_factor_directions.py` | Diagnostic behind the monitoring-density bias finding |
| `feature_baselines.py` | Rule-based threshold baselines |

---

## Live data pipeline

A GitHub Actions workflow (`.github/workflows/live_data_pull.yml`) re-pulls
discharge data via POOPy, recomputes the affected sewage features and re-runs
inference for locations whose inputs have changed.

**Known issues:**

- The upstream POOPy feed has returned no data since 9 September 2026. The
  deployed dashboard reflects the last successful run.
- The prediction database is committed on each run and previously exceeded
  GitHub's 100 MB file limit, blocking the pipeline. It is now pruned to the
  most recent prediction per site. The 365-day discharge history in
  `staging_edm_live` must be retained, since the features depend on it.
- The scheduled interval is not reliably honoured on the free tier, as runs
  take longer than the interval.

---

## Database

`data/freshwater_risk.db` (SQLite)

| Table | Contents |
|---|---|
| `feat_matrix` | 36,280 rows: 17 features and the label |
| `predictions` | Latest prediction per site |
| `staging_edm_live` | Live discharge feed, append-only, 365-day window |
| `model_metrics` | Per-fold and per-class performance |
| `ingestion_log` | Audit trail of every load |

---

## Licences

Environment Agency data under the Open Government Licence v3.0, FreshWater Watch
under Creative Commons terms, UKCEH Land Cover Map under its own open terms.

This tool is not an official Environment Agency assessment and gives no advice
on whether water is safe for recreational contact.