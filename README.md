# freshwater-risk-england

# Normal one-time setup 
1. db_loader.py           -- create tables
2. feature_engineering.py -- build feat_matrix
3. model_evaluation.py    -- compare 3 ML models + 4 traditional rules, pick winner (RF)
4. model_training.py      -- train RF one more time, save rf_model.pkl + label_encoder.pkl
5. shap_analysis.py       -- explain rf_model.pkl, save shap results
6. inference.py           -- run predictions using rf_model.pkl, save to predictions table

# Retraining (run when new EDM/UKCEH/WFD data arrives)
python src/feature_engineering.py   # rebuild feature matrix with new data
python src/db_loader.py             # reload feat_matrix table
python src/model_training.py        # retrain RF with retrain=True
python src/shap_analysis.py         # update SHAP explanations
