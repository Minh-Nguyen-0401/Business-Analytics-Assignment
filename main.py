import logging
import warnings
import re
import argparse
import yaml

import pandas as pd

from src.ingest import *
from src.aggregate import *
from src.preprocess import *
from src.hypertune import *
from src.train import *
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
warnings.filterwarnings("ignore")

def main(model_name=None, n_trials=60, ext_pct = 0.1):
    data_dict = Ingestion().load_data()
    agg = Aggregation(data_dict, required_shiftback=12)
    merged = agg.merge_data()

    # separate original vs. supp data for different preprocessing schema
    org_feats = agg.org_cols
    org_df = agg.engineer_feats(merged[org_feats])
    org_df_cols = org_df.columns.tolist()
    ext_df = (
        agg.engineer_feats(merged)
        .drop(org_df_cols, axis=1)
        .join(org_df["New_Sales"], how="left")
    )

    # train/test split
    X_org_tr, y_tr, X_org_te, y_te = agg.split_train_test(org_df, target_feat="New_Sales")
    X_ext_tr, _,  X_ext_te,  _   = agg.split_train_test(ext_df, target_feat="New_Sales")

    # preprocess
    org_prep = CustomPreprocessor()
    org_prep.fit(X_org_tr, y_tr, mi_pct=0.0)
    X_org_tr = org_prep.transform(X_org_tr)
    X_org_te = org_prep.transform(X_org_te)

    ext_prep = CustomPreprocessor()
    ext_prep.fit(X_ext_tr, y_tr, mi_pct=(1 - ext_pct))
    X_ext_tr = ext_prep.transform(X_ext_tr)
    X_ext_te = ext_prep.transform(X_ext_te)

    X_tr = pd.concat([X_org_tr, X_ext_tr], axis=1)
    X_te = pd.concat([X_org_te, X_ext_te], axis=1)

    # Define model configurations
    model_configs = {
        "linear_regression": {"features": 50, "trials": n_trials},
        "lightgbm": {"features": 80, "trials": n_trials},
        "catboost": {"features": 80, "trials": n_trials},
        "lstm": {"features": 60, "trials": n_trials},
        "rnn": {"features": 60, "trials": n_trials}
    }
    
    if model_name and model_name in model_configs:
        if model_name in ["linear_regression", "lightgbm", "catboost"]:
            run_traditional_model(model_name, model_configs[model_name]["features"], X_tr, y_tr, X_te, y_te, model_configs[model_name]["trials"])
        elif model_name in ["lstm", "rnn"]:
            run_neural_model(model_name, model_configs[model_name]["features"], X_tr, y_tr, X_te, y_te, model_configs[model_name]["trials"])
        return
    
    for traditional_model, config in {k: v for k, v in model_configs.items() if k in ["linear_regression", "lightgbm"]}.items():
        run_traditional_model(traditional_model, config["features"], X_tr, y_tr, X_te, y_te, config["trials"])

    if not model_name or model_name == "lstm":
        run_neural_model("lstm", model_configs["lstm"]["features"], X_tr, y_tr, X_te, y_te, model_configs["lstm"]["trials"])

def run_traditional_model(model_name, n_features, X_tr, y_tr, X_te, y_te, n_trials):
    """Run a traditional model (linear_regression, lightgbm, catboost)"""
    logging.info(f"Running {model_name} model with {n_features} features and {n_trials} trials")
    tuner = HyperTuner(model_name, n_features=n_features, ts_splits=5)
    feats = tuner.run_permutation_importance(X_tr, y_tr)
    # mandatory = [
    #     c for c in X_tr.columns
    #     if re.match(r"(?i)(promo_type|month|quarter)", c)
    # ]
    # feats = list(dict.fromkeys(feats + mandatory))
    tuner.tune(n_trials=n_trials)

    print(f"NUMBER OF FINAL FEATURES FOR {model_name.upper()}: {len(feats)}")
    trainer = ModelTrainer(model_name)
    trainer.fit(X_tr[feats], y_tr)
    preds = trainer.predict(X_te[feats])
    trainer.evaluate(y_te, preds)
    trainer.get_feature_importance(X_train=X_tr[feats], save=True)
    trainer.generate_shap_plot(X_train=X_tr[feats], save=True)

def run_neural_model(model_name, n_features, X_tr, y_tr, X_te, y_te, n_trials):
    """Run a neural network model (lstm, rnn)"""
    logging.info(f"Running {model_name} model with {n_features} features and {n_trials} trials")
    
    # Since Neural networks are more sensitive to heterogeneous features with multicollinearity
    # We need to use only the original necessary features
    data_dict = Ingestion().load_data()
    agg = Aggregation(data_dict, required_shiftback=12)
    merged = agg.merge_data()
    
    org_df = merged[agg.org_cols]
    lag_ws = [12, 15]
    org_df = generate_lagged_feats(org_df, ["New_Sales"], lag_ws)
    org_df["Month"] = org_df.index.month.astype(str)
    org_df["Year"] = org_df.index.year
    org_df["Quarter"] = org_df.index.quarter.astype(str)
    nn_X_tr, nn_y_tr, nn_X_te, nn_y_te = agg.split_train_test(org_df, target_feat="New_Sales")

    # Preprocess
    nn_preprocessor = CustomPreprocessor()
    nn_preprocessor.fit(nn_X_tr, nn_y_tr, mi_pct = 0.0)
    nn_X_tr = nn_preprocessor.transform(nn_X_tr)
    nn_X_te = nn_preprocessor.transform(nn_X_te)

    # Hyperparameter tuning
    tuner = HyperTuner(model_name, n_features=n_features, ts_splits=5)
    feats = tuner.run_permutation_importance(nn_X_tr, nn_y_tr)
    mandatory = [
        c for c in nn_X_tr.columns
        if re.match(r"(?i)(month|quarter|year)", c)
    ]
    feats = list(dict.fromkeys(feats + mandatory))
    tuner.tune(n_trials=n_trials)
    
    with open("./src/hypertune_results.yaml", "r") as f:
        results = yaml.safe_load(f)
    if model_name in results.get("results", {}) and "params" in results["results"][model_name]:
        model_params = results["results"][model_name]["params"]
        print(f"NUMBER OF FINAL FEATURES FOR {model_name.upper()}: {len(feats)}")

        trainer = ModelTrainer(model_name, params=model_params)
        trainer.fit(nn_X_tr[feats], nn_y_tr)
        preds = trainer.predict(nn_X_te[feats])
        trainer.evaluate(nn_y_te, preds)
        trainer.generate_shap_plot(X_train=nn_X_tr[feats], save=True)
    else:
        logging.error(f"No hyperparameters found for {model_name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Sales Data Analysis')
    parser.add_argument('--model', type=str, help='Model to run (linear_regression, lightgbm, catboost, lstm, rnn)', 
                        choices=['linear_regression', 'lightgbm', 'catboost', 'lstm', 'rnn'])
    parser.add_argument('--trials', type=int, help='Number of trials for hyperparameter tuning', default=60)
    parser.add_argument('--ext_pct', type=float, help='% of external data to be used', default=0.1)
    args = parser.parse_args()
    
    main(model_name=args.model, n_trials=args.trials)