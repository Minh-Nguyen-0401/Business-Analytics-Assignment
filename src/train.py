import os, sys, yaml, logging, joblib
import pandas as pd, numpy as np
import matplotlib.pyplot as plt
from typing import Union
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
import lightgbm as lgb
from catboost import CatBoostRegressor
import logging
import json
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings("ignore")

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
root = os.path.dirname(CURRENT_DIR)
sys.path.insert(0, root)
from utils.helper_func import *

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

class ModelTrainer:
    def load_config(self, path=f'{CURRENT_DIR}/search_space.yaml'):
        cfg = yaml.safe_load(open(path)) or {}
        return cfg.get('results', {})

    def __init__(self, model_name: str, config_path: str = f'{CURRENT_DIR}/search_space.yaml'):
        results = self.load_config(config_path)
        params = results.get(model_name, {}).get('params')
        if params is None:
            raise ValueError(f"No tuned params for '{model_name}' in {config_path}")
        self.model_name = model_name
        self.params = params
        self.model = None
        self.y_history = None
        os.makedirs(os.path.join(root, 'models'), exist_ok=True)
        os.makedirs(os.path.join(root, 'models', f'{model_name}_evaluation'), exist_ok=True)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        logging.info(f"Training {self.model_name} with {self.params}")
        if self.model_name == "lightgbm":
            mdl = lgb.LGBMRegressor(**self.params)
        elif self.model_name == "catboost":
            mdl = CatBoostRegressor(**self.params)
        elif self.model_name == "linear_regression":
            mdl = LinearRegression(**self.params)
        else:
            raise ValueError(f"Unknown model '{self.model_name}'")
        self.model = mdl.fit(X, y)
        self.y_history = y.copy()
        joblib.dump(self.model, os.path.join(root, 'models', f'{self.model_name}.pkl'))
        logging.info(f"Saved model to models/{self.model_name}.pkl")

    def load_model(self):
        if self.model is None:
            self.model = joblib.load(os.path.join(root, 'models', f'{self.model_name}.pkl'))
        return self.model

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.array(self.load_model().predict(X))

    def evaluate(self, y_true: Union[pd.Series, np.ndarray], y_pred: Union[pd.Series, np.ndarray], X_test: pd.DataFrame) -> dict:
        yt, yp = np.array(y_true), np.array(y_pred)
        m = {'rmse': mean_squared_error(yt, yp, squared=False),
             'mae': mean_absolute_error(yt, yp),
             'r2': r2_score(yt, yp),
             'mape': mean_absolute_percentage_error(yt, yp)}
        logging.info(f"Metrics: {m}")
        evdir = os.path.join(root, 'models', f'{self.model_name}_evaluation')
        # save metrics to json
        with open(os.path.join(evdir, 'metrics.json'), 'w') as f:
            json.dump(m, f, indent=4)

        plt.scatter(yt, yp, alpha=0.6); plt.savefig(os.path.join(evdir,'scatter.png')); plt.clf()
        plt.plot(self.y_history.index, self.y_history, label='History')
        plt.plot(X_test.index, yt, label='Actual')
        plt.plot(X_test.index, yp, label='Predicted')
        plt.legend(); plt.savefig(os.path.join(evdir,'timeseries.png')); plt.clf()
        plt.hist(yt-yp, bins=30, density=True, alpha=0.7)
        plt.savefig(os.path.join(evdir,'residuals.png')); plt.clf()
        return m
    
    def get_feature_importance(self, X_train, save=True):
        if self.model is None:
            logging.info(f"Loading pre-trained model from models/{self.model_name}.pkl")
            self.model = joblib.load(os.path.join(root, 'models', f'{self.model_name}.pkl'))

        feature_names = X_train.columns.tolist()
        if hasattr(self.model, "feature_importances_"):
            importances = self.model.feature_importances_
        elif hasattr(self.model, "coef_"):
            importances = np.abs(self.model.coef_)
        else:
            raise ValueError(f"Feature importance not available for {self.model_name}")
        df_importance = (
            pd.DataFrame({"feature": feature_names, "importance": importances})
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )
        if save:
            df_importance.to_csv(os.path.join(root, 'models', f'{self.model_name}_feature_importance.csv'), index=False)
        return df_importance
