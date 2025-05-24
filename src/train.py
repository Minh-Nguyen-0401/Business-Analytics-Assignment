import os
import sys
import json
import joblib
import logging
from typing import Union
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
from sklearn.linear_model import LinearRegression
import lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from tensorflow.keras import layers, callbacks, optimizers

target_dir = os.path.dirname(os.path.abspath(__file__))
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

class ModelTrainer:
    def __init__(self, name: str, params: dict = None, seq_len: int = 12, base_path=None):
        self.name = name.lower()
        self.params = params or {}
        self.seq_len = seq_len
        self.base = base_path or os.path.dirname(os.path.dirname(__file__))
        self.model = None
        # Apply MinMaxScaler to all model types since sales increase heavily over time
        self.y_scaler = MinMaxScaler()
        self.tail_X = None
        self.tail_indices = None
        self.eval_dir = os.path.join(self.base, 'models', f'{self.name}_evaluation')
        os.makedirs(self.eval_dir, exist_ok=True)

    def _to_sequences(self, X: np.ndarray, y: np.ndarray):
        X_seq = X.reshape((X.shape[0], 1, X.shape[1]))
        return X_seq, y, list(range(len(X)))

    def build_rnn(self, input_shape):
        kind = self.name
        units = self.params.get('units', 64 if kind == 'lstm' else 112)
        lr = self.params.get('lr', 0.007805759152905008 if kind == 'lstm' else 0.008753743880976054)
        model = keras.Sequential()
        if kind == 'lstm':
            model.add(layers.LSTM(units, activation='relu', input_shape=input_shape))
        else:
            model.add(layers.SimpleRNN(units, activation='relu', input_shape=input_shape))
        model.add(layers.Dense(1))
        optimizer = optimizers.Adam(
            learning_rate=lr
        )
        model.compile(optimizer=optimizer, loss='mse')
        return model

    def fit(self, X: pd.DataFrame, y: pd.Series):
        y_scaled = self.y_scaler.fit_transform(y.values.reshape(-1, 1)).ravel()
        
        if self.name in ('lstm', 'rnn'):
            X_seq, y_seq, self.seq_indices = self._to_sequences(X.values, y_scaled)
            self.tail_X = X.values[-1:]
            self.tail_indices = [len(X) - 1]
            val_size = 12 
            train_size = len(X_seq) - val_size
            X_train, X_val = X_seq[:train_size], X_seq[train_size:]
            y_train, y_val = y_seq[:train_size], y_seq[train_size:]
            self.model = self.build_rnn(input_shape=(1, X.values.shape[1]))
            patience = self.params.get('patience', 10)
            batch_size = self.params.get('batch_size', 16)
            epochs = self.params.get('epochs', 50)
            early_stopping = callbacks.EarlyStopping(
                monitor='val_loss', 
                patience=patience, 
                restore_best_weights=True,
                mode='min'
            )
            
            history = self.model.fit(
                X_train, y_train,
                epochs=epochs, 
                batch_size=batch_size, 
                validation_data=(X_val, y_val),
                callbacks=[early_stopping],
                verbose=1
            )
            
            # Create evaluation plots
            plt.figure(figsize=(12, 6))
            plt.plot(history.history['loss'], label='Training Loss')
            plt.plot(history.history['val_loss'], label='Validation Loss')
            plt.title(f'{self.name.upper()} Training History')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.legend()
            plt.savefig(os.path.join(self.eval_dir, f'{self.name}_training_history.png'))
            plt.close()
            
            # Save the model and related objects
            os.makedirs(os.path.join(self.base, 'models'), exist_ok=True)
            save_path = os.path.join(self.base, 'models', f'{self.name}_model.pkl')
            joblib.dump((self.model, self.y_scaler, self.tail_X, self.tail_indices), save_path)
            logger.info(f"Saved {self.name} model to {save_path}")
            
            # Save training history
            history_data = {}
            for k, v in history.history.items():
                try:
                    if hasattr(v, 'tolist'):
                        history_data[k] = v.tolist()
                    elif isinstance(v, (list, tuple)):
                        history_data[k] = [x.tolist() if hasattr(x, 'tolist') else x for x in v]
                    else:
                        history_data[k] = v
                except (TypeError, AttributeError):
                    history_data[k] = str(v)
            
            # Save the processed history
            with open(os.path.join(self.eval_dir, 'training_history.json'), 'w') as f:
                json.dump(history_data, f)
        else:
            if self.name == 'lightgbm':
                self.model = lgb.LGBMRegressor(**self.params)
            elif self.name == 'catboost':
                self.model = CatBoostRegressor(**self.params)
            elif self.name == 'linear_regression':
                self.model = LinearRegression(**self.params)
            else:
                raise ValueError(f"Unsupported model {self.name}")
            self.model.fit(X, y_scaled)
            os.makedirs(os.path.join(self.base, 'models'), exist_ok=True)
            save_path = os.path.join(self.base, 'models', f'{self.name}_model.pkl')
            joblib.dump((self.model, self.y_scaler), save_path)
            logger.info(f"Saved {self.name} model to {save_path}")
            return self.model

    def _load(self):
        if self.model is None:
            path = os.path.join(self.base, 'models', f'{self.name}_model.pkl')
            if not os.path.exists(path):
                raise FileNotFoundError(f"Model file {path} not found. Please train the model first.")
            
            if self.name in ('lstm', 'rnn'):
                loaded = joblib.load(path)
                if len(loaded) >= 4:
                    self.model, self.y_scaler, *_, self.tail_X, self.tail_indices = loaded
                else:
                    self.model, self.y_scaler, self.tail_X, self.tail_indices = loaded
            else:
                loaded = joblib.load(path)
                if isinstance(loaded, tuple) and len(loaded) >= 2:
                    self.model, self.y_scaler = loaded[:2]
                else:
                    self.model = loaded
                    if not hasattr(self, 'y_scaler') or self.y_scaler is None:
                        self.y_scaler = MinMaxScaler()
        return self.model

    def predict(self, X_future: pd.DataFrame):
        if self.name in ('lstm', 'rnn'):
            self._load()
            
            X_future_seq = X_future.values.reshape((X_future.values.shape[0], 1, X_future.values.shape[1]))
            scaled_preds = self.model.predict(X_future_seq, verbose=0)
            
            return self.y_scaler.inverse_transform(scaled_preds).ravel()
        
        preds = self._load().predict(X_future)
        return self.y_scaler.inverse_transform(preds.reshape(-1, 1)).ravel()

    def evaluate(self, y_true: Union[pd.Series, np.ndarray], y_pred: Union[pd.Series, np.ndarray]):
        yt, yp = np.array(y_true), np.array(y_pred)
        metrics = {
            'rmse': mean_squared_error(yt, yp, squared=False),
            'mae': mean_absolute_error(yt, yp),
            'r2': r2_score(yt, yp),
            'mape': mean_absolute_percentage_error(yt, yp)
        }
        with open(os.path.join(self.eval_dir, 'metrics.json'), 'w') as f:
            json.dump(metrics, f)
        plt.figure(); plt.scatter(yt, yp, alpha=0.6); plt.savefig(os.path.join(self.eval_dir, 'scatter.png')); plt.clf()
        plt.figure(); plt.plot(y_true.index, yt, label='Actual'); plt.plot(y_true.index, yp, '--', label='Predicted'); plt.savefig(os.path.join(self.eval_dir, 'timeseries.png')); plt.clf()
        plt.figure(); plt.hist(yt-yp, bins=30, density=True, alpha=0.7); plt.savefig(os.path.join(self.eval_dir, 'residuals.png')); plt.clf()
        return metrics

    def get_feature_importance(self, X_train: pd.DataFrame, save: bool = True):
        model = self._load()
        if hasattr(model, 'feature_importances_'):
            imp = model.feature_importances_
        elif hasattr(model, 'coef_'):
            imp = np.abs(model.coef_)
        else:
            raise ValueError('No feature importance available')
        df = pd.DataFrame({'feature': X_train.columns, 'importance': imp}).sort_values('importance', ascending=False)
        if save:
            df.to_csv(os.path.join(self.eval_dir, f'{self.name}_feature_importance.csv'), index=False)
        return df

    def generate_shap_plot(self, X_train: pd.DataFrame, y_train: pd.Series, save: bool = True):
        try:
            import shap
            model = self._load()
            if self.name not in ('lstm', 'rnn'):
                explainer = shap.TreeExplainer(model)
                shap_vals = explainer.shap_values(X_train)
                plt.figure(figsize=(12,8)); shap.summary_plot(shap_vals, X_train, show=False, max_display=30);
                if save: plt.savefig(os.path.join(self.eval_dir, f'{self.name}_shap.png'), bbox_inches='tight'); plt.clf()
        except Exception as e:
            logger.warning(f"SHAP plot failed: {e}")