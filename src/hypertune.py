import os
import yaml
import numpy as np
import pandas as pd
import lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from scikeras.wrappers import KerasRegressor
import optuna
from optuna.integration import OptunaSearchCV
from optuna.distributions import IntDistribution, FloatDistribution
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, SimpleRNN, Dense
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_squared_error
from tensorflow.keras import callbacks
from sklearn.preprocessing import MinMaxScaler
from optuna.pruners import MedianPruner

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_SPACE = os.path.join(CURRENT_DIR, 'search_space.yaml')
DEFAULT_RESULTS = os.path.join(CURRENT_DIR, 'hypertune_results.yaml')

class HyperTuner:
    def __init__(self, model_name, n_features, ts_splits,
                 search_space_path=DEFAULT_SPACE,
                 results_path=DEFAULT_RESULTS):
        self.name = model_name
        self.nf = n_features
        self.splits = ts_splits
        self.space_path = search_space_path
        self.results_path = results_path
        self.X = None
        self.y = None

    def run_permutation_importance(self, X, y):
        self.X, self.y = X, y
        base_map = {
            'lightgbm': lgb.LGBMRegressor(random_state=42, verbose=-1),
            'catboost': CatBoostRegressor(verbose=0, random_state=42),
            'linear_regression': LinearRegression(),
            'rnn': lgb.LGBMRegressor(random_state=42, verbose=-1),
            'lstm': lgb.LGBMRegressor(random_state=42, verbose=-1)
        }
        if self.name not in base_map:
            raise ValueError(f"PI not supported for model {self.name}")
        model_pi = base_map[self.name]
        imp = np.zeros(X.shape[1])
        tss = TimeSeriesSplit(n_splits=self.splits)
        for tr, te in tss.split(X):
            model_pi.fit(X.iloc[tr], y.iloc[tr])
            imp += permutation_importance(
                model_pi, X.iloc[te], y.iloc[te], n_repeats=5, random_state=0
            ).importances_mean
        imp /= self.splits
        feats = X.columns[np.argsort(imp)[::-1][:self.nf]].tolist()
        promo = [c for c in X.columns if 'promo_type' in c.lower() or 'months_since_last_promo' in c.lower()]
        feats = list(dict.fromkeys(feats + promo))
        # Write PI results
        results = yaml.safe_load(open(self.results_path)) if os.path.exists(self.results_path) else {}
        results.setdefault('results', {}).setdefault(self.name, {})['features'] = feats
        with open(self.results_path, 'w') as f:
            yaml.safe_dump(results, f, sort_keys=False)
        print(f"Selected features for {self.name}: {feats}")
        return feats

    def _build_distributions(self, space):
        dist = {}
        for p, cfg in space.items():
            low, high, log = cfg['low'], cfg['high'], cfg.get('log', False)
            if cfg['type'] == 'int':
                dist[p] = IntDistribution(low=low, high=high, step=1, log=log)
            else:
                dist[p] = FloatDistribution(low=low, high=high, log=log)
        return dist

    def tune(self, n_trials=50):
        if self.X is None:
            raise RuntimeError('Call run_permutation_importance first')
        # Load search space
        space = yaml.safe_load(open(self.space_path))
        if self.name not in space:
            raise ValueError(f"No search space for {self.name}")
        params_space = space[self.name]
        dists = self._build_distributions(params_space)
        # Load features
        if os.path.exists(self.results_path):
            try:
                with open(self.results_path, 'r') as f:
                    results = yaml.safe_load(f) or {}
                if results.get('results', {}).get(self.name, {}).get('features'):
                    feats = results['results'][self.name]['features']
                    Xsub = self.X[feats]
                else:
                    Xsub = self.X.copy()
            except Exception as e:
                print(f"Warning: Error loading results file: {e}")
                Xsub = self.X.copy()
        else:
            Xsub = self.X.copy()

        if self.name in ['lightgbm', 'catboost', 'linear_regression']:
            model_map = {
                'lightgbm': lgb.LGBMRegressor(random_state=42),
                'catboost': CatBoostRegressor(verbose=0, random_state=42),
                'linear_regression': LinearRegression()
            }
            model = model_map[self.name]
            opt = OptunaSearchCV(
                model,
                param_distributions=dists,
                cv=TimeSeriesSplit(n_splits=self.splits),
                scoring='neg_mean_squared_error',
                n_trials=n_trials,
                random_state=0
            )
            # Apply MinMaxScaler to all models
            y_scaler = MinMaxScaler()
            y_scaled = y_scaler.fit_transform(self.y.values.reshape(-1, 1)).ravel()
            
            opt.fit(Xsub, y_scaled)
            best = opt.best_params_
        else:
            def objective(trial):
                if self.name == 'lstm':
                    lr = trial.suggest_float('lr', 0.006, 0.009)
                    units = trial.suggest_int('units', 48, 80, step=16)  
                else:
                    lr = trial.suggest_float('lr', 0.007, 0.01)
                    units = trial.suggest_int('units', 96, 128, step=16)
                
                batch_size = trial.suggest_categorical('batch_size', [16, 32]) 
                patience = trial.suggest_int('patience', 8, 12, step=2) 
                
                val_size = 12
                
                y_scaler = MinMaxScaler()
                
                X_values = Xsub.values
                y_array = self.y.values
                
                y_scaled = y_scaler.fit_transform(y_array.reshape(-1, 1)).ravel()
                
                X_seq = X_values.reshape((X_values.shape[0], 1, X_values.shape[1]))
                
                train_size = len(X_seq) - val_size
                X_train, X_val = X_seq[:train_size], X_seq[train_size:]
                y_train, y_val = y_scaled[:train_size], y_scaled[train_size:]
                
                model = Sequential()
                
                if self.name == 'lstm':
                    model.add(LSTM(units, activation='relu', input_shape=(1, X_values.shape[1])))
                else:  # SimpleRNN
                    model.add(SimpleRNN(units, activation='relu', input_shape=(1, X_values.shape[1])))
                
                model.add(Dense(1))
                
                optimizer = Adam(learning_rate=lr)
                model.compile(optimizer=optimizer, loss='mse')
                
                # Early stopping callback
                early_stopping = callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=patience,
                    restore_best_weights=True,
                    mode='min'
                )
                
                # Optuna pruning callback
                pruning_callback = optuna.integration.TFKerasPruningCallback(trial, 'val_loss')
                
                # Train the model
                model.fit(
                    X_train, y_train,
                    epochs=50,  # Fixed at 50 like in EDA notebook
                    batch_size=batch_size,
                    validation_data=(X_val, y_val),
                    callbacks=[early_stopping, pruning_callback],
                    verbose=0
                )
                
                # Predict on validation set and calculate MSE
                pred = model.predict(X_val, verbose=0)
                return mean_squared_error(y_val, pred)
            study = optuna.create_study(direction='minimize')
            study.optimize(objective, n_trials=n_trials)
            best = study.best_trial.params
        # save best params
        results.setdefault('results', {}).setdefault(self.name, {})['params'] = best
        with open(self.results_path, 'w') as f:
            yaml.safe_dump(results, f, sort_keys=False)
        print(f"Best params for {self.name}: {best}")
        return best