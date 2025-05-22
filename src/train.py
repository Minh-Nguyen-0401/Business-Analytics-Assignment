import os, sys, yaml, json, joblib, warnings, logging
from typing import Union
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
from sklearn.linear_model import LinearRegression
import lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras import layers, callbacks

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
PAR_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(PAR_DIR)
sys.path.insert(0, ROOT)

def build_rnn(kind: str, n_f: int, seq_len: int) -> keras.Model:
    rnn = {"lstm": layers.LSTM, "rnn": layers.SimpleRNN}[kind]
    inp = keras.Input((seq_len, n_f))
    x = layers.Conv1D(64, 3, padding="causal", activation="relu")(inp)
    x = layers.Bidirectional(rnn(128, return_sequences=False))(x)
    x = layers.Dropout(0.1)(x)
    x = layers.Dense(64, activation="relu")(x)
    out = layers.Dense(1)(x)
    model = keras.Model(inp, out)
    model.compile(keras.optimizers.Adam(1e-4), loss="mse", metrics=[keras.metrics.RootMeanSquaredError()])
    return model

class ModelTrainer:
    def __init__(self, name: str, cfg=f"{PAR_DIR}/search_space.yaml"):
        self.name = name.lower()
        self.params = (yaml.safe_load(open(cfg)) or {}).get("results", {}).get(self.name, {}).get("params", {})
        self.seq_len = 36
        self.tail_X = None
        self.full_y = None
        self.model = None
        self.y_scaler = StandardScaler()
        os.makedirs(f"{ROOT}/models/{self.name}_evaluation", exist_ok=True)

    def _to_sequences(self, X, y):
        xs, ys = [], []
        xv, yv = X.values, y.values
        for i in range(len(xv) - self.seq_len + 1):
            xs.append(xv[i:i + self.seq_len]); ys.append(yv[i + self.seq_len - 1])
        return np.asarray(xs), np.asarray(ys)

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.full_y = y.copy()
        if self.name in {"lstm", "rnn"}:
            Xs = X.copy()
            ys_scaled = pd.Series(self.y_scaler.fit_transform(y.to_frame()).ravel(), index=y.index)
            self.tail_X = Xs.iloc[-(self.seq_len - 1):].copy()
            X_seq, y_seq = self._to_sequences(Xs, ys_scaled)
            val_size = 12
            X_tr, y_tr = X_seq[:-val_size], y_seq[:-val_size]
            X_val, y_val = X_seq[-val_size:], y_seq[-val_size:]
            self.model = build_rnn(self.name, X_seq.shape[2], self.seq_len)
            hist = self.model.fit(X_tr, y_tr,
                                  validation_data=(X_val, y_val),
                                  epochs=100,
                                  batch_size=32,
                                  shuffle=False,
                                  verbose=1,
                                  callbacks=[callbacks.EarlyStopping(patience=20, restore_best_weights=True)])
            joblib.dump((self.model, self.y_scaler, self.tail_X), f"{ROOT}/models/{self.name}.pkl")
            hist_dict = {k: [float(x) for x in v] for k, v in hist.history.items()}
            with open(f"{ROOT}/models/{self.name}_evaluation/training_history.json", "w") as f:
                json.dump(hist_dict, f, indent=2)
        else:
            if self.name == "lightgbm":
                self.model = lgb.LGBMRegressor(**self.params)
            elif self.name == "catboost":
                self.model = CatBoostRegressor(**self.params)
            elif self.name == "linear_regression":
                self.model = LinearRegression(**self.params)
            else:
                raise ValueError
            self.model.fit(X, y)
            joblib.dump(self.model, f"{ROOT}/models/{self.name}.pkl")

    def _load_model(self):
        if self.model is None:
            if self.name in {"lstm", "rnn"}:
                self.model, self.y_scaler, self.tail_X = joblib.load(f"{ROOT}/models/{self.name}.pkl")
            else:
                self.model = joblib.load(f"{ROOT}/models/{self.name}.pkl")
        return self.model

    def predict(self, X_future: pd.DataFrame):
        if self.name in {"lstm", "rnn"}:
            self._load_model()
            buf = pd.concat([self.tail_X, X_future.copy()])
            preds = []
            for i in range(len(X_future)):
                win = buf.iloc[i:i + self.seq_len].values.reshape(1, self.seq_len, -1)
                p = self.model.predict(win, verbose=0)[0, 0]
                preds.append(p)
                buf.iat[i + self.seq_len - 1, 0] = p
            return self.y_scaler.inverse_transform(np.array(preds).reshape(-1, 1)).ravel()
        return np.asarray(self._load_model().predict(X_future))

    def evaluate(self, y_true: Union[pd.Series, np.ndarray], y_pred: Union[pd.Series, np.ndarray]):
        yt, yp = np.asarray(y_true), np.asarray(y_pred)
        evdir = f"{ROOT}/models/{self.name}_evaluation"
        json.dump({"rmse": mean_squared_error(yt, yp, squared=False),
                   "mae": mean_absolute_error(yt, yp),
                   "r2": r2_score(yt, yp),
                   "mape": mean_absolute_percentage_error(yt, yp)},
                  open(f"{evdir}/metrics.json", "w"), indent=2)
        plt.scatter(yt, yp, alpha=0.6); plt.savefig(f"{evdir}/scatter.png"); plt.clf()
        plt.plot(self.full_y.index, self.full_y.values, label="History")
        plt.plot(y_true.index, yt, label="Actual")
        plt.plot(y_true.index, yp, linestyle="--", lw=2, label="Predicted")
        plt.legend(); plt.gcf().autofmt_xdate(); plt.savefig(f"{evdir}/timeseries.png"); plt.clf()
        plt.hist(yt - yp, bins=30, density=True, alpha=0.7); plt.savefig(f"{evdir}/residuals.png"); plt.clf()
        return json.load(open(f"{evdir}/metrics.json"))

    def get_feature_importance(self, X_train: pd.DataFrame, save=True):
        self._load_model()
        if hasattr(self.model, "feature_importances_"):
            imp = self.model.feature_importances_
        elif hasattr(self.model, "coef_"):
            imp = np.abs(self.model.coef_)
        else:
            raise ValueError("Feature importance not available")
        df = pd.DataFrame({"feature": X_train.columns, "importance": imp}).sort_values("importance", ascending=False)
        if save:
            df.to_csv(f"{ROOT}/models/{self.name}_evaluation/{self.name}_feature_importance.csv", index=False)
        return df