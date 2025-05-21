import os, sys, yaml, numpy as np, pandas as pd, lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.inspection import permutation_importance
from catboost import CatBoostRegressor
import optuna
from optuna.integration import OptunaSearchCV
from optuna.distributions import IntDistribution, FloatDistribution
from sklearn.linear_model import LinearRegression

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
root = os.path.dirname(CURRENT_DIR)
sys.path.insert(0, root)

class HyperTuner:
    def __init__(self, model_name, n_features, ts_splits,
                 search_space_path=os.path.join(CURRENT_DIR, 'search_space.yaml'),
                 output_path=None):
        self.name = model_name
        self.nf = n_features
        self.splits = ts_splits
        self.space_path = search_space_path
        self.out_path = output_path or search_space_path
        self.X = self.y = None

    def run_permutation_importance(self, X, y):
        self.X, self.y = X, y
        if self.name == 'lightgbm':
            model = lgb.LGBMRegressor(random_state=42, verbose=-1)
        elif self.name == 'catboost':
            model = CatBoostRegressor(verbose=0, random_state=42)
        elif self.name == 'linear_regression':
            model = LinearRegression()
        else:
            raise ValueError(f"Model not yet integrated: {self.name}")
        imp = np.zeros(X.shape[1])
        for tr, te in TimeSeriesSplit(n_splits=self.splits).split(X):
            model.fit(X.iloc[tr], y.iloc[tr])
            imp += permutation_importance(
                model, X.iloc[te], y.iloc[te],
                n_repeats=5, random_state=0
            ).importances_mean
        imp /= self.splits
        feats = X.columns[np.argsort(imp)[::-1][:self.nf]].tolist()
        data = yaml.safe_load(open(self.out_path)) or {}
        data.setdefault('results', {})
        data['results'].setdefault(self.name, {})['features'] = feats
        yaml.safe_dump(data, open(self.out_path,'w'), sort_keys=False)
        return feats

    def _build_distributions(self, space):
        dist = {}
        for p,c in space.items():
            low,high,log = c['low'],c['high'],c.get('log',False)
            if c['type']=='int':
                dist[p] = IntDistribution(low=low, high=high, step=1, log=log)
            else:
                dist[p] = FloatDistribution(low=low, high=high, log=log)
        return dist

    def tune(self, n_trials=50):
        if self.X is None:
            raise RuntimeError('Call run_permutation_importance first')
        data = yaml.safe_load(open(self.out_path)) or {}
        feats = data['results'][self.name]['features']
        space = yaml.safe_load(open(self.space_path))[self.name]
        dist = self._build_distributions(space)
        Xsub = self.X[feats]
        if self.name == 'lightgbm':
            model = lgb.LGBMRegressor(random_state=42)
        elif self.name == 'catboost':
            model = CatBoostRegressor(verbose=0, random_state=42)
        elif self.name == 'linear_regression':
            model = LinearRegression()
        else:
            raise ValueError(f"Unknown model: {self.name}")
        opt = OptunaSearchCV(
            model,
            param_distributions=dist,
            cv=TimeSeriesSplit(n_splits=self.splits),
            scoring='neg_root_mean_squared_error',
            n_trials=max(1,n_trials),
            random_state=0
        )
        opt.fit(Xsub, self.y)
        best = opt.best_params_
        data.setdefault('results', {})
        data['results'].setdefault(self.name, {})['params'] = best
        yaml.safe_dump(data, open(self.out_path,'w'), sort_keys=False)
        # return best