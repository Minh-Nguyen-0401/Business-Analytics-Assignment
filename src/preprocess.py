import yaml
import pandas as pd
import numpy as np
import os
import re
from pathlib import Path
import logging
from tabulate import tabulate
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, PowerTransformer, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import mutual_info_regression
from statsmodels.stats.outliers_influence import variance_inflation_factor

BASE_DIR = Path(__file__).resolve().parent
CONFIG = yaml.safe_load((BASE_DIR / 'config.yaml').open())
LOG_DIR = BASE_DIR / 'log'
LOG_DIR.mkdir(exist_ok=True)
logging.basicConfig(
    filename=str(LOG_DIR / 'preprocess.log'),
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    filemode='w'
)
logger = logging.getLogger(__name__)

class FeatureSelector(BaseEstimator, TransformerMixin):
    def winsorize(self, X):
        Xc, bounds = X.copy(), {}
        cols = [c for c in Xc.select_dtypes(include=np.number).columns
                if not re.search(r'(?i)(month|quarter|promo_type)', c)]
        for c in cols:
            q1, q3 = Xc[c].quantile([.25, .75])
            iqr = q3 - q1
            lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            Xc[c] = Xc[c].clip(lo, hi)
            bounds[c] = (lo, hi)
        Xc[Xc.select_dtypes(include=np.number).columns] = \
            Xc.select_dtypes(include=np.number).clip(-1e9, 1e9)
        return Xc, bounds

    def vif(self, X):
        num = [c for c in X.select_dtypes(include=np.number).columns
               if not re.search(r'(?i)(promo_type|month|quarter)', c)]
        Xf = X[num].fillna(X[num].mean(numeric_only=True))
        df = pd.DataFrame({
            'feature': num,
            'vif': [variance_inflation_factor(Xf.values, i) for i in range(len(num))]
        }).sort_values('vif', ascending=False)
        self.vif_df = df
        print(tabulate(df.head(), headers='keys', tablefmt='fancy_grid'))

    def filter_vif(self, X, low=-np.inf, high=np.inf):
        drop = self.vif_df[(self.vif_df.vif <= low) |
                           (self.vif_df.vif >= high) |
                           (self.vif_df.vif.isnull())]['feature']
        keep_exc = [c for c in self.vif_df.feature
                    if re.match(r'(?i)(Promo_Type|days_since_last_promo|month|quarter)', c)]
        cols = [c for c in X.columns if c not in drop or c in keep_exc]
        logger.info(f'Dropped {len([c for c in drop if c not in keep_exc])} features by VIF')
        return X[cols]

    def mi(self, X, y, pct):
        num = [c for c in X.select_dtypes(include=np.number).columns
               if not re.search(r'(?i)(promo_type|month|quarter)', c)]
        Xf = X[num].fillna(X[num].mean())
        yf = y.fillna(y.mean())
        scores = mutual_info_regression(Xf, yf)
        series = pd.Series(scores, index=num)
        thresh = np.percentile(series, pct * 100)
        keep = series[series >= thresh].index.tolist()
        print(tabulate(pd.DataFrame(series, columns=['MI']).head(), headers='keys', tablefmt='fancy_grid'))
        logger.info(f'Dropped {len(num) - len(keep)} features by MI')
        return X[keep], series

class CustomPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.conf = CONFIG['preprocess']
        self.sel = FeatureSelector()

    def fit(self, X, y, mi_pct):
        print("\n=== Initial DataFrame ===")
        print(f"Columns: {X.columns.tolist()}")
        print(f"Dtypes:\n{X.dtypes}")
        
        Xc = X.dropna(axis=1, thresh=int(0.05 * len(X)))
        print("\n=== After dropna ===")
        print(f"Remaining columns: {Xc.columns.tolist()}")
        
        cycle_feats = [c for c in ['month_sin', 'month_cos', 'quarter_sin', 'quarter_cos']
                       if c in Xc.columns]
        print(f"\n=== Cycle features ===")
        print(cycle_feats)
        
        num_feats = [c for c in Xc.select_dtypes(include=np.number).columns
                     if c not in cycle_feats]
        num_feats = [c for c in num_feats if Xc[c].nunique() > 1]
        print(f"\n=== Numerical features ===")
        print(f"Count: {len(num_feats)}")
        print(f"First 10: {num_feats[:10]}")
        
        cat_feats = [c for c in Xc.columns if c not in num_feats + cycle_feats]
        print(f"\n=== Categorical features ===")
        print(f"Count: {len(cat_feats)}")
        print(cat_feats)
        
        if num_feats:
            Xc[num_feats] = Xc[num_feats].clip(-1e9, 1e9)
        
        # Create preprocessing pipelines
        transformers = []
        if num_feats:
            num_pipe = Pipeline([
                ('impute', SimpleImputer(strategy='mean')),
                ('std_scaler', StandardScaler()),
                ('scale', MinMaxScaler())
            ])
            transformers.append(('num', num_pipe, num_feats))
        
        if cycle_feats:
            transformers.append(('cycle', 'passthrough', cycle_feats))
        
        if cat_feats:
            cat_pipe = Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='ignore', drop='first', sparse=False))
            ])
            transformers.append(('cat', cat_pipe, cat_feats))
        
        self.pre = ColumnTransformer(transformers, remainder='drop')
        
        arr = self.pre.fit_transform(Xc)
        
        feature_names = []
        for name, trans, cols in self.pre.transformers_:
            if name == 'cat':
                ohe = trans.named_steps['onehot']
                if hasattr(ohe, 'get_feature_names_out'):
                    cat_names = ohe.get_feature_names_out(cols)
                else:
                    cat_names = [f"{col}_{i}" for i, col in enumerate(cols) 
                               for i in range(len(ohe.categories_[i]) - 1)]
                feature_names.extend(cat_names)
            elif name == 'num' or name == 'cycle':
                feature_names.extend(cols)
        
        Xt = pd.DataFrame(arr, columns=feature_names, index=Xc.index)
        
        if num_feats and (self.conf.get('winsorize') or self.conf.get('use_mi') or self.conf.get('use_vif')):
            num_transformed = [f for f in Xt.columns if any(f.startswith(f"num__{n}") for n in num_feats) or f in num_feats]
            
            if num_transformed:
                if self.conf.get('winsorize'):
                    existing_num = [f for f in num_transformed if f in Xt.columns]
                    if existing_num:
                        Xt_selected, self.bounds = self.sel.winsorize(Xt[existing_num])
                        Xt[existing_num] = Xt_selected
                        num_transformed = list(Xt_selected.columns)
            
            if num_transformed:
                if self.conf.get('use_mi') and y is not None:
                    existing_num = [f for f in num_transformed if f in Xt.columns]
                    if existing_num:
                        Xt_selected, self.mi_scores = self.sel.mi(Xt[existing_num], y, mi_pct)
                        non_num_cols = [c for c in Xt.columns if c not in existing_num]
                        selected_cols = list(Xt_selected.columns) + non_num_cols
                        Xt = Xt[selected_cols]
                        num_transformed = list(Xt_selected.columns)
            
            if num_transformed:
                if self.conf.get('use_vif'):
                    existing_num = [f for f in num_transformed if f in Xt.columns]
                    if existing_num:
                        self.sel.vif(Xt[existing_num])
                        Xt_selected = self.sel.filter_vif(Xt[existing_num])
                        non_num_cols = [c for c in Xt.columns if c not in existing_num]
                        selected_cols = list(Xt_selected.columns) + non_num_cols
                        Xt = Xt[selected_cols]
        
        self.features_ = Xt.columns.tolist()
        return self

    def transform(self, X):
        Xc = X.copy()
        
        num_feats = []
        cycle_feats = []
        cat_feats = []
        
        for name, trans, cols in self.pre.transformers_:
            if name == 'num':
                num_feats = cols
            elif name == 'cycle':
                cycle_feats = cols
            elif name == 'cat':
                cat_feats = cols
        
        if num_feats:
            Xc[num_feats] = Xc[num_feats].clip(-1e9, 1e9)
        
        arr = self.pre.transform(Xc)
        
        feature_names = []
        for name, trans, cols in self.pre.transformers_:
            if name == 'cat':
                ohe = trans.named_steps['onehot']
                if hasattr(ohe, 'get_feature_names_out'):
                    cat_names = ohe.get_feature_names_out(cols)
                else:
                    cat_names = [f"{col}_{i}" for i, col in enumerate(cols) 
                               for i in range(len(ohe.categories_[i]) - 1)]
                feature_names.extend(cat_names)
            elif name == 'num' or name == 'cycle':
                feature_names.extend(cols)
        
        Xt = pd.DataFrame(arr, columns=feature_names, index=Xc.index)
        
        if hasattr(self, 'bounds'):
            for c, (lo, hi) in self.bounds.items():
                if c in Xt.columns:
                    Xt[c] = Xt[c].clip(lo, hi).clip(-1e9, 1e9)
        
        available_features = [f for f in self.features_ if f in Xt.columns]
        if not available_features:
            raise ValueError("No features available after transformation. Check if the input data matches the training data structure.")
            
        return Xt[available_features]