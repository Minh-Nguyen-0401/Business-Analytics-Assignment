import os
import joblib
import yaml
import pandas as pd
import numpy as np
from tabulate import tabulate
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, PowerTransformer
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.feature_selection import mutual_info_regression
from sklearn.pipeline import Pipeline
import logging

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
root = os.path.dirname(CURRENT_DIR)

import sys
sys.path.insert(0, root)
from utils.helper_func import *

LOG_DIR = os.path.join(os.path.dirname(__file__), 'log')
os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(
    filename=f"{LOG_DIR}/ingest.log",
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    filemode="w"
)
logger = logging.getLogger()

class FeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def winsorize_num_cols(self, X):
        Xc = X.copy()
        bounds = {}
        for col in Xc.select_dtypes(include=np.number).columns:
            q1, q3 = Xc[col].quantile([.25, .75])
            iqr = q3 - q1
            lo, hi = q1 - 1.5*iqr, q3 + 1.5*iqr
            Xc[col] = Xc[col].clip(lo, hi)
            bounds[col] = (lo, hi)
        Xc[Xc.select_dtypes(include=np.number).columns] = Xc.select_dtypes(include=np.number).clip(-1e9, 1e9)
        return Xc, bounds

    def calc_VIF(self, X):
        Xf = X.fillna(X.mean(numeric_only=True))
        features = X.columns.tolist()
        vif_df = pd.DataFrame({
            'feature': features,
            'vif': [variance_inflation_factor(Xf.values, i) for i in range(len(features))]
        }).sort_values('vif', ascending=False)
        self.vif_df = vif_df
        print(tabulate(vif_df, headers="keys", tablefmt="fancy_grid"))

    def filter_by_vif(self, X, lower_thres=1, upper_thres=np.inf):
        to_remove = self.vif_df[
            (self.vif_df['vif'] <= lower_thres) |
            (self.vif_df['vif'] >= upper_thres) |
            (self.vif_df['vif'].isnull())
        ]['feature'].tolist()

        # exceptional case for promo columns:
        promo_type_feat = [col for col in self.vif_df['feature'].unique().tolist() if bool(re.match(r"(?i)(promo_type|days_since_last_promo).*",col))]
        remaining_feats = [c for c in X.columns.tolist() if c not in [i for i in to_remove if i not in promo_type_feat]]
        logger.info(f"Dropped {len([i for i in to_remove if i not in promo_type_feat])} features with VIF <= {lower_thres} or >= {upper_thres}")
        return X[remaining_feats]

    def filter_by_mi(self, X, y, percentile=0.5):
        num_cols = X.select_dtypes(include=np.number).columns
        Xf = X[num_cols].fillna(X[num_cols].mean())
        yf = y.fillna(y.mean())
        mi = mutual_info_regression(Xf, yf, discrete_features='auto')
        mi_series = pd.Series(mi, index=num_cols)
        mi_df = pd.DataFrame(mi_series, columns=['MI'])
        thresh = np.percentile(mi_series, percentile*100)
        selected = mi_series[mi_series >= thresh].index.tolist()
        print(tabulate(mi_df, headers="keys", tablefmt="fancy_grid"))
        logger.info(f"Dropped {len(num_cols) - len(selected)} features with MI >= {percentile}")
        return X[selected], mi_series

class CustomPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.config = self.load_config()
        self.feature_selector = FeatureSelector()

    def load_config(self):
        with open(os.path.join(CURRENT_DIR, 'config.yaml'), 'r') as f:
            return yaml.safe_load(f)

    def fit(self, X, y, mi_pct):
        Xc = X.copy()
        org_features = Xc.columns
        min_non_null = int(0.1 * len(Xc))
        Xc = Xc.dropna(axis=1, thresh=min_non_null)
        logger.info(f"Drop columns with more than {min_non_null} non-null values: {org_features.difference(Xc.columns)}")

        # initial impute for pipeline
        num_feats = Xc.select_dtypes(include=np.number).columns.tolist()
        test_1 = num_feats.copy()
        num_feats = [c for c in num_feats if Xc[c].nunique() > 1]
        test_2 = num_feats.copy()
        logger.info(f"Drop columns with only one unique value: {[c for c in test_1 if c not in test_2]}")
        cat_feats = Xc.columns.difference(num_feats).tolist()
        Xc[num_feats] = Xc[num_feats].clip(-1e9,1e9)

        numeric_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('yeo', PowerTransformer(method='yeo-johnson')),
            ('minmax', MinMaxScaler())
        ])
        self.preprocessor = ColumnTransformer([
            ('num_pipeline', numeric_pipeline, num_feats),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False, drop='first'), cat_feats)
        ])
        # fit and transform
        X_trans = self.preprocessor.fit_transform(Xc)
        feat_names = self._get_feature_names()
        Xc = pd.DataFrame(X_trans, columns=feat_names, index=Xc.index)

        # now apply winsorize, MI, VIF sequentially
        if self.config['preprocess']['winsorize']:
            Xc, self.bound_dict = self.feature_selector.winsorize_num_cols(Xc)
        if self.config['preprocess']['use_mi']:
            Xc, self.mi_series = self.feature_selector.filter_by_mi(Xc, y, mi_pct)
        if self.config['preprocess']['use_vif']:
            self.feature_selector.calc_VIF(Xc)
            Xc = self.feature_selector.filter_by_vif(Xc)

        self.rel_features = Xc.columns.tolist()
        logger.info(f"Overall dropped {len(org_features) - len(self.rel_features)} features, including {org_features.difference(self.rel_features)}")
        # logger.info(f"Drop {len(org_features) - len(self.rel_features)} features, including {org_features.difference(self.rel_features)}")

    def transform(self, X):
        Xc = X.copy()
        num_feats = Xc.select_dtypes(include=np.number).columns.tolist()
        Xc[num_feats] = Xc[num_feats].clip(-1e9,1e9)
        # pipeline transform
        X_trans = self.preprocessor.transform(Xc)
        feat_names = self._get_feature_names()
        Xc = pd.DataFrame(X_trans, columns=feat_names, index=Xc.index)
        # winsorize
        if 'bound_dict' in self.__dict__:
            for col, (lo, hi) in self.bound_dict.items():
                if col in Xc.columns:
                    Xc[col] = Xc[col].clip(lo, hi).clip(-1e9, 1e9)
        # select MI/VIF features
        return Xc[self.rel_features]

    def _get_feature_names(self):
        num_cols = self.preprocessor.transformers_[0][2]
        ohe = self.preprocessor.transformers_[1][1]
        cat_cols = self.preprocessor.transformers_[1][2]
        ohe_names = ohe.get_feature_names_out(cat_cols) if hasattr(ohe, 'get_feature_names_out') else []
        return list(num_cols) + list(ohe_names)
