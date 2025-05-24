import yaml
import pandas as pd
import numpy as np
import os, re
from pathlib import Path
import logging
import warnings
warnings.filterwarnings("ignore")

from utils.helper_func import ms_converter, ffill_timeseries_resamp
from utils.feature_generation import (
    generate_this_year_agg_feats,
    generate_this_quarter_agg_feats,
    generate_lagged_feats,
    generate_rolling_std_feats,
    generate_ewm_feats,
    generate_diff_feats,
    generate_pct_change_feats,
    cal_rsi,
    force_backshift
)

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(CURRENT_DIR, 'log')
os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(
    filename=f"{LOG_DIR}/aggregate.log",
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    filemode="w"
)
logger = logging.getLogger()

class Aggregation:
    def __init__(self, data_dict, required_shiftback=12):
        self.config = yaml.safe_load(open(os.path.join(CURRENT_DIR, 'config.yaml')))
        self.data_dict = data_dict
        self.required_shiftback = required_shiftback
        self.org_cols = None
        self.supp_cols = None

    def merge_data(self):
        sales_df = ms_converter(self.data_dict['sales_data'], "Date", "%Y-%m-%d")
        promo_df = ms_converter(self.data_dict['promotion'], "Active_Month", "%Y-%m")
        org_merge = (
            sales_df
            .merge(promo_df.drop(columns=['Promo_ID'], errors='ignore'),
                   left_on='Date', right_on='Active_Month', how='left')
            .drop("Active_Month", axis=1)
        )
        org_merge.set_index("Date", inplace=True)
        org_merge.index = org_merge.index.to_timestamp().to_period('M').to_timestamp()
        org_merge.sort_index(inplace=True)
        if "Unnamed: 0" in org_merge.columns:
            org_merge.drop("Unnamed: 0", axis=1, inplace=True)
        self.org_cols = org_merge.columns.tolist()
        self.org_df = org_merge

        cas = self.data_dict['casual_outerwear_avg_cpi'].reset_index().rename(columns={'year':'Date'})
        cas = ffill_timeseries_resamp(cas, "Date", "%Y", "MS")

        indus = self.data_dict['industry_data'].reset_index().rename(columns={'index':'Date'})
        indus["Date"] = pd.to_datetime(indus["Date"], format="%Y-%m-%d")
        indus.set_index("Date", inplace=True)
        indus.index = indus.index.to_period('M').to_timestamp()

        gold = self.data_dict['gold_data']
        mac  = self.data_dict['macro_data']
        tick = self.data_dict['rel_tickers_data']
        for df in (gold, mac, tick):
            df.index = pd.to_datetime(df.index, format="%Y-%m-%d").to_period('M').to_timestamp()
        gold.columns = [f"{c}_gold" for c in gold.columns]

        agg_macro = cas.merge(gold,    left_index=True, right_index=True, how='outer')
        agg_macro = agg_macro.merge(indus,left_index=True, right_index=True, how='outer')
        agg_macro = agg_macro.merge(mac,   left_index=True, right_index=True, how='outer')
        agg_macro = agg_macro.merge(tick,  left_index=True, right_index=True, how='outer')
        agg_macro.sort_index(inplace=True)
        self.supp_cols = agg_macro.columns.tolist()

        return org_merge.merge(agg_macro, left_index=True, right_index=True, how='outer')

    def engineer_feats(self, agg_df):
        df = agg_df.copy()
        df.index = df.index.to_period('M').to_timestamp()
        df['Year'] = df.index.year
        df['Month'] = df.index.month.astype(str)
        df['Quarter'] = df.index.quarter.astype(str)
        for period, freq in (('month',12), ('quarter',4)):
            df[f'{period}_sin'] = np.sin(2*np.pi*(getattr(df.index,period)-1)/freq)
            df[f'{period}_cos'] = np.cos(2*np.pi*(getattr(df.index,period)-1)/freq)

        df['Budget_USD'].fillna(0, inplace=True)
        df['Promo_Type'].fillna('No_Promo', inplace=True)

        exclude = ['Promo_Type','Year','Month','Quarter',
                   'month_sin','month_cos','quarter_sin','quarter_cos']
        df, _ = generate_this_year_agg_feats(df, ["Budget_USD", "New_Sales"], ['Year'])
        df, _ = generate_this_quarter_agg_feats(df, ["Budget_USD", "New_Sales"], ['Year','Quarter'])

        ws = self.config.get('aggregation',{}).get('lags',[3,6,12])
        feats = df.columns.difference(exclude).tolist()
        for fn in (generate_lagged_feats,
                #    generate_rolling_std_feats,
                   generate_ewm_feats,
                #    generate_diff_feats,
                #    generate_pct_change_feats
                   ):
            df = fn(df, feats, ws)

        closes = [c for c in feats if re.match(r'(?i)^close', c)]
        df = cal_rsi(df, closes, self.config.get('aggregation',{}).get('rsi_windows',[6]))

        df['budget_to_sales'] = df['Budget_USD'] / df['New_Sales']
        last = df.index.to_series().where(df['Promo_Type']!='No_Promo').ffill()
        df['months_since_last_promo'] = ((df.index.year-last.dt.year)*12 +
                                        (df.index.month-last.dt.month))

        static = ['months_since_last_promo'] + [c for c in df.columns if re.match(r"(?i)budget_usd.*", c)]
        back_cols = df.columns.difference(exclude+static+['New_Sales']).tolist()
        df, shifted = force_backshift(df, back_cols, self.required_shiftback)

        keep = static + exclude + ['New_Sales'] + shifted
        df = df[keep].iloc[self.required_shiftback:]
        self.agg_features = df.columns.tolist()
        return df

    def split_train_test(self, df, target_feat=None):
        span = self.required_shiftback
        train, test = df.iloc[:-span], df.iloc[-span:]
        self.train_set, self.test_set = train, test
        if target_feat:
            return (train.drop(target_feat,axis=1), train[target_feat],
                    test.drop(target_feat,axis=1),  test[target_feat])
        return train, test