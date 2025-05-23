import yaml
import pandas as pd
import os, re
import glob
from pathlib import Path
from collections import defaultdict
import logging
import warnings
warnings.filterwarnings("ignore")

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
root = os.path.dirname(CURRENT_DIR)

import sys
sys.path.insert(0, root)
from utils.helper_func import *
from utils.feature_generation import *

LOG_DIR = os.path.join(os.path.dirname(__file__), 'log')
os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(filename=f"{LOG_DIR}/aggregate.log", level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", filemode = "w")
logger = logging.getLogger()

class Aggregation:
    def __init__(self, data_dict):
        self.config = self.load_config()
        self.data_dict = data_dict
        self.org_cols = None
        self.supp_cols = None
    
    def load_config(self):
        with open(os.path.join(CURRENT_DIR, 'config.yaml'), 'r') as f:
            config = yaml.safe_load(f)
        return config

    def merge_data(self):
        data_dict = self.data_dict
        for key, df in data_dict.items():
            setattr(self, key, df)
        
        # merge sales & promotion data
        sales_df = ms_converter(self.sales_data, "Date", "%Y-%m-%d")
        promo_df = ms_converter(self.promotion, "Active_Month", "%Y-%m")

        org_merge = sales_df.merge(promo_df, how = "left", left_on = "Date", right_on = "Active_Month").drop("Active_Month", axis = 1)
        org_merge.set_index("Date", inplace = True)
        org_merge.index = org_merge.index.to_timestamp()
        org_merge.index = org_merge.index.to_period('M').to_timestamp()
        org_merge.sort_index(ascending = True, inplace = True)
        if "Unnamed: 0" in org_merge.columns:
            org_merge.drop("Unnamed: 0", axis = 1, inplace = True)
        org_merge.drop("Promo_ID", axis = 1, inplace = True)

        self.org_cols = org_merge.columns

        # merge macro data
        cas_wear_df = self.casual_outerwear_avg_cpi
        cas_wear_df.reset_index(inplace = True)
        cas_wear_df.rename(columns = {"year":"Date"}, inplace = True)
        cas_wear_df = ffill_timeseries_resamp(cas_wear_df, "Date", "%Y", "MS")

        gold_df = self.gold_data
        indus_df = self.industry_data
        mac_df = self.macro_data
        ticker_df = self.rel_tickers_data

        indus_df.reset_index(inplace=True)
        indus_df.rename(columns = {"index":"Date"}, inplace = True)
        indus_df["Date"] = pd.to_datetime(indus_df["Date"], format="%Y-%m-%d")
        indus_df.set_index("Date", inplace = True)
        indus_df.index = indus_df.index.to_period('M').to_timestamp()

        for df in [gold_df, mac_df, ticker_df]:
            df.index = pd.to_datetime(df.index, format="%Y-%m-%d").to_period("M").to_timestamp()
        
        gold_df.columns = [f"{col}_gold" for col in gold_df.columns]
        agg_macro_df = pd.merge(cas_wear_df, gold_df, how="outer", left_index=True, right_index=True)
        agg_macro_df = pd.merge(agg_macro_df, indus_df, how="outer", left_index=True, right_index=True)
        agg_macro_df = pd.merge(agg_macro_df, mac_df, how="outer", left_index=True, right_index=True)
        agg_macro_df = pd.merge(agg_macro_df, ticker_df, how="outer", left_index=True, right_index=True)
        agg_macro_df.sort_index(ascending = True, inplace = True)

        self.supp_cols = agg_macro_df.columns

        # merge data
        agg_df = pd.merge(org_merge, agg_macro_df, how="outer", left_index=True, right_index=True)
        return agg_df
    
    def engineer_feats(self, agg_df, required_shiftback=12):
        """_summary_

        During inference, we need to shift back N months to calculate the input features for the testing/forecasting
        Args:
            agg_df (_type_): _description_
        """
        self.required_shiftback = required_shiftback

        # Time-based features
        agg_df["Year"] = agg_df.index.year
        agg_df["Month"] = agg_df.index.month
        agg_df["Quarter"] = agg_df.index.quarter
        logger.info("Finished generating time-based features")

        # Fillna for necessary columns
        agg_df["Budget_USD"] = agg_df["Budget_USD"].fillna(0)
        agg_df["Promo_Type"] = agg_df["Promo_Type"].fillna("No_Promo")

        exclude_feats = [
                'Promo_Type',
                'Year',
                'Month',
                'Quarter'
            ]
        # Generate directly last year agg features & YoY
        agg_df, this_year_agg_feats = generate_this_year_agg_feats(agg_df, agg_df.columns.difference(exclude_feats+["Month","Quarter"]), ["Year"])
        agg_df, this_qrt_agg_feats = generate_this_quarter_agg_feats(agg_df, agg_df.columns.difference(exclude_feats+this_year_agg_feats+["Month"]), ["Year","Quarter"])
        logger.info("Finished generating agg_facts this year/quarter")

        # Mass feature gen
        ws = [3, 6, 12]
        
        include_feats = agg_df.columns.difference(exclude_feats).tolist()

        agg_df = generate_lagged_feats(agg_df, include_feats, ws)
        agg_df = generate_rolling_std_feats(agg_df, include_feats, ws)
        agg_df = generate_ewm_feats(agg_df, include_feats, ws)
        agg_df = generate_diff_feats(agg_df, include_feats, ws)
        agg_df = generate_pct_change_feats(agg_df, include_feats, ws)

        logger.info("Finished generating mass_gen features")

        # experimental ...
        ticker_cols = [col for col in include_feats if bool(re.match(r"(?i)close.*", col))]
        rsi_span = [6]
        agg_df = cal_rsi(agg_df, ticker_cols, rsi_span)
        logger.info("Finished generating RSI features")

        # Promo Indicators
        agg_df["has_promo"] = agg_df["Promo_Type"].apply(lambda x: 1 if x != "No_Promo" else 0)
        agg_df["budget_to_sales"] = agg_df["Budget_USD"] / agg_df["New_Sales"]
        agg_df["last_promo_time"] = agg_df.index.to_series().where(agg_df["has_promo"] == 1).ffill()
        agg_df["months_since_last_promo"] = (agg_df.index.year - agg_df["last_promo_time"].dt.year) * 12 + (agg_df.index.month - agg_df["last_promo_time"].dt.month)
        agg_df.drop(["last_promo_time", "has_promo"], axis = 1, inplace = True)
        logger.info("Finished generating Promo Indicators")

        promo_to_keep_unchanged = ["budget_to_sales", "months_since_last_promo"] + [c for c in agg_df.columns if bool(re.match(r"(?i)(budget|promo_type).*",c))]

        # Shift back
        # all_org_cols_drop = self.org_cols + self.supp_cols
        backshift_cols = agg_df.columns.difference(promo_to_keep_unchanged + ["Year", "Month", "Quarter"]).tolist()
        agg_df, bshifted_feats = force_backshift(agg_df, backshift_cols, required_shiftback)

        # Keep all promo_cols to date
        agg_df.drop(agg_df.columns.difference(bshifted_feats + promo_to_keep_unchanged + ["New_Sales", "Year", "Month", "Quarter"]).tolist(), axis=1, inplace=True)
        logger.info(f"Finished shifting back predictive features {required_shiftback} months")

        # Regenerate new features for promos_unchanged:
        promo_gen_new = ["budget_to_sales"]
        agg_df = generate_lagged_feats(agg_df, promo_gen_new + ["months_since_last_promo"], ws)
        agg_df = generate_rolling_std_feats(agg_df, promo_gen_new, ws)
        agg_df = generate_ewm_feats(agg_df, promo_gen_new, ws)
        agg_df = generate_diff_feats(agg_df, promo_gen_new, ws)
        agg_df = generate_pct_change_feats(agg_df, promo_gen_new, ws)        

        # Drop N first rows
        agg_df = agg_df.iloc[required_shiftback:]
        self.agg_featurees = agg_df.columns.tolist()
        return agg_df
    
    def split_train_test(self, agg_df, target_feat=None):
        span = self.required_shiftback
        train_df = agg_df[:-span]
        test_df = agg_df[-span:]

        self.train_set = train_df
        self.test_set = test_df
        if target_feat is None:
            return train_df, test_df
        else:
            X_train, X_test = train_df.drop(target_feat, axis=1), test_df.drop(target_feat, axis=1)
            y_train, y_test = train_df[target_feat], test_df[target_feat]
            return X_train, y_train, X_test, y_test


        