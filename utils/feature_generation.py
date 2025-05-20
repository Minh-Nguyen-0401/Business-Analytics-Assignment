import pandas as pd
import numpy as np
import os, re

def force_backshift(df, col_list, required_window):
    df_copy = df.copy()
    org_list = df_copy.columns.tolist()
    df_copy[[f"{col}_backshift_{required_window}" for col in col_list]] = df_copy[col_list].transform(lambda x: x.shift(required_window))
    bshifted_feats = [c for c in df_copy.columns if c not in org_list]
    print(f"Backshifted {col_list} by {required_window}")
    return df_copy, bshifted_feats

def generate_lagged_feats(df, col_list, windows):
    df_copy = df.copy()
    for window in windows:
            df_copy[[f"{col}_lag_{window}" for col in col_list]] = df_copy[col_list].transform(lambda x: x.shift(window))
    return df_copy

def generate_ma_feats(df, col_list, windows):
    df_copy = df.copy()
    for window in windows:
            df_copy[[f"{col}_roll_ma_{window}" for col in col_list]] = df_copy[col_list].transform(lambda x: x.rolling(window).mean())
    return df_copy

def generate_rolling_std_feats(df, col_list, windows):
    df_copy = df.copy()
    if 1 in windows:
        windows.remove(1)
    for window in windows:
            df_copy[[f"{col}_rolling_std_{window}" for col in col_list]] = df_copy[col_list].transform(lambda x: x.rolling(window).std())
    return df_copy

def generate_ewm_feats(df, col_list, windows):
    df_copy = df.copy()
    for window in windows:
            df_copy[[f"{col}_ewm_{window}" for col in col_list]] = df_copy[col_list].transform(lambda x: x.ewm(span=window).mean())
    return df_copy

def generate_diff_feats(df, col_list, windows):
    df_copy = df.copy()
    for window in windows:
            df_copy[[f"{col}_diff_{window}" for col in col_list]] = df_copy[col_list].transform(lambda x: x.diff(window))
    return df_copy

def generate_pct_change_feats(df, col_list, windows):
    df_copy = df.copy()
    for window in windows:
            df_copy[[f"{col}_pct_change_{window}" for col in col_list]] = df_copy[col_list].transform(lambda x: x.pct_change(window))
    return df_copy

# aggregation by year/quarter func
def generate_this_year_agg_feats(df, col_list, groupby_cols):
    df_copy = df.copy()
    if 'Year' not in df_copy.columns:
        df_copy['Year'] = df_copy.index.year
    year_groupby = [c for c in groupby_cols if c.lower() == 'year']
    agg = df_copy.groupby(year_groupby)[col_list].agg(['mean','std'])
    agg.columns = [f"{col}_{stat}_this_year" for col, stat in agg.columns]
    agg.reset_index(inplace=True)
    df_merged = df_copy.reset_index().merge(agg, how='left', on=year_groupby).set_index(df_copy.index.name)
    agg_col_list = [c for c in agg.columns if c not in year_groupby]
    return df_merged, agg_col_list

def generate_this_quarter_agg_feats(df, col_list, groupby_cols):
    df_copy = df.copy()
    if 'Year' not in df_copy.columns:
        df_copy['Year'] = df_copy.index.year
    if 'Quarter' not in df_copy.columns:
        df_copy['Quarter'] = df_copy.index.quarter
    year_groupby = [c for c in groupby_cols if c.lower() == 'year']
    quarter_groupby = [c for c in groupby_cols if c.lower() == 'quarter']
    grp = year_groupby + quarter_groupby
    agg = df_copy.groupby(grp)[col_list].agg(['mean','std'])
    agg.columns = [f"{col}_{stat}_this_quarter" for col, stat in agg.columns]
    agg.reset_index(inplace=True)
    df_merged = df_copy.reset_index().merge(agg, how='left', on=grp).set_index(df_copy.index.name)
    agg_col_list = [c for c in agg.columns if c not in grp]
    return df_merged, agg_col_list


# stock indicators
def cal_rsi(df, ticker_close_cols: list, windows: list):
    df_copy = df.copy()
    if not isinstance(windows, (list, tuple)):
        windows = [windows]
    for window in windows:
        try:
            for ticker in ticker_close_cols:
                rsi_colname = f"{ticker}_rsi_{window}M"
                avg_gain = df_copy[ticker].transform(
                    lambda x: x.where(x > 0, 0).rolling(window=window, min_periods=window).mean())
                avg_loss = df_copy[ticker].transform(
                    lambda x: x.where(x < 0, 0).abs().rolling(window=window, min_periods=window).mean())
                # calculate rsi
                rs = avg_gain / avg_loss.replace(0, 1e-10)
                df_copy[rsi_colname] = 100 - 100 / (1 + rs)
                # print(f"{rsi_colname} generated")
        except Exception as e:
            print(e)
    return df_copy