import pandas as pd
import numpy as np
import os, re

def ffill_timeseries_resamp(df, time_col, cur_fmt, to_period):
    try:
        df[time_col] = pd.to_datetime(df[time_col], format=cur_fmt)
        current_freq = pd.infer_freq(df[time_col])
        print(f"Original frequency: {current_freq}")

        df = df.set_index(time_col).sort_index(ascending=True).resample(to_period).ffill()
        return df
    except Exception as e:
        print(e)

def ms_converter(data, time_col, fmt="%Y-%m-%d"):
    try:
        data[time_col] = pd.to_datetime(data[time_col], format=fmt)
        current_freq = pd.infer_freq(data[time_col])
        print(f"Original frequency: {current_freq}")

        output = data[time_col].dt.to_period("M")
        print(f"Converted format: {output.dtype}")
        data[time_col] = output
        return data
    except Exception as e:
        print(e)

def standardize_name(x):
    x = re.sub("[^a-zA-Z0-9]", " ", x)
    x = re.sub("\s+", " ", x)
    x = x.strip().replace(" ", "_").lower()
    return x
