import pandas as pd
import numpy as np
import os, re
import yaml
import yfinance as yf
from dotenv import load_dotenv
from fredapi import Fred
load_dotenv()
import logging
import warnings
warnings.filterwarnings("ignore")


CONFIG_DIR = os.path.join(os.path.dirname(__file__), 'supplementary_data', 'collect_config.yaml')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'supplementary_data')
LOG_DIR = os.path.join(os.path.dirname(__file__), 'log')

os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(filename=f"{LOG_DIR}/collect_macro.log", level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", filemode = "w")
logger = logging.getLogger()

fred_key = os.getenv("FRED_API_KEY")
if not fred_key:
    raise ValueError("Please set the FRED_API_KEY environment variable.")

def ensure_monthly(df):
    freq = pd.infer_freq(df.index)
    if freq in ('MS', 'M'):
        dfm = df.copy()
        dfm.index = dfm.index.to_period('M').to_timestamp()
    else:
        dfm = df.resample('M').mean()
        dfm.index = dfm.index.to_period('M').to_timestamp()
    return dfm.ffill()

def flatten_col_name(df):
    df.columns = [f"{col[0]}_{col[1]}" for col in df.columns]
    return df
def main():
    fred = Fred(api_key=fred_key)

    industry_specific_idx_dict = {
    'retail_sales_apparel': 'RSCCAS',
    'inv_sales_ratio_apparel': 'MRTSIR448USS',
    'pce_apparel': 'DCAFRC1A027NBEA',
    'real_pce_apparel': 'DCAFRX1A020NBEA',
    'cotton_price': 'PCOTTINDUSDM',
    'raw_cotton_ppi': 'WPU01510101'
    }

    macro_idx_dict = {
        'gdp': 'GDPC1',
        'cpi_u': 'CPIAUCSL',
        'oil': 'DCOILWTICO',
        'gold': 'GOLDAMGBD228NLBM',
        'usd_index': 'DTWEXBGS',
        'sp500': 'SP500',
        'umich_sentiment': 'UMCSENT',
        'fed_funds_rate': 'FEDFUNDS',
        'unemployment_rate': 'UNRATE',
        'industrial_prod': 'INDPRO',
        'import_price_all': 'IMPITW'
    }
    
    with open(CONFIG_DIR, 'r') as f:
        config = yaml.safe_load(f)
    min_year = config['min_year']
    max_year = config['max_year']

    series_ind_data, errors = {}, {}
    for name, code in industry_specific_idx_dict.items():
        try:
            logger.info(f"Fetching {name} ({code})")
            s = fred.get_series(
                code,
                observation_start=f"{min_year}-01-01",
                observation_end=f"{max_year}-12-31"
            ).rename(name)
            s = ensure_monthly(s)
            series_ind_data[name] = s
        except Exception as e:
            logger.error(f"{name} ({code}) failed: {e}")
            errors[name] = str(e)

    series_macro_data, errors = {}, {}
    for name, code in macro_idx_dict.items():
        try:
            logger.info(f"Fetching {name} ({code})")
            s = fred.get_series(
                code,
                observation_start=f"{min_year}-01-01",
                observation_end=f"{max_year}-12-31"
            ).rename(name)

            s = ensure_monthly(s)
            series_macro_data[name] = s
        except Exception as e:
            logger.error(f"{name} ({code}) failed: {e}")
            errors[name] = str(e)

    series_ind_data = pd.concat(series_ind_data.values(), axis=1, join='outer').sort_index(ascending=True)
    series_macro_data = pd.concat(series_macro_data.values(), axis=1, join='outer').sort_index(ascending=True)
    logger.info("Finished downloading data for industry and macro indicators")

    # gold commodities
    gold_df = yf.download('GC=F', start=f"{min_year}-01-01", end=f"{max_year}-12-31")\
        .droplevel('Ticker', axis=1)[["Close", "Volume"]]
    gold_df = gold_df.reset_index()
    gold_df["Date"] = gold_df["Date"].dt.to_period('M').dt.to_timestamp()
    gold_df_agg = gold_df.groupby("Date").agg({"Close": "mean", "Volume": "sum"})
    logger.info("Finished downloading data for gold")

    # tickers
    tickers = [
    "XRT",              # SPDR S&P Retail ETF
    "XLY",              # SPDR Consumer Discretionary Select Sector ETF
    "VCR",              # Vanguard Consumer Discretionary ETF
    "RTH"               # VanEck Retail ETF
    ]

    data = yf.download(
        tickers,
        start=f"{min_year}-01-01",
        end=f"{max_year}-12-31"
    )[["Close", "Volume"]]

    data = flatten_col_name(data)
    data = data.reset_index()
    data["Date"] = data["Date"].dt.to_period('M').dt.to_timestamp()
    vol_cols = [col for col in data.columns if "Volume" in col]
    price_cols = [col for col in data.columns if "Close" in col]
    data = data.groupby("Date").agg({**{vol: "sum" for vol in vol_cols}, **{price: "mean" for price in price_cols}})
    logger.info("Finished downloading data for tickers")

    # save data
    series_ind_data.to_csv(os.path.join(OUTPUT_DIR, 'industry_data.csv'))
    series_macro_data.to_csv(os.path.join(OUTPUT_DIR, 'macro_data.csv'))
    gold_df_agg.to_csv(os.path.join(OUTPUT_DIR, 'gold_data.csv'))
    data.to_csv(os.path.join(OUTPUT_DIR, 'rel_tickers_data.csv'))
    logger.info("Finished saving data")

if __name__ == '__main__':
    main()


