import pandas as pd 
import numpy as np
import cpi
import warnings
import os, re
import yaml
from collections import defaultdict
warnings.filterwarnings("ignore")
import logging
logging.basicConfig(filename=f"./log/extract_inflation.log", level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", filemode = "w")
logger = logging.getLogger()

casual_outerwear_items = [
    "Men's shirts and sweaters",
    "Men's pants and shorts",
    "Men's suits, sport coats, and outerwear",
    "Women's outerwear",
    "Women's dresses",
    "Women's suits and separates"
]


SALES_PATH = os.path.join(os.path.dirname(__file__), 'original_data/Sales_Data.csv')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'supplementary_data')

BASE_PRICE = 100

def standardize_itemname(x):
    x = re.sub("[^a-zA-Z0-9]", " ", x)
    x = re.sub("\s+", " ", x)
    x = x.strip().replace(" ", "_").lower()
    return x

def main():
    sales_df = pd.read_csv(SALES_PATH)
    sales_df["Date"] = pd.to_datetime(sales_df["Date"])
    sales_df["Year"] = sales_df["Date"].dt.year
    min_year = sales_df["Year"].min()
    max_year = sales_df["Year"].max()

    CONFIG_PATH = os.path.join(OUTPUT_DIR, "collect_config.yaml")
    with open(CONFIG_PATH, "w") as f:
        yaml.dump({"min_year": min_year, "max_year": max_year}, f)

    final_inflation = defaultdict()
    for year in range(min_year, max_year + 1):
        temp_inflation_rate = defaultdict(float)
        total_cpi_list = []
        for item in casual_outerwear_items:
            if year == min_year:
                inflation_rate = None
            else:
                try:
                    item_cpi = cpi.inflate(BASE_PRICE, year - 1, to=year, area="U.S. city average", items=item)
                    total_cpi_list.append(item_cpi)
                    inflation_rate = (item_cpi - BASE_PRICE) / BASE_PRICE
                except Exception as e:
                    logger.error(f"Error inflating {item} for {year}: {e}")
                    item_cpi = None
                    inflation_rate = None

            temp_inflation_rate[f"{standardize_itemname(item)}_inflation_rate"] = inflation_rate
            print(f"{item} inflation rate for {year}: {inflation_rate}")

        if year == min_year:
            temp_inflation_rate["avg_inflation_rate"] = None
            final_inflation[year] = temp_inflation_rate
            continue
        elif len(total_cpi_list) == 0:
            temp_inflation_rate["avg_inflation_rate"] = None
        else:
            avg_inf_rate = sum(total_cpi_list) / (len(total_cpi_list) * BASE_PRICE) - 1
            temp_inflation_rate["avg_inflation_rate"] = avg_inf_rate

        final_inflation[year] = temp_inflation_rate
    
    final_inflation_df = pd.DataFrame(final_inflation)
    final_inflation_df = final_inflation_df.T

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    final_inflation_df.to_csv(os.path.join(OUTPUT_DIR, 'casual_outerwear_inflation_rate.csv'))
if __name__ == "__main__":
    main()