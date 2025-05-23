import pandas as pd
import cpi
import warnings
import os
import yaml
from collections import OrderedDict
import logging

warnings.filterwarnings("ignore")
logging.basicConfig(
    filename="./log/extract_cpi.log",
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    filemode="w"
)
logger = logging.getLogger()

casual_outerwear_items = [
    "Men's shirts and sweaters",
    "Men's pants and shorts",
    "Men's suits, sport coats, and outerwear",
    "Women's outerwear",
    "Women's dresses",
    "Women's suits and separates"
]

BASE_DIR = os.path.dirname(__file__)
SALES_PATH = os.path.join(BASE_DIR, 'original_data', 'Sales_Data.csv')
OUTPUT_DIR = os.path.join(BASE_DIR, 'supplementary_data')
CONFIG_PATH = os.path.join(OUTPUT_DIR, "collect_config.yaml")

BASE_PRICE = 100

def main():
    df = pd.read_csv(SALES_PATH, parse_dates=["Date"])
    df['Year'] = df['Date'].dt.year
    min_year, max_year = int(df['Year'].min()), int(df['Year'].max())

    # Save config
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(CONFIG_PATH, 'w') as f:
        yaml.dump({"min_year": min_year, "max_year": max_year}, f)

    BASE_YEAR = min_year

    records = []
    for year in range(min_year, max_year + 1):
        if year == BASE_YEAR:
            avg_cpi = BASE_PRICE
        else:
            cpis = []
            for item in casual_outerwear_items:
                try:
                    inflated = cpi.inflate(
                        BASE_PRICE,
                        BASE_YEAR,
                        to=year,
                        area="U.S. city average",
                        items=item
                    )
                    cpis.append(inflated)
                except Exception as e:
                    logger.error(f"Error inflating {item} for {year}: {e}")
            avg_cpi = sum(cpis) / len(cpis) if cpis else None

        records.append({"year": year, "avg_cpi": avg_cpi})
        print(f"{year}: avg_cpi = {avg_cpi}")

    # Write output CSV
    out_df = pd.DataFrame(records)
    out_path = os.path.join(OUTPUT_DIR, 'casual_outerwear_avg_cpi.csv')
    out_df.to_csv(out_path, index=False)
    print(f"Saved average CPI series to {out_path}")

if __name__ == "__main__":
    main()
