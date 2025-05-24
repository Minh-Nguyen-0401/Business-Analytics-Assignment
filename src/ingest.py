# src/ingest.py

import yaml
import pandas as pd
from pathlib import Path
from collections import defaultdict
import logging
import warnings
warnings.filterwarnings("ignore")

CURRENT_DIR = Path(__file__).resolve().parent
import sys
sys.path.insert(0, str(CURRENT_DIR.parent))
from utils.helper_func import *

LOG_DIR = CURRENT_DIR / 'log'
LOG_DIR.mkdir(exist_ok=True)
logging.basicConfig(
    filename=str(LOG_DIR / 'ingest.log'),
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    filemode="w"
)
logger = logging.getLogger()

class Ingestion:
    def __init__(self):
        self.config = self.load_config()
        self.original_path = Path(self.config['ingest']['original_path'])
        self.supp_path = Path(self.config['ingest']['supp_path'])
        self.data_dict = defaultdict(pd.DataFrame)

    def load_config(self):
        with open(CURRENT_DIR / 'config.yaml') as f:
            return yaml.safe_load(f)

    def load_data(self):
        data_dict = defaultdict(pd.DataFrame)

        # Load org data
        for csv_file in self.original_path.rglob('*.csv'):
            key = csv_file.stem.lower()
            data_dict[key] = pd.read_csv(csv_file)

        # Load supp data
        all_supp = list(self.supp_path.rglob('*.csv'))
        skip = [p for p in all_supp if p.match('*casual_outerwear_inflation_rate.csv')]
        supp_files = [p for p in all_supp if p not in skip]
        logger.info(f"Skipping files: {[s.name for s in skip]}")

        for csv_file in supp_files:
            key = csv_file.stem.lower()
            data_dict[key] = pd.read_csv(csv_file, index_col=0)

        self.data_dict = data_dict
        return data_dict
