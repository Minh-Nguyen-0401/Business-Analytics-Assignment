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

LOG_DIR = os.path.join(os.path.dirname(__file__), 'log')
os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(filename=f"{LOG_DIR}/ingest.log", level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", filemode = "w")
logger = logging.getLogger()

class Ingestion:
    def __init__(self):
        self.config = self.load_config()
        self.original_path = self.config['ingest']['original_path']
        self.supp_path = self.config['ingest']['supp_path']
        self.data_dict = defaultdict()

    def load_config(self):
        with open(os.path.join(CURRENT_DIR, 'config.yaml'), 'r') as f:
            config = yaml.safe_load(f)
        return config
        
    def load_data(self):
        data_dict = defaultdict()
        # Load original data
        org_pathfiles_list = Path(self.original_path).rglob('*.csv')
        org_pathfiles_list = [str(path) for path in org_pathfiles_list]
        file_names = [os.path.basename(path).split('.')[0].lower() for path in org_pathfiles_list]
        for name, path in zip(file_names, org_pathfiles_list):
            data_dict[name] = pd.read_csv(path)
        
        # Load supp data
        supp_pathfiles_list = Path(self.supp_path).rglob('*.csv')
        supp_pathfiles_list = [str(path) for path in supp_pathfiles_list]
        file_names = [os.path.basename(path).split('.')[0].lower() for path in supp_pathfiles_list]
        for name, path in zip(file_names, supp_pathfiles_list):
            data_dict[name] = pd.read_csv(path,index_col=0)
        
        self.data_dict = data_dict
        return data_dict

        


        
        