# Sales Data Analysis Assignment 1
An automated pipeline for sales forecasting and analysis.

## Repository Structure
```
.
├── .env                     # environment variables
├── .gitignore               # git ignore rules
├── docs/                    # documentation
├── log/                     # log files
├── models/                  # trained model artifacts
├── original_data/           # original datasets
├── supplementary_data/      # additional data sources
├── collect_macro_indicators.py  # macro data collection script
├── extract_inflation_rate.py    # inflation rate extraction script
├── main.py                  # main orchestrator script
├── main_model.ipynb         # exploratory notebook
├── test.ipynb               # test notebook
├── test_module.ipynb        # module tests notebook
├── src/                     # core modules
│   ├── ingest.py            # data ingestion
│   ├── preprocess.py        # data preprocessing
│   ├── aggregate.py         # feature aggregation
│   ├── hypertune.py         # hyperparameter tuning
│   ├── train.py             # model training
│   └── search_space.yaml    # tuning configurations
└── utils/                   # helper utilities