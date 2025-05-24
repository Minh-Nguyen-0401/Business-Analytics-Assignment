# Sales Data Analysis and Forecasting

An advanced machine learning pipeline for sales forecasting and analysis with multiple model options, hyperparameter tuning, and flexible execution.

## Project Overview

This project implements a comprehensive sales forecasting system using both traditional machine learning and neural network approaches. The pipeline includes data ingestion, preprocessing, feature engineering, model training, hyperparameter optimization, and evaluation.

## Key Features

- **Multiple Model Support**: Linear Regression, LightGBM, CatBoost, LSTM, and RNN
- **Consistent Target Variable Scaling**: Using MinMaxScaler to handle increasing sales trends
- **Hyperparameter Optimization**: Using Optuna for systematic parameter tuning
- **Feature Importance Analysis**: Permutation importance for feature selection
- **Time Series Cross-Validation**: To ensure robust model evaluation
- **Separate Preprocessing Flows**: Different approaches for traditional vs. neural network models
- **Command-Line Interface**: For targeted model training and evaluation
- **Visualization**: Model performance metrics and SHAP value plots

## Models

### Traditional Models
- **Linear Regression**: Simple baseline model
- **LightGBM**: Gradient boosting framework optimized for efficiency and performance. Especially useful for comprehensive inclusion of supplementary (macroeconomic) features, as boosted models support both efficiency and explainability.
- **CatBoost**: Gradient boosting with better handling of categorical features and strong support for supplementary features. These boosted models are ideal for extracting insights from macroeconomic variables and understanding their impact on sales.

**Note:** Boosted models (LightGBM, CatBoost) are particularly leveraged to include a wide range of supplementary and macroeconomic features, providing both high predictive power and interpretability. After training, the main trees of these models are fully plotted to facilitate easier navigation and deeper insights into feature importance and decision paths.

### Neural Network Models
- **LSTM**: Long Short-Term Memory networks for capturing temporal dependencies
- **RNN**: Recurrent Neural Networks for sequence modeling

## Installation

```bash
# Clone the repository
git clone https://github.com/Minh-Nguyen-0401/Business-Analytics-Assignment.git
cd Business-Analytics-Assignment

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Command-Line Interface

The main script supports command-line arguments to run specific models:

```bash
# Run all models
python main.py

# Run a specific model with custom trial count
python main.py --model lightgbm --trials 100

# Available model options
python main.py --model [linear_regression|lightgbm|catboost|lstm|rnn]
```

### Model Evaluation

After training, model evaluation results are stored in the `models` directory:

```
models/
├── model_name_model.pkl            # Saved model
└── model_name_evaluation/          # Evaluation artifacts
    ├── metrics.json                # Performance metrics
    ├── actual_vs_predicted.png     # Visualization
    └── model_name_shap.png         # Feature importance
```

## Repository Structure

```
.
├── .env                     # Environment variables
├── .gitignore               # Git ignore rules
├── docs/                    # Documentation
├── log/                     # Log files
├── models/                  # Trained model artifacts
├── original_data/           # Original datasets
├── supplementary_data/      # Additional data sources
├── collect_macro_indicators.py  # Macro data collection script
├── extract_inflation_rate.py    # Inflation rate extraction script
├── main.py                  # Main orchestrator script
├── main_model.ipynb         # Exploratory notebook
├── test.ipynb               # Test notebook
├── test_module.ipynb        # Module tests notebook
├── src/                     # Core modules
│   ├── ingest.py            # Data ingestion
│   ├── preprocess.py        # Data preprocessing
│   ├── aggregate.py         # Feature aggregation
│   ├── hypertune.py         # Hyperparameter tuning
│   ├── train.py             # Model training
│   ├── hypertune_results.yaml  # Tuning results
│   └── search_space.yaml    # Tuning configurations
└── utils/                   # Helper utilities
```

## Implementation Details

### Data Preprocessing
- The pipeline applies feature engineering to create lagged features, rolling statistics, and domain-specific indicators
- Different preprocessing flows for traditional vs neural models to address their different feature sensitivities

### Hyperparameter Tuning
- Uses Optuna for efficient hyperparameter optimization
- Custom search spaces defined in `search_space.yaml`
- Time series cross-validation to prevent data leakage

### Target Variable Scaling
- Consistent MinMaxScaler application across all models
- Properly handles the increasing trend in sales data

## Future Improvements

- Ensemble methods combining multiple models
- More advanced feature engineering
- Automated retraining schedule
- Model explainability enhancements
- Online learning capabilities