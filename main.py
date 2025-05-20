from src.ingest import Ingestion
from src.aggregate import Aggregation   
from src.preprocess import FeatureSelector, CustomPreprocessor
from src.hypertune import HyperTuner
from src.train import ModelTrainer
from utils.helper_func import *
from utils.feature_generation import *
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

import warnings
warnings.filterwarnings("ignore")

def main():
    ing = Ingestion()
    data_dict = ing.load_data()
    agg = Aggregation(data_dict)

    agg_data = agg.merge_data()

    org_agg = agg.engineer_feats(agg_data[agg.org_cols])
    org_agg_cols = org_agg.columns.tolist()

    ext_agg = agg.engineer_feats(agg_data).drop(org_agg_cols, axis=1).merge(org_agg["New_Sales"], how="left", left_index=True, right_index=True)

    X_org_train, y_train, X_org_test, y_test = agg.split_train_test(org_agg, target_feat="New_Sales")
    X_ext_train, y_train, X_ext_test, y_test = agg.split_train_test(ext_agg, target_feat="New_Sales")

    # Further Preprocessing
    pre_processor_org = CustomPreprocessor()
    pre_processor_org.fit(X_org_train, y_train, mi_pct = 0.0)

    X_org_train = pre_processor_org.transform(X_org_train)
    X_org_test = pre_processor_org.transform(X_org_test)

    pre_processor_ext = CustomPreprocessor()
    pre_processor_ext.fit(X_ext_train, y_train, mi_pct = 0.6)
    X_ext_train = pre_processor_ext.transform(X_ext_train)
    X_ext_test = pre_processor_ext.transform(X_ext_test)

    X_train = pd.concat([X_org_train, X_ext_train], axis=1)
    X_test = pd.concat([X_org_test, X_ext_test], axis=1)

    # Hypertunining
    hypertuner = HyperTuner('lightgbm', 100, 5)
    feats = hypertuner.run_permutation_importance(X_train, y_train)
    # keep all promo_type features
    # promo_type_feat = [col for col in X_train.columns if bool(re.match(r"(?i)(promo_type|days_since_last_promo).*",col))]
    # feats.extend(promo_type_feat)
    # print(f"Extended obligatory features: {', '.join(promo_type_feat)}")
    # feats = list(set(feats))
    params = hypertuner.tune(n_trials=100)

    # Model Training
    trainer = ModelTrainer('lightgbm')
    trainer.fit(X_train[feats], y_train)
    preds = trainer.predict(X_test[feats])
    trainer.evaluate(y_test, preds, X_test=X_test[feats])
    trainer.get_feature_importance(save=True)

if __name__ == "__main__":
    main()