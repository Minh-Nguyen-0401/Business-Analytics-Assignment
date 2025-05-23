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
    # ing = Ingestion()
    # data_dict = ing.load_data()
    # agg = Aggregation(data_dict)

    # agg_data = agg.merge_data()

    # org_agg = agg.engineer_feats(agg_data[agg.org_cols])
    # org_agg_cols = org_agg.columns.tolist()

    # ext_agg = agg.engineer_feats(agg_data).drop(org_agg_cols, axis=1).merge(org_agg["New_Sales"], how="left", left_index=True, right_index=True)

    # X_org_train, y_train, X_org_test, y_test = agg.split_train_test(org_agg, target_feat="New_Sales")
    # X_ext_train, y_train, X_ext_test, y_test = agg.split_train_test(ext_agg, target_feat="New_Sales")

    # # Further Preprocessing
    # pre_processor_org = CustomPreprocessor()
    # pre_processor_org.fit(X_org_train, y_train, mi_pct = 0.0)

    # X_org_train = pre_processor_org.transform(X_org_train)
    # X_org_test = pre_processor_org.transform(X_org_test)

    # pre_processor_ext = CustomPreprocessor()
    # pre_processor_ext.fit(X_ext_train, y_train, mi_pct = 0.7)
    # X_ext_train = pre_processor_ext.transform(X_ext_train)
    # X_ext_test = pre_processor_ext.transform(X_ext_test)

    # X_train = pd.concat([X_org_train, X_ext_train], axis=1)
    # X_test = pd.concat([X_org_test, X_ext_test], axis=1)

    # X_train.to_csv("./temp_test_data/X_train.csv", index=True)
    # X_test.to_csv("./temp_test_data/X_test.csv", index=True)
    # y_train.to_csv("./temp_test_data/y_train.csv", index=True)
    # y_test.to_csv("./temp_test_data/y_test.csv", index=True)

    X_train = pd.read_csv("./temp_test_data/X_train.csv", index_col=0)
    X_test = pd.read_csv("./temp_test_data/X_test.csv", index_col=0)
    y_train = pd.read_csv("./temp_test_data/y_train.csv", index_col=0)["New_Sales"]
    y_test = pd.read_csv("./temp_test_data/y_test.csv", index_col=0)["New_Sales"]

    # # LR & Boosted Models
    # feats_to_keep = {
    #     "linear_regression" : 30,
    #     "lightgbm" : 100,
    #     "catboost" : 100
    # }
    # for model in ["linear_regression", "lightgbm", "catboost"]:
    #     hypertuner = HyperTuner(model, feats_to_keep[model], 5)
    #     feats = hypertuner.run_permutation_importance(X_train, y_train)
    #     # keep all promo_type features
    #     promo_type_feat = [col for col in X_train.columns if bool(re.match(r"(?i)(promo_type|months_since_last_promo).*",col))]
    #     feats.extend(promo_type_feat)
    #     print(f"Extended obligatory features: {', '.join(promo_type_feat)}")
    #     feats = list(set(feats))
    #     hypertuner.tune(n_trials=60)
    
    #     trainer = ModelTrainer(model)
    #     trainer.fit(X_train[feats], y_train)
    #     preds = trainer.predict(X_test[feats])
    #     trainer.evaluate(y_test, preds)
    #     trainer.get_feature_importance(X_train=X_train[feats], save=True)
    #     trainer.generate_shap_plot(X_train=X_train[feats], y_train=y_train, save=True)

    # ANN Models
    hypertuner = HyperTuner("lstm", 120, 5)
    feats = hypertuner.run_permutation_importance(X_train, y_train)
    promo_type_feat = [col for col in X_train.columns if bool(re.match(r"(?i)(promo_type|months_since_last_promo).*",col))]
    feats.extend(promo_type_feat)
    print(f"Extended obligatory features: {', '.join(promo_type_feat)}")
    feats = list(set(feats))

    for model in ["rnn", "lstm"]:
        trainer = ModelTrainer(model)
        trainer.fit(X_train[feats], y_train)
        preds = trainer.predict(X_test[feats])
        trainer.evaluate(y_test, preds)
        trainer.generate_shap_plot(X_train=X_train[feats], y_train=y_train, save=True)
        # trainer.get_feature_importance(X_train=X_train[feats], save=True)

if __name__ == "__main__":
    main()