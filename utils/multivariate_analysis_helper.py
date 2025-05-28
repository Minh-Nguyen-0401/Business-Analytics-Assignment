from abc import ABC, abstractmethod
from sklearn.metrics import roc_auc_score, mutual_info_score
from scipy import stats
from scipy.stats import entropy, chi2_contingency
from tabulate import tabulate
from collections import defaultdict

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class MultiVarAnalysisStrategy(ABC):
    @abstractmethod
    def analysis(self, data, **kwargs):
        pass

class SalesPerformanceAnalysis(MultiVarAnalysisStrategy):
    def analysis(self, data, target_col, granularity="monthly", display_cost = False, time_bounds = (None,None)):
        data_copy = data.copy()

        # redefine time bounds
        min_ts, max_ts = time_bounds
        min_ts = data_copy["Date_ts"].min() if min_ts is None else min_ts
        max_ts = data_copy["Date_ts"].max() if max_ts is None else max_ts
        data_copy = data_copy[data_copy["Date_ts"].between(min_ts, max_ts)]

        fig, axes = plt.subplots(nrows = 2, ncols = 1, figsize=(24,12), sharex=True)
        fig.suptitle(f"\n{granularity.capitalize()} Sales Overview", fontsize=24, fontweight="bold", family="Cambria")
        axes = axes.flatten()

        group_dict = {
            "year_monthly" : ["Date_ts"],
            "year_quarterly" : ["Year", "Quarter"],
            "half_yearly" : ["Year", "half_year"],
            "yearly" : ["Year"],
            "ovr_monthly": ["Month"],
            "ovr_quarterly": ["Quarter"]
        }

        granularity = granularity.lower()
        if granularity not in group_dict.keys():
            raise Exception(f"Granularity {granularity} not supported")
        else:
            groupby_cols = group_dict.get(granularity)
            temp_data = data_copy.groupby(groupby_cols).agg(
                total_sales = (target_col, "sum"),
                total_promo = ("Has_Promo", "sum"),
                promo_cost = ("Budget_USD", "sum")
            ).reset_index()
            temp_data["has_promo"] = temp_data["total_promo"].apply(lambda x: 1 if x > 0 else 0)
            temp_data["Budget-to-sales"] = temp_data["promo_cost"] / temp_data["total_sales"]
            
            #append indices into one col
            temp_data["new_index"] = temp_data[groupby_cols].astype(str).agg('-'.join, axis=1)

            sns.lineplot(data=temp_data, x="new_index", y="total_sales", ax=axes[0], color= "green", label="Total Sales")
            
            if display_cost:
                sns.lineplot(data=temp_data, x="new_index", y="promo_cost", ax=axes[0], color= "orange", linestyle="--", label="Promo Cost")
                if len(temp_data) <= 180:
                    for idx, row in temp_data.iterrows():
                        uplift = max(temp_data["total_sales"])*0.04
                        axes[0].text(row["new_index"], row["promo_cost"] + uplift, f"{int(row['promo_cost']):,.0f}", ha="center", va="bottom", color="orange", fontsize=10, fontweight="bold", alpha = 1)
            
            promo = temp_data[temp_data["has_promo"] == 1]
            axes[0].scatter(
                promo["new_index"],
                promo["total_sales"],
                s=30,      
                color="red",
                marker="o",
                label="Promo"
            )

            for idx, row in promo.iterrows():
                uplift = max(promo["total_sales"])*0.04
                axes[0].text(row["new_index"], row["total_sales"] + uplift, int(row["total_promo"]), ha="center", va="bottom", color="red", fontsize=12, fontweight="bold")
            axes[0].legend()

            axes[1].bar(temp_data["new_index"], temp_data["Budget-to-sales"], color= ["green" if x > 0 else "red" for x in temp_data["Budget-to-sales"]], alpha = 0.7, label="ROI")
            import numpy as np

            if len(temp_data) <= 180:
                for i, val in enumerate(temp_data["Budget-to-sales"]):
                    if not np.isfinite(val):
                        continue
                    downlift = 0 if val > 0 else temp_data["Budget-to-sales"].min() * 0.08
                    axes[1].text(i, val + downlift, f"{val:.2f}", ha="center", va="bottom", color="black", fontsize=10, fontweight="bold")

            # axes[1].set_xlabel(f"{granularity.capitalize()}", fontsize=12)
            axes[1].legend()
            
            axes[1].tick_params(axis="x", rotation=45)
            if granularity == "monthly":
                for ax in axes:
                    ax.set_xticks([])

            for ax in axes:
                ax.grid(axis="y", linestyle=":", alpha=0.9)
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            plt.show()

class MAAnalysis(MultiVarAnalysisStrategy):
    def analysis(self, data, time_col, target_col, n=12):
        data_copy = data.copy()
        data_copy["MA"] = data_copy[target_col].transform(lambda x: x.rolling(n).mean())
        fig, ax = plt.subplots(figsize=(20,8))
        sns.lineplot(data=data_copy, x=time_col, y=target_col, ax=ax, color= "green", alpha = 0.7, label=f"{target_col}")
        sns.lineplot(data=data_copy, x=time_col, y="MA", ax=ax, color= "red", alpha = 0.7, label=f"MA({n})")
        ax.legend()
        plt.show()
        
class MultiVarAnalysisProcessor:
    def __init__(self, strategy: MultiVarAnalysisStrategy):
        self._strategy = strategy

    def set_strategy(self, strategy: MultiVarAnalysisStrategy):
        self._strategy = strategy

    def analysis(self, data, **kwargs):
        return self._strategy.analysis(data, **kwargs)