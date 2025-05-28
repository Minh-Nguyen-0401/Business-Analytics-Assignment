from abc import ABC, abstractmethod
from sklearn.metrics import roc_auc_score, mutual_info_score
from scipy import stats
from scipy.stats import entropy, chi2_contingency
from tabulate import tabulate
from collections import defaultdict
import pandas as pd
import numpy as np


class UniAnalysisStrategy(ABC):
    @abstractmethod
    def analysis(self, data, column_name):
        pass

class DataOverviewAnalysis(UniAnalysisStrategy):
    def analysis(self, data, column_list: list = None):
        """
            Phân tích tống quan dữ liệu
            - data: pd.DataFrame, dữ liệu cần phân tích.
        """
        if column_list is None:
            column_list = data.columns
            
        d_dict = defaultdict(tuple)

        for col in column_list:
            if col not in data.columns:
                print(f"Column {col} not exist in df given - skipping")
                continue
            else:
                dtype = data[col].dtypes
                valid_instances = data[col].shape[0]
                unique = data[col].nunique()
                null_count = data[col].isnull().sum()
                null_pct = (lambda x,y: f"{round((x/y)*100,2)}%")(null_count, valid_instances)
                uni_value = data[col].unique().tolist()
                d_dict[col] = dtype,valid_instances,unique,null_count, null_pct, uni_value

        data_check = pd.DataFrame(d_dict, index=["dtype","valid_instances","unique","total_null","null_pct","duplicates"]).T
        print(tabulate(data_check, headers='keys', tablefmt='fancy_grid'))

class NullValueAnalysis(UniAnalysisStrategy):
    def analysis(self, data, column_name=None):
        """
            Phát hiện null value trong dữ liệu
            - data: pd.DataFrame, dữ liệu cần phân tích.
            - column_name: str, tên cột cần phân tích.
        """
        null_counts = data[column_name].isnull().sum()
        null_percentage = (null_counts / len(data[column_name])) * 100
        null_detection_summary_df = pd.DataFrame({
                'Number of Null values': [null_counts],
                'Percentage (%)': [null_percentage]
            })
        null_detection_summary_df.index = ['Summary']
        print(f"Null Detection Summary for {column_name}:")
        print(tabulate(null_detection_summary_df, headers='keys', tablefmt='fancy_grid'))
        return null_counts, null_percentage


class OutlierAnalysis(UniAnalysisStrategy):
    def analysis(self, data, column_name=None, method='iqr', z_threshold=3, display_asc=False):
        """
            Phát hiện outliers trong một cột của DataFrame.
            - data: pd.DataFrame, dữ liệu cần phân tích.
            - column_name: str, Tên cột cần phân tích.
            - method: Phương pháp phát hiện outliers, 'iqr' (Interquartile Range) hoặc 'zscore'.
            - z_threshold: Ngưỡng Z-score (chỉ dùng khi method='zscore'). Mặc định là 3.
        """
        df_cleaned = data.copy()
        if method == 'iqr':
            Q1 = df_cleaned[column_name].quantile(0.25)
            Q3 = df_cleaned[column_name].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = df_cleaned[(df_cleaned[column_name] < lower_bound) | (df_cleaned[column_name] > upper_bound)]
        
        elif method == 'zscore':
            mean = df_cleaned[column_name].mean()
            std_dev = df_cleaned[column_name].std()
            z_scores = (df_cleaned[column_name] - mean) / std_dev
            
            outliers = df_cleaned[np.abs(z_scores) > z_threshold]
        
        else:
            raise ValueError("Invalid method. Please choose 'iqr' or 'zscore'.")
        
        # In kết quả
        num_outliers = outliers.shape[0]
        outlier_ratio = num_outliers / df_cleaned.shape[0] * 100
        outlier_detection_summary_df = pd.DataFrame({
                'Number of Outliers': [num_outliers],
                'Percentage (%)': [outlier_ratio]
            })
        outlier_detection_summary_df.index = ['Summary']
        print(f"Outlier Detection Summary for {column_name}:")
        print(tabulate(outlier_detection_summary_df, headers='keys', tablefmt='fancy_grid'))

        if display_asc:
            outliers = outliers.sort_values(by=column_name, ascending=True)
        else:
            outliers = outliers.sort_values(by=column_name, ascending=False)
        print(tabulate(outliers.head(10), headers='keys', tablefmt='fancy_grid'))
        return outliers, num_outliers, outlier_ratio
    
class QuantitativeAnalysis(UniAnalysisStrategy):
    def analysis(self, data, column_name=None):
        """
            Phân tích dữ liệu cho biến định lượng.
            - data: pd.DataFrame, dữ liệu cần phân tích.
            - column_name: str, tên cột cần phân tích.
        """
        if column_name in data.columns:
            # Thống kê cơ bản
            summary = data[column_name].describe()
            summary_df = pd.DataFrame(summary).T
            summary_df.index = ['Summary']
            # Tính độ lệch và độ nhọn
            summary_df['Skewness'] = data[column_name].skew()
            summary_df['Kurtosis'] = data[column_name].kurtosis()
            # Kiểm tra tính phân phối chuẩn (normally distributed)
            stat, p_value = stats.shapiro(data[column_name])
            normality_test = {'Shapiro-Wilk Statistic': stat, 'p-value': p_value}
            normality_df = pd.DataFrame([normality_test])
            normality_result = "Normally distributed" if p_value > 0.05 else "Not normally distributed"
            normality_df['Result'] = normality_result
            # In kết quả
            print("Summary Statistics:")
            print(tabulate(summary_df, headers='keys', tablefmt='fancy_grid'))
            print("\nNormality Test (Shapiro-Wilk):")
            print(tabulate(normality_df, headers='keys', tablefmt='fancy_grid'))
        else:
            print(f"Error: '{column_name}' column not found in data.")

class CategoricalAnalysis(UniAnalysisStrategy):
    def analysis(self, data, column_name):
        """
            Phân tích dữ liệu cho biến phân loại.
            - data: pd.DataFrame, dữ liệu cần phân tích.
            - column_name: str, tên cột cần phân tích.
        """
        if column_name in data.columns:
            # Tính tần suất và tính phần trăm của các giá trị
            num_values = data[column_name].nunique()
            frequency = data[column_name].value_counts()
            percentage = data[column_name].value_counts(normalize=True) * 100
            summary_table = pd.DataFrame({
                'Frequency': frequency,
                'Percentage (%)': percentage
            })
            # In kết quả
            print(f"Number of unique values: {num_values}. Summary Statistics:")
            print(tabulate(summary_table, headers='keys', tablefmt='fancy_grid'))
        else:
            print(f"Error: '{column_name}' column not found in data.")

# Context class for Strategy
class UniAnalysisProcessor:
    def __init__(self, strategy: UniAnalysisStrategy):
        self._strategy = strategy

    def set_strategy(self, strategy: UniAnalysisStrategy):
        self._strategy = strategy

    def analysis(self, data, **kwargs):
        return self._strategy.analysis(data, **kwargs)