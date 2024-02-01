import pandas as pd
import numpy as np
from sklearn.model_selection import LeaveOneOut, KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import ast
from xgboost import XGBRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

class RegressionAnalysis:
    def __init__(self, ela_data_path, ert_data_path, target_columns):
        self.ela_data = pd.read_csv(ela_data_path)
        self.ert_data = pd.read_csv(ert_data_path)
        self.target_columns = target_columns
        self.merged_data = None
        self.rf_model = RandomForestRegressor(n_estimators=500, random_state=42)
        self.svm_model = SVR(kernel='rbf')
        self.model_xgboost = XGBRegressor(objective='rank:pairwise', random_state=42)
        self.dt_model = DecisionTreeRegressor(random_state=42)
        self.n_splits = 5  # Change this value based on your preference
        self.loo = LeaveOneOut()
        self.mse_dict = {}
        self.result_data = pd.DataFrame(
            columns=['mbse for RandomForest', 'mbse for xgboost', 'mbse for SVM', 'mbse for RPart', 'Algorithm']
        )

    def safe_eval(self, x):
        try:
            return ast.literal_eval(x)
        except (SyntaxError, ValueError):
            return None

    def preprocess_data(self):
        self.merged_data = pd.merge(self.ela_data, self.ert_data, on=['fid', 'dim'])
        self.merged_data['features'] = self.merged_data['features'].apply(self.safe_eval)
        self.merged_data = self.merged_data.replace([np.inf, -np.inf], np.nan)
        self.merged_data = self.merged_data.dropna(subset=['features'])
        max_len = self.merged_data['features'].apply(len).max()
        # self.merged_data['features'] = self.merged_data['features'].apply(
        #     lambda x: x + [0] * (max_len - len(x)) if x is not None else [0] * max_len
        # )

    def train_and_evaluate_models(self):
        for target_column in self.target_columns:
            self.mse_dict[target_column] = {}
            X = np.array(self.merged_data['features'].tolist())
            y = np.array(self.merged_data[target_column].tolist())

            mse_list = []
            mse_xgboost_list = []
            mse_svm_list = []
            mse_dt_list = []
            mse_mars_list = []

            for train_index, test_index in self.loo.split(self.merged_data):
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]

                self.rf_model.fit(X_train, y_train)
                y_pred = self.rf_model.predict(X_test)
                mse = mean_squared_error(y_test, y_pred)
                mse_list.append(mse)

                self.model_xgboost.fit(X_train, y_train)
                y_pred_xgboost = self.model_xgboost.predict(X_test)
                mbse_xgboost = mean_squared_error(y_test, y_pred_xgboost)
                mse_xgboost_list.append(mbse_xgboost)

                self.svm_model.fit(X_train, y_train)
                y_pred_svm = self.svm_model.predict(X_test)
                mbse_svm = mean_squared_error(y_test, y_pred_svm)
                mse_svm_list.append(mbse_svm)

                self.dt_model.fit(X_train, y_train)
                y_pred_dt = self.dt_model.predict(X_test)
                mbse_dt = mean_squared_error(y_test, y_pred_dt)
                mse_dt_list.append(mbse_dt)

            avg_mse = np.mean(mse_list)
            avg_mse_xgboost = np.mean(mse_xgboost_list)
            avg_mse_svm = np.mean(mse_svm_list)
            avg_mse_dt = np.mean(mse_dt_list)

            self.result_data = pd.concat(
                [self.result_data, pd.DataFrame({
                    'mbse for RandomForest': [avg_mse],
                    'mbse for xgboost': [avg_mse_xgboost],
                    'mbse for SVM': [avg_mse_svm],
                    'mbse for RPart': [avg_mse_dt],
                    'Algorithm': [target_column]
                })], ignore_index=True
            )

    def export_results(self, output_path="mbse_data_regression.csv"):
        self.result_data.to_csv(output_path, encoding='utf-8', index=False)


# Example usage:
if __name__ == "__main__":
    ela_data_path = 'Processed_Median_Features.csv'
    ert_data_path = 'rel_ERT.csv'
    target_columns = ["BSqi", "BSrr", "CMA-CSA", "fmincon", "fminunc", "HCMA",
                      "HMLSL", "IPOP400D", "MCS", "MLSL", "OQNLP", "SMAC-BBOB"]

    regression_analysis = RegressionAnalysis(ela_data_path, ert_data_path, target_columns)
    regression_analysis.preprocess_data()
    regression_analysis.train_and_evaluate_models()
    regression_analysis.export_results()
