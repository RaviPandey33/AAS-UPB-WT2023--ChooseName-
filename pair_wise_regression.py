from mlxtend.feature_selection import SequentialFeatureSelector
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
# from pyearth import Earth
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score

class PairWiseRegression:

    def __init__(self):
        self.models = {}
        self.trained_models = {}
        self.selected_features = {}

    def train_pairwise_regression(self, X_train, y_train, model_name, feature_selection):
        
        model_per_solver = {}
        
        # Use the selected features for the current model_name
        selected_cols = self.selected_features.get(model_name, list(X_train.columns))
        X_train_selected = X_train[selected_cols]

        features = len(X_train_selected.columns)
                
        for i, solver in enumerate(y_train.columns):
            for j in range(i + 1, len(y_train.columns)):
                solver1, solver2 = solver, y_train.columns[j]
                pair = tuple(sorted([solver1, solver2]))

                # Ensure the pair is unique
                if pair in model_per_solver:
                    continue

                # Select data for the current pair
                y_pair = y_train[[solver1, solver2]].values

                 

                # Train the model
                if feature_selection in ['sffs', 'sbfs'] and model_name == 'random_forest':
                    model = RandomForestRegressor(n_estimators=500, max_features=features // 3)
                
                else:
                    model = self.models[model_name]

                model.fit(X_train_selected, y_pair)
                model_per_solver[pair] = model

        self.trained_models[model_name] = model_per_solver
        # print(self.trained_models)
    # ... (rest of the methods remain the same)


    def predict(self, X_test, y_test):
        predictions = {}
        best_models = {}

        for model_name, model_per_solver in self.trained_models.items():
            model_predictions = {}
            model_best = {}
            
            # Use the selected features for the current model_name
            selected_cols = self.selected_features.get(model_name, list(X_test.columns))
            X_test_selected = X_test[selected_cols]
            
            for pair, model in model_per_solver.items():
                solver1, solver2 = pair  # Unpack the tuple to get individual solvers

                # Extract the corresponding columns from y_test
                y_test_pair = y_test[[solver1, solver2]]

                # Predict on the test set
                y_pred = model.predict(X_test_selected)
                # Evaluate the model
                mse = mean_squared_error(y_test_pair, y_pred)

                # Store predictions correctly
                model_predictions[pair] = {'MSE': mse, 'Predictions': y_pred}

                model_best[pair] = mse

            # Identify the best model for each pair based on MSE
            if model_best:
                best_model_pair = min(model_best, key=model_best.get)
                best_models[model_name] = best_model_pair
                predictions[model_name] = model_predictions
            
        return predictions, best_models

    def perform_pairwise_regression(self, normalized=False,feature_selection = ""):
        # The function assumes that the feature files are already generated
        
        
        
        if normalized:
            x_file = 'n_median_features.csv'
        else:
            x_file = 'median_features.csv'

        X = pd.read_csv(x_file)

        epsilon = 1e-9
        X = X.replace([np.nan, -np.inf], epsilon)

        
        # write feature selection conditions here
        features = len(X.columns)
        #do max features based on feature number
        
        
        #we have disabled kernel_svm because it doesn't work with pairs and requires only single y values like in regular regression
        self.models = {
            # 'kernel_svm': SVR(),
            'rpart': DecisionTreeRegressor(),
            'xgboost': XGBRegressor(),
            'random_forest': RandomForestRegressor(n_estimators=500, max_features = features//3),
            
            # 'mars': Earth(),
        }
        

        model_names = self.models.keys()
        
        file_name = 'rel_ERT.csv'
        df_y = pd.read_csv(file_name)

        df_y = df_y.iloc[:, 3:]

        solvers = df_y.columns

         
        
        
        
        loo = LeaveOneOut()
        
        mse_results = {'Model': [], 'Solver': [], 'MSE': [] }
        all_predictions = []
        
        
        for model_name in self.models.keys():
            # Feature selection
            if feature_selection == 'sffs':
                # Sequential Forward Floating Selection
                if model_name not in self.selected_features:
                    self.selected_features[model_name] = {}

               # Randomly select a solver pair for feature selection
                random_solver_pair = np.random.choice(df_y.columns, 1, replace=False)
                pair = random_solver_pair

                sffs = SequentialFeatureSelector(
                    self.models[model_name],
                    k_features='best',
                    forward=True,
                    floating=True,
                    scoring='neg_mean_squared_error',
                    cv=False
                )
        
                # Perform feature selection once for the current model_name
                sffs.fit(X, df_y[pair])
                selected_cols = list(X.columns[np.array(sffs.k_feature_idx_).astype(int)])


                self.selected_features[model_name] = selected_cols
                print(f"{model_name}, {len(selected_cols)}")
            elif feature_selection == 'sbfs':
                # Sequential Backward Floating Selection
                if model_name not in self.selected_features:
                    self.selected_features[model_name] = {}

                # Randomly select a solver pair for feature selection
                random_solver_pair = np.random.choice(df_y.columns, 1, replace=False)
                pair = random_solver_pair

                sbfs = SequentialFeatureSelector(
                    self.models[model_name],
                    k_features='best',
                    forward=False,
                    floating=True,
                    scoring='neg_mean_squared_error',
                    cv=False
                )

                # Perform feature selection once for the current model_name
                sbfs.fit(X, df_y[pair])
                selected_cols = list(X.columns[np.array(sbfs.k_feature_idx_).astype(int)])


                self.selected_features[model_name] = selected_cols
                print(f"{model_name}, {len(selected_cols)}")
            else:
                # No feature selection
                selected_cols = list(X.columns)

            # Train pairwise regression models with leave-one-out cross-validation
            loo = LeaveOneOut()

            for train_index, test_index in loo.split(X):
                X_train, X_test = X.iloc[train_index], X.iloc[test_index]
                y_train, y_test = df_y.iloc[train_index], df_y.iloc[test_index]

                # Use the selected features for the current model_name
                X_train_selected = X_train[selected_cols]

                self.train_pairwise_regression(X_train_selected, y_train, model_name, feature_selection)


            predictions, best_models = self.predict(X_test, y_test)

            for model_name, model_predictions in predictions.items():
                for pair, metrics in model_predictions.items():
                    solver1, solver2 = pair

                    # Save results
                    mse_results['Model'].append(model_name)
                    mse_results['Solver'].append(pair)
                    mse_results['MSE'].append(metrics['MSE'])

                    # Calculate and save the length of predictions
                    # predictions_length = len(metrics['Predictions'][pair])
                    # mse_results['Predictions_Length'].append(predictions_length)

                    # Ensure consistent lengths by extending the list
                    # all_predictions.extend(metrics['Predictions'][pair])
        # Create a DataFrame from the results
        mse_df = pd.DataFrame(mse_results)
        
        # If all arrays have consistent lengths, the DataFrame creation should work
        # Save MSE and Predictions to a CSV file
        
        # Save MSE, R2, and Predictions to a CSV file
        file_name = f"metrics_results_{feature_selection}_{normalized}_rforest.csv"
        mse_df.to_csv(file_name, index=False)

        # Calculate average MSE, R2 for each model and solver pair
        avg_metrics_df = mse_df.groupby(['Model', 'Solver'])[['MSE']].mean().reset_index()
        # Save average MSE, R2 to a CSV file
        avg_metrics_df.to_csv(f'average_metrics_results_{feature_selection}_{normalized}_rforest.csv', index=False)

        # Identify the best and worst solver
        avg_r2_per_solver = avg_metrics_df.groupby('Solver')['MSE'].mean()
        best_solver = avg_r2_per_solver.idxmax()
        worst_solver = avg_r2_per_solver.idxmin()

        return best_solver, worst_solver

# ... (rest of the code)



pwr = PairWiseRegression()

best_solver, worst_solver = pwr.perform_pairwise_regression()
best_solvern, worst_solvern = pwr.perform_pairwise_regression(normalized = True)


print(f'Best Solver: {best_solver}')
print(f'Worst Solver: {worst_solver}')

print(f'Best Solver: {best_solvern}')
print(f'Worst Solver: {worst_solvern}')


best_solver_sbfs, worst_solver_sbfs = pwr.perform_pairwise_regression(feature_selection= "sbfs")
best_solver_sffs, worst_solver_sffs = pwr.perform_pairwise_regression(feature_selection= "sffs")
print(f'Best Solver: {best_solver_sbfs}')
print(f'Worst Solver: {worst_solver_sbfs}')

print(f'Best Solver: {best_solver_sffs}')
print(f'Worst Solver: {worst_solver_sffs}')




best_solvern_sbfs, worst_solvern_sbfs = pwr.perform_pairwise_regression(normalized = True,feature_selection= "sbfs")
best_solvern_sffs, worst_solvern_sffs = pwr.perform_pairwise_regression(normalized = True,feature_selection= "sffs")

print(f'Best Solver: {best_solvern_sbfs}')
print(f'Worst Solver: {worst_solvern_sbfs}')

print(f'Best Solver: {best_solvern_sffs}')
print(f'Worst Solver: {worst_solvern_sffs}')


# 
