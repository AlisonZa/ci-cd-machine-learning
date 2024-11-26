import os, sys
from src.utils import logger_obj,CustomException
from src.entities import DataPreprocessingArtifacts, ModelTrainingArtifacts, ModelTrainingParams
import pandas as pd
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import os
import pandas as pd
import numpy as np

# Modelling
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor
from sklearn.linear_model import LinearRegression, Ridge,Lasso
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
import joblib


class ModelTraining:
    def __init__(self):
        self.model_preprocessing_artifacts= DataPreprocessingArtifacts()
        self.model_training_artifacts = ModelTrainingArtifacts()
        self.model_training_params = ModelTrainingParams()

        try:
            logger_obj.info(f"Creating the model_training folder")
            os.makedirs(self.model_training_artifacts.model_training_root_folder, exist_ok= True)
            logger_obj.info(f"Succesfully created the model_training folder at: \n{self.model_training_artifacts.model_training_root_folder}")
        
        except Exception as e:
            logger_obj.error(f"Error during creating the model_training folder, Error:\n{CustomException(e ,sys)}")
            raise CustomException(e ,sys)


    def get_scorer(self):
        """Define scoring metrics for GridSearchCV."""
        scoring_dict = {
            'r2_score': 'r2',
            'mae': 'neg_mean_absolute_error',
            'rmse': 'neg_root_mean_squared_error',
            'mse': 'neg_mean_squared_error'
        }
        scoring_criteria = self.model_training_params.main_scoring_criteria

        if scoring_criteria not in scoring_dict:
            raise ValueError("scoring_criteria must be one of ['r2_score', 'mae', 'rmse', 'mse']")
        return scoring_dict[scoring_criteria]

    def get_metric_func(self):
        """Get the metric function for final evaluation."""
        metrics_dict = {
            'r2_score': r2_score,
            'mae': mean_absolute_error,
            'rmse': lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred)),
            'mse': mean_squared_error
        }

        scoring_criteria = self.model_training_params.main_scoring_criteria

        return metrics_dict[scoring_criteria]

    def train_and_evaluate_models_cv_regression(self, 
                                  models_with_hyperparameters = [
                                    (LinearRegression(), {'fit_intercept': [True, False]}),                                    
                                    (Lasso(random_state=42), {'alpha': [0.1, 1, 10], 'fit_intercept': [True, False]}),                                    
                                    (Ridge(random_state=42), {'alpha': [0.1, 1, 10], 'fit_intercept': [True, False]}),                                    
                                    (KNeighborsRegressor(), {'n_neighbors': [3, 5, 7, 10], 'weights': ['uniform', 'distance'], 'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']}),                                    
                                    (DecisionTreeRegressor(random_state=42), {'max_depth': [None, 5, 10], 'min_samples_split': [2, 5, 10], 'min_samples_leaf': [1, 2, 5]}),                                    
                                    (RandomForestRegressor(random_state=42), {'n_estimators': [50, 100], 'max_depth': [None, 10], 'min_samples_split': [2, 5, 10], 'min_samples_leaf': [1, 2, 5]}),                                    
                                    (XGBRegressor(random_state=42), {'n_estimators': [50, 100], 'learning_rate': [0.01, 0.1], 'max_depth': [3, 5, 7], 'subsample': [0.8, 1.0]}),                                    
                                    (CatBoostRegressor(random_state=42, verbose=False), {'iterations': [500, 1000], 'depth': [3, 5, 7], 'learning_rate': [0.01, 0.1], 'l2_leaf_reg': [1, 3, 5]}),                                    
                                    (AdaBoostRegressor(random_state=42), {'n_estimators': [50, 100], 'learning_rate': [0.01, 0.1, 0.5]})]
                                  ):
        """
        Evaluate multiple models using k-fold cross-validation with pre-transformed data.
        
        Parameters:
        -----------
        """
        # Load the data
        X_train_transformed = np.load(self.model_preprocessing_artifacts.X_train_transformed_path)
        X_test_transformed = np.load(self.model_preprocessing_artifacts.X_test_transformed_path)
        y_train = np.load(self.model_preprocessing_artifacts.y_train_path)
        y_test = np.load(self.model_preprocessing_artifacts.y_test_path)
        
        k= self.model_training_params.number_of_folds_kfold
        
        scoring_criteria=self.model_training_params.main_scoring_criteria 
        random_state=42

            # Initialize variables
        best_models = []
        results = []
        
        # Set up cross-validation
        kf = KFold(n_splits=k, shuffle=True, random_state=random_state)
        
        # Get appropriate scorer for GridSearchCV
        scorer = self.get_scorer()
        metric_func = self.get_metric_func()
        
        for model, hyperparameters in models_with_hyperparameters:
            print(f"Evaluating {model.__class__.__name__}...")
            
            # Add random_state if applicable
            if hasattr(model, 'random_state'):
                hyperparameters['random_state'] = [random_state]
            
            # Perform Grid Search with cross-validation
            grid_search = GridSearchCV(
                estimator=model,
                param_grid=hyperparameters,
                cv=kf,
                n_jobs=-1,
                scoring=scorer,
                error_score='raise'
            )
            
            # Fit the grid search
            grid_search.fit(X_train_transformed, y_train)
            
            # Get best model
            best_model = grid_search.best_estimator_
            
            # Make predictions
            y_pred = best_model.predict(X_test_transformed)
            
            # Calculate metrics
            performance = {
                'Model': model.__class__.__name__,
                'Best Hyperparameters': grid_search.best_params_,
                'R2 Score': r2_score(y_test, y_pred),
                'MAE': mean_absolute_error(y_test, y_pred),
                'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
                'MSE': mean_squared_error(y_test, y_pred),
                'CV Score': -grid_search.best_score_ if 'neg_' in scorer else grid_search.best_score_
            }
            
            results.append(performance)
            best_models.append(best_model)
        
        # Create results DataFrame
        results_df = pd.DataFrame(results)
        
        # Sort results based on scoring criteria
        if scoring_criteria == 'r2_score':
            results_df = results_df.sort_values(by='R2 Score', ascending=False)
        else:
            metric_name = scoring_criteria.upper()
            results_df = results_df.sort_values(by=metric_name, ascending=True)
        
        # Get best overall model
        best_model_idx = results_df.index[0]
        best_model_overall = best_models[best_model_idx]
        
        print("Evaluation Complete.")
         
        self.model_training_artifacts.best_models= best_models 
        self.model_training_artifacts.results = results_df 
        self.model_training_artifacts.best_model_overall= best_model_overall


    def save_results(self):
        results_dataframe = self.model_training_artifacts.results
        path_to_save = os.path.join(self.model_training_artifacts.model_training_root_folder, "results.xlsx")
        results_dataframe.to_excel(path_to_save)


    def save_best_models(self):
        for model in self.model_training_artifacts.best_models:
            os.makedirs(self.model_training_artifacts.best_models_folder, exist_ok= True)

            filename = f"{model.__class__.__name__}.joblib"

            model_path = os.path.join(self.model_training_artifacts.best_models_folder, filename)
            joblib.dump(model, model_path)


    def save_best_model_overall(self):      
        model = self.model_training_artifacts.best_model_overall
        
        joblib.dump(model, self.model_training_artifacts.best_model_overall_path)


    def run_model_training_and_evaluation(self):
        try:
            self.train_and_evaluate_models_cv_regression()
            self.save_results()
            self.save_best_models()
            self.save_best_model_overall()
  
        except Exception as e:
            logger_obj.error(f"Error during run model training and evaluation :\n{CustomException(e ,sys)}")
            raise CustomException(e ,sys)







