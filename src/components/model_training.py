import os
import sys
import pandas as pd
import numpy as np
import joblib
import warnings
from dotenv import load_dotenv
from datetime import datetime

import mlflow
import mlflow.sklearn

from src.utils import logger_obj, CustomException, e_mail_obj
from configurations import pipeline_config_obj

from src.entities import DataPreprocessingArtifacts, ModelTrainingArtifacts

from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from catboost import CatBoostRegressor
from xgboost import XGBRegressor

from src.entities import EmailMessages
e_mail_messages = EmailMessages()


warnings.filterwarnings("ignore")


class ModelTraining:
    """
    A class to handle the model training process, including folder creation and metric selection.
    
    Attributes:
    model_preprocessing_artifacts : DataPreprocessingArtifacts
        An object with the paths of the transformed arrays and the trained_preprocessor
    model_training_artifacts : ModelTrainingArtifacts
        An object to store the model training artifacts that are going to be passed across the methods, and the paths to save the artifacts.
    model_training_params : dict
        Configuration parameters for model training, loaded from pipeline configuration, (input by user).
    """
    
    def __init__(self):
        """Initializes the ModelTraining class and creates the model training folder."""
        self.model_preprocessing_artifacts = DataPreprocessingArtifacts()
        self.model_training_artifacts = ModelTrainingArtifacts()
        self.model_training_params = pipeline_config_obj.model_training

        try:
            logger_obj.info("Creating the model_training folder.")
            os.makedirs(self.model_training_artifacts.model_training_root_folder, exist_ok=True)
            logger_obj.info(f"Successfully created the model_training folder at: {self.model_training_artifacts.model_training_root_folder}")
        
        except Exception as e:
            logger_obj.error(f"Error during creating the model_training folder. Error:\n{CustomException(e, sys)}")
            raise CustomException(e, sys)

    def get_scorer(self):
        """
        Defines scoring metrics for GridSearchCV based on configuration.
        
        Returns:
        str
            The scoring metric string to be used by GridSearchCV.
        
        Raises:
        ValueError
            If the `scoring_criteria` is not valid.
        """
        scoring_dict = {
            'r2_score': 'r2',
            'mae': 'neg_mean_absolute_error',
            'rmse': 'neg_root_mean_squared_error',
            'mse': 'neg_mean_squared_error'
        }
        
        scoring_criteria = self.model_training_params.main_scoring_criteria

        if scoring_criteria not in scoring_dict:
            logger_obj.error(f"Invalid scoring criteria: {scoring_criteria}. Must be one of ['r2_score', 'mae', 'rmse', 'mse']")
            raise ValueError("scoring_criteria must be one of ['r2_score', 'mae', 'rmse', 'mse']")
        
        logger_obj.info(f"Scorer selected: {scoring_dict[scoring_criteria]}")
        return scoring_dict[scoring_criteria]

    def get_metric_func(self):
        """
        Retrieves the metric function for final model evaluation.
        
        Returns:
        function
            The metric function (e.g., r2_score, mean_absolute_error) based on configuration.
        """
        metrics_dict = {
            'r2_score': r2_score,
            'mae': mean_absolute_error,
            'rmse': lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred)),
            'mse': mean_squared_error
        }

        scoring_criteria = self.model_training_params.main_scoring_criteria

        logger_obj.info(f"Metric function selected: {metrics_dict[scoring_criteria]}")
        return metrics_dict[scoring_criteria]

    def train_and_evaluate_models_cv_regression(self, 
                                                models_with_hyperparameters=[
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
        Evaluate multiple regression models using k-fold cross-validation and hyperparameter tuning.

        Parameters:
        -----------
        models_with_hyperparameters : list of tuples, default=predefined models
            List of tuples where each tuple contains a model and a dictionary of hyperparameters to be evaluated.
        
        Returns:
        --------
        None
            The method updates the `model_training_artifacts` with the best models, evaluation results, and overall best model.
        """
        
        # Load the transformed training and test data
        logger_obj.info("Loading transformed training and test data.")
        X_train_transformed = np.load(self.model_preprocessing_artifacts.X_train_transformed_path)
        X_test_transformed = np.load(self.model_preprocessing_artifacts.X_test_transformed_path)
        y_train = np.load(self.model_preprocessing_artifacts.y_train_path)
        y_test = np.load(self.model_preprocessing_artifacts.y_test_path)

        # Retrieve parameters from configuration
        k = self.model_training_params.number_of_folds_kfold
        scoring_criteria = self.model_training_params.main_scoring_criteria
        random_state = 42
        
        # Initialize variables for storing best models and results
        best_models = []
        results = []
        
        # Set up k-fold cross-validation
        logger_obj.info(f"Setting up {k}-fold cross-validation.")
        kf = KFold(n_splits=k, shuffle=True, random_state=random_state)
        
        # Get scorer and metric function based on the configured scoring criteria
        scorer = self.get_scorer()
        metric_func = self.get_metric_func()

        # Loop over each model and hyperparameter combination
        for model, hyperparameters in models_with_hyperparameters:
            logger_obj.info(f"Evaluating {model.__class__.__name__}...")
            
            # Add random_state to hyperparameters if applicable
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
            
            logger_obj.info(f"Starting grid search for {model.__class__.__name__}.")
            grid_search.fit(X_train_transformed, y_train)
            
            # Get the best model from grid search
            best_model = grid_search.best_estimator_

            # Make predictions using the best model
            y_pred = best_model.predict(X_test_transformed)
            
            # Calculate evaluation metrics
            performance = {
                'Model': model.__class__.__name__,
                'Best Hyperparameters': grid_search.best_params_,
                'R2 Score': r2_score(y_test, y_pred),
                'MAE': mean_absolute_error(y_test, y_pred),
                'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
                'MSE': mean_squared_error(y_test, y_pred),
                'CV Score': -grid_search.best_score_ if 'neg_' in scorer else grid_search.best_score_
            }
            
            # Log the performance of the model
            logger_obj.info(f"Model {model.__class__.__name__} performance: {performance}")
            
            # Append results and best model
            results.append(performance)
            best_models.append(best_model)
        
        # Create results DataFrame
        logger_obj.info("Creating results DataFrame.")
        results_df = pd.DataFrame(results)
        
        # Sort results based on scoring criteria
        logger_obj.info(f"Sorting results by {scoring_criteria}.")
        if scoring_criteria == 'r2_score':
            results_df = results_df.sort_values(by='R2 Score', ascending=False)
            best_score_value = results_df['R2 Score'].iloc[0]
        else:
            metric_name = scoring_criteria.upper()
            results_df = results_df.sort_values(by=metric_name, ascending=True)
            best_score_value = results_df[metric_name].iloc[0]
        
        # Get the best overall model
        best_model_idx = results_df.index[0]
        best_model_overall = best_models[best_model_idx]
        
        logger_obj.info("Evaluation complete. Best model selected.")
        
        # Store the best models, results, and best model overall
        self.model_training_artifacts.best_models = best_models
        self.model_training_artifacts.results = results_df
        self.model_training_artifacts.best_model_overall = best_model_overall
        self.model_training_artifacts.best_score_value = best_score_value  # Save the best score value
        
        # Optional: log the best score for clarity
        logger_obj.info(f"Best scoring criteria: {scoring_criteria} with score value: {best_score_value}")


    def save_results(self):
        """
        Save the evaluation results as an Excel file.
        
        This method saves the model evaluation results stored in `model_training_artifacts.results`
        to an Excel file in the specified folder.
        
        Returns:
        --------
        None
        """
        results_dataframe = self.model_training_artifacts.results
        path_to_save = os.path.join(self.model_training_artifacts.model_training_root_folder, "results.xlsx")
        
        try:
            logger_obj.info(f"Saving results to {path_to_save}.")
            results_dataframe.to_excel(path_to_save)
            logger_obj.info("Results saved successfully.")
        except Exception as e:
            logger_obj.error(f"Error saving results: {CustomException(e, sys)}")
            raise CustomException(e, sys)


    def save_best_models(self):
        """
        Save the best models to disk.
        
        This method saves all the best models, which are stored in 
        `model_training_artifacts.best_models`, as `.joblib` files in the specified folder.
        
        Returns:
        --------
        None
        """
        try:
            for model in self.model_training_artifacts.best_models:
                os.makedirs(self.model_training_artifacts.best_models_folder, exist_ok=True)
                
                filename = f"{model.__class__.__name__}.joblib"
                model_path = os.path.join(self.model_training_artifacts.best_models_folder, filename)
                
                logger_obj.info(f"Saving best model {model.__class__.__name__} to {model_path}.")
                joblib.dump(model, model_path)
                
            logger_obj.info("All best models saved successfully.")
        except Exception as e:
            logger_obj.error(f"Error saving best models: {CustomException(e, sys)}")
            raise CustomException(e, sys)


    def save_best_model_overall(self):
        """
        Save the best overall model to disk.
        
        This method saves the best overall model, which is stored in 
        `model_training_artifacts.best_model_overall`, as a `.joblib` file in the specified location.
        
        Returns:
        --------
        None
        """
        try:
            model = self.model_training_artifacts.best_model_overall
            logger_obj.info(f"Saving the best overall model to {self.model_training_artifacts.best_model_overall_path}.")
            joblib.dump(model, self.model_training_artifacts.best_model_overall_path)
            logger_obj.info("Best overall model saved successfully.")
        except Exception as e:
            logger_obj.error(f"Error saving the best overall model: {CustomException(e, sys)}")
            raise CustomException(e, sys)


    def train_and_evaluate_models_cv_regression_mlflow(self, 
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
        Evaluate multiple models using k-fold cross-validation with MLflow tracking.
        
        This method performs the following tasks:
        1. Loads pre-transformed data.
        2. Evaluates multiple models with grid search cross-validation.
        3. Logs performance metrics and model artifacts with MLflow.
        4. Saves the best models and evaluation results.
        
        Parameters:
        -----------
        models_with_hyperparameters : list of tuples, optional (default is a predefined list)
            List of (model, hyperparameters) to evaluate.
        """
        load_dotenv()
        
        # Load the data
        X_train_transformed = np.load(self.model_preprocessing_artifacts.X_train_transformed_path)
        X_test_transformed = np.load(self.model_preprocessing_artifacts.X_test_transformed_path)
        y_train = np.load(self.model_preprocessing_artifacts.y_train_path)
        y_test = np.load(self.model_preprocessing_artifacts.y_test_path)
        
        k = self.model_training_params.number_of_folds_kfold
        scoring_criteria = self.model_training_params.main_scoring_criteria 
        random_state = 42

        # Initialize variables
        best_models = []
        results = []
        
        # Set up cross-validation
        kf = KFold(n_splits=k, shuffle=True, random_state=random_state)
        
        # Get appropriate scorer for GridSearchCV
        scorer = self.get_scorer()
        
        # MLflow setup with unique experiment name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_name = f"Model_Comparison_{timestamp}"
        
        # Set tracking URI and create experiment
        mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI'))
        mlflow.create_experiment(experiment_name)
        mlflow.set_experiment(experiment_name)
        
        for model, hyperparameters in models_with_hyperparameters:
            # Start an MLflow run for each model
            with mlflow.start_run(nested=True):
                logger_obj.info(f"Evaluating {model.__class__.__name__}...")
                
                # MLflow log model name
                mlflow.set_tag('model_name', model.__class__.__name__)
                
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
                r2 = r2_score(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                mse = mean_squared_error(y_test, y_pred)
                cv_score = -grid_search.best_score_ if 'neg_' in scorer else grid_search.best_score_
                
                # Log metrics to MLflow
                mlflow.log_params(grid_search.best_params_)
                mlflow.log_metrics({
                    'r2_score': r2,
                    'mean_absolute_error': mae,
                    'root_mean_squared_error': rmse,
                    'mean_squared_error': mse,
                    'cross_validation_score': cv_score
                })
                
                # Log the model
                mlflow.sklearn.log_model(best_model, f"{model.__class__.__name__}_best_model")
                
                # Prepare performance dictionary
                performance = {
                    'Model': model.__class__.__name__,
                    'Best Hyperparameters': grid_search.best_params_,
                    'R2 Score': r2,
                    'MAE': mae,
                    'RMSE': rmse,
                    'MSE': mse,
                    'CV Score': cv_score
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
        
        logger_obj.info("Evaluation Complete.")
        
        self.model_training_artifacts.best_models = best_models 
        self.model_training_artifacts.results = results_df 
        self.model_training_artifacts.best_model_overall = best_model_overall

    def approve_model(self):

        # Get the scorer (the metrics or model evaluation results)        
        main_scoring_criteria = self.get_scorer()
        best_model_performance = self.model_training_artifacts.best_score_value
        minimal_performance = self.model_training_artifacts.minimal_performance
        logger_obj.info(f"Entering the step of approving the model")
        logger_obj.info(f"main_scoring_criteria: {main_scoring_criteria}, best_model_performance: {best_model_performance}, minimal_performance: {minimal_performance}")

        if main_scoring_criteria == 'r2':
            if best_model_performance >= minimal_performance:
                logger_obj.info(f"Model achieved the minimum expected performance with R2: {best_model_performance}")
            else:
                logger_obj.error(f"Model performance under the minimum (R2: {best_model_performance})\nEnding pipeline...") 
                e_mail_obj.send_email(e_mail_messages.model_approval_error_email_subject, e_mail_messages.model_approval_error_message)
                sys.exit(1)  

        else:
            if best_model_performance <= minimal_performance:
                logger_obj.info(f"Model achieved the minimum expected error (performance: {best_model_performance})")
            else:
                logger_obj.error(f"Model performance under the minimum (error: {best_model_performance})\nEnding pipeline...")
                e_mail_obj.send_email(e_mail_messages.model_approval_error_email_subject, e_mail_messages.model_approval_error_message)
                sys.exit(1)

  
    def run_model_training_and_evaluation(self):
        try:
            # self.train_and_evaluate_models_cv_regression_mlflow() # Uncomment if you want to track in MLflow
            self.train_and_evaluate_models_cv_regression() # Comment if you want to track changes in MLFlow
            self.approve_model() # comment this if running train_and_evaluate_models_cv_regression_mlflow, it still does not support 
            self.save_results()
            self.save_best_models()
            self.save_best_model_overall()
  
        except Exception as e:
            logger_obj.error(f"Error during run model training and evaluation :\n{CustomException(e ,sys)}")
            raise CustomException(e ,sys)


if __name__ == "__main__":
    model_training_obj = ModelTraining()
    model_training_obj.run_model_training_and_evaluation()


