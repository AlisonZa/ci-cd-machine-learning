import os
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from typing import Union

# TODO comment, typing suggestion


@dataclass
class EmailMessages:
    """
    
    
    """
    data_ingestion_error_email_subject: str = f"""Error Data Ingestion"""
    data_ingestion_success_email_subject: str = f"""Data Ingestion Completed"""
    data_ingestion_error_message: str = f"""Error during Data Ingestion, please check the logs"""
    data_ingestion_success_message: str = f"""Successfully performed Data Ingestion, starting next step"""


    data_validation_error_email_subject: str = f"""Error Data Validation"""
    data_validation_success_email_subject: str = f"""Data Validation Completed"""
    data_validation_error_message: str = f"""Error during Data Validation, please check the logs"""
    data_validation_success_message: str = f"""Successfully performed Data Validation, starting next step"""

    data_preprocessing_error_email_subject: str = f"""Error Data Preprocessing"""
    data_preprocessing_success_email_subject: str = f"""Data Preprocessing Completed"""
    data_preprocessing_error_message: str = f"""Error during Data Preprocessing, please check the logs"""
    data_preprocessing_success_message: str = f"""Successfully performed Data Preprocessing, starting next step"""

@dataclass
class DataIngestionConfig:
    raw_data_path: str = os.path.join("raw_data", "students.csv")

@dataclass
class DataIngestionArtifacts:
    data_ingestion_root_folder: str = os.path.join( "artifacts", "data_ingestion") # Creates the step folder
    saved_raw_data_path: str = os.path.join(data_ingestion_root_folder, 'students.csv') 

@dataclass
class DataValidationConfig:
    """
    Configuration class for data validation thresholds.

    Attributes:
        numerical_tolerance (float): The maximum allowed deviation for numerical features 
            before they are flagged as having drift. Default is 1.3.
        categorical_tolerance (float): The maximum allowed deviation for categorical features 
            before they are flagged as having drift. Default is 0.2.
    """
    numerical_tolerance: float = 1.3
    categorical_tolerance: float = 0.2
    reference_statistics: str = os.path.join( "schemas", "reference_stats.json")


@dataclass(frozen = True)
class DataValidationArtifacts:
    data_validation_root_folder: str = os.path.join( "artifacts", "data_validation") # Creates the step folder
    validated_data_path: str = os.path.join(data_validation_root_folder, 'validated_data.csv')



@dataclass
class DataPreprocessingArtifacts:
    data_preprocessing_root_folder: str = os.path.join( "artifacts", "data_preprocessing") # Creates the step folder
    
    trained_preprocessor_path: str = os.path.join(data_preprocessing_root_folder, "trained_preprocessor.pkl") 
    
    preprocessed_arrays_folder: str = os.path.join(data_preprocessing_root_folder, "preprocessed_data")

    # Paths to store the files that are going to be passed to our next step
    X_train_transformed_path: str = os.path.join( preprocessed_arrays_folder, "X_train_transformed.npy")
    X_test_transformed_path: str = os.path.join( preprocessed_arrays_folder, "X_test_transformed.npy")
    y_train_path: str = os.path.join( preprocessed_arrays_folder, "y_train.npy")
    y_test_path: str = os.path.join( preprocessed_arrays_folder, "y_test_transformed.npy")

    # Attributes to store the "variables" that are going to be passed across the different functions
    X_train: Union[np.ndarray, pd.Series, pd.DataFrame] = None
    X_test: Union[np.ndarray, pd.Series, pd.DataFrame] = None
    y_train: Union[np.ndarray, pd.Series, pd.DataFrame] = None
    y_test: Union[np.ndarray, pd.Series, pd.DataFrame] = None
        
    X_train_transformed: Union[np.ndarray, pd.Series, pd.DataFrame] = None
    X_test_transformed: Union[np.ndarray, pd.Series, pd.DataFrame] = None

    trained_preprocessor = None


@dataclass
class FeatureDefinition:
    target_column: str = "math_score"
    categorical_ordinals: list[str] = field(default_factory=lambda: ["parental_level_of_education"])
    categorical_nominals: list[str] = field(default_factory=lambda: ["gender", "race_ethnicity", "lunch", "test_preparation_course"])
    numeric_scalars: list[str] = field(default_factory=lambda: ["reading_score", "writing_score"])


@dataclass
class ModelTrainingArtifacts:
    model_training_root_folder: str = os.path.join( "artifacts", "model_training") # Creates the step folder
    best_models_folder = os.path.join(model_training_root_folder, "best_models")

    best_model_overall = None
    best_models = None
    results = None
    

@dataclass
class ModelTrainingParams:
    main_scoring_criteria: str = 'r2_score'
    number_of_folds_kfold:int = 5



