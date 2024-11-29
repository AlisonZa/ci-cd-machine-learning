import os
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from typing import Union
from typing import List, Optional, Dict, Any

######### Communications:
@dataclass
class EmailMessages:
    """
    A dataclass that encapsulates email subjects and messages for various stages 
    of a data pipeline. It provides predefined email content for both success and 
    error scenarios during different pipeline stages, including data ingestion, 
    validation, preprocessing, model training, and the overall model training pipeline.
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

    model_training_error_email_subject: str = f"""Error Model Training"""
    model_training_success_email_subject: str = f"""Model Training Completed"""
    model_training_error_message: str = f"""Error during Model Training, please check the logs"""
    model_training_success_message: str = f"""Successfully performed Model Training, starting next step"""

    model_approval_error_email_subject: str = f"""Error Model approval"""
    model_approval_error_message: str = f"""Error during Model approval, Model did not achieve the minimum performance, please check the logs"""

#### Training pipeline:###############################################################################################################

@dataclass(frozen=True)
class DataIngestionArtifacts:
    """
    A dataclass that defines the paths for storing artifacts generated 
    during the data ingestion stage of the pipeline.

    Attributes:
        data_ingestion_root_folder (str): Path to the root folder for data 
            ingestion artifacts. Defaults to "artifacts/data_ingestion".
        saved_raw_data_path (str): Path to the file where raw data is saved 
            during the data ingestion process. Defaults to 
            "artifacts/data_ingestion/raw_data.csv".
    """
    data_ingestion_root_folder: str = os.path.join("artifacts", "data_ingestion")  # Creates the step folder
    saved_raw_data_path: str = os.path.join(data_ingestion_root_folder, 'raw_data.csv')


@dataclass(frozen=True)
class DataValidationArtifacts:
    """
    A dataclass that defines the paths for storing artifacts generated 
    during the data validation stage of the pipeline.

    Attributes:
        data_validation_root_folder (str): Path to the root folder for data 
            validation artifacts. Defaults to "artifacts/data_validation".
        validated_data_path (str): Path to the file where validated data is saved 
            during the data validation process. Defaults to 
            "artifacts/data_validation/validated_data.csv".
    """
    data_validation_root_folder: str = os.path.join("artifacts", "data_validation")  # Creates the step folder
    validated_data_path: str = os.path.join(data_validation_root_folder, 'validated_data.csv')

@dataclass
class DataPreprocessingArtifacts:
    """
    A dataclass that defines the paths and intermediate data used and 
    generated during the data preprocessing stage of the pipeline. 

    Attributes:
        data_preprocessing_root_folder (str): Path to the root folder for 
            data preprocessing artifacts. Defaults to "artifacts/data_preprocessing".
        trained_preprocessor_path (str): Path to store the trained preprocessing 
            object as a pickle file. Defaults to "artifacts/data_preprocessing/trained_preprocessor.pkl".
        preprocessed_arrays_folder (str): Path to the folder containing 
            preprocessed data arrays. Defaults to "artifacts/data_preprocessing/preprocessed_data".
        X_train_transformed_path (str): Path to the file storing the preprocessed 
            training features. Defaults to "artifacts/data_preprocessing/preprocessed_data/X_train_transformed.npy".
        X_test_transformed_path (str): Path to the file storing the preprocessed 
            test features. Defaults to "artifacts/data_preprocessing/preprocessed_data/X_test_transformed.npy".
        y_train_path (str): Path to the file storing the training labels. Defaults to 
            "artifacts/data_preprocessing/preprocessed_data/y_train.npy".
        y_test_path (str): Path to the file storing the test labels. Defaults to 
            "artifacts/data_preprocessing/preprocessed_data/y_test_transformed.npy".
        X_train (Union[np.ndarray, pd.Series, pd.DataFrame]): Training features before preprocessing.
        X_test (Union[np.ndarray, pd.Series, pd.DataFrame]): Test features before preprocessing.
        y_train (Union[np.ndarray, pd.Series, pd.DataFrame]): Training labels.
        y_test (Union[np.ndarray, pd.Series, pd.DataFrame]): Test labels.
        X_train_transformed (Union[np.ndarray, pd.Series, pd.DataFrame]): Preprocessed training features.
        X_test_transformed (Union[np.ndarray, pd.Series, pd.DataFrame]): Preprocessed test features.
        trained_preprocessor: The preprocessing pipeline or object trained during the preprocessing stage.
    """
    data_preprocessing_root_folder: str = os.path.join("artifacts", "data_preprocessing")  # Creates the step folder
    trained_preprocessor_path: str = os.path.join(data_preprocessing_root_folder, "trained_preprocessor.pkl")
    preprocessed_arrays_folder: str = os.path.join(data_preprocessing_root_folder, "preprocessed_data")

    X_train_transformed_path: str = os.path.join(preprocessed_arrays_folder, "X_train_transformed.npy")
    X_test_transformed_path: str = os.path.join(preprocessed_arrays_folder, "X_test_transformed.npy")
    y_train_path: str = os.path.join(preprocessed_arrays_folder, "y_train.npy")
    y_test_path: str = os.path.join(preprocessed_arrays_folder, "y_test_transformed.npy")

    # Atributes shared across the diferents methods of the step
    X_train: Union[np.ndarray, pd.Series, pd.DataFrame] = None
    X_test: Union[np.ndarray, pd.Series, pd.DataFrame] = None
    y_train: Union[np.ndarray, pd.Series, pd.DataFrame] = None
    y_test: Union[np.ndarray, pd.Series, pd.DataFrame] = None
    X_train_transformed: Union[np.ndarray, pd.Series, pd.DataFrame] = None
    X_test_transformed: Union[np.ndarray, pd.Series, pd.DataFrame] = None
    trained_preprocessor = None


@dataclass
class ModelTrainingArtifacts:
    """
    A dataclass that defines the paths and intermediate results for 
    artifacts generated during the model training stage of the pipeline.

    Attributes:
        model_training_root_folder (str): Path to the root folder for model 
            training artifacts. Defaults to "artifacts/model_training".
        best_models_folder (str): Path to the folder storing the best models 
            identified during training. Defaults to "artifacts/model_training/best_models".
        best_model_overall_path (str): Path to the file storing the best overall 
            trained model. Defaults to "artifacts/model_training/best_model.joblib".
        best_model_overall: The best overall model identified during training.
        best_models: A collection or list of the best models trained during the 
            process.
        results: Training results or metrics (e.g., scores, performance evaluations).
        best_score_value: The best model performance according to the scoring criteria
        minimal_preformance: The minimal performance to approve the model, pay attention, because some metrics can be maximized while other minimized

    """
    model_training_root_folder: str = os.path.join("artifacts", "model_training")  # Creates the step folder
    best_models_folder = os.path.join(model_training_root_folder, "best_models")
    best_model_overall_path = os.path.join(model_training_root_folder, "best_model.joblib")

    # Atributes shared across the diferents methods of the step
    best_model_overall = None
    best_models = None
    results = None
    best_score_value:float = None
    
    # TODO move it to the configurator, so the user can manage it
    minimal_performance:float = 0.80 # 80% of r2_score



########### Prediction Pipeline: ##############################################################################################################

@dataclass
class PredictionInput:
    """
    Standardized input data structure for predictions.
    
    Represents the features required for predicting student performance. 
    Each attribute corresponds to a specific input feature.

    Attributes:
        gender (str): Gender of the student (e.g., "male", "female").
        race_ethnicity (str): Race/ethnicity group of the student.
        parental_level_of_education (str): Highest level of education achieved 
            by the student's parents (e.g., "high school", "bachelor's degree").
        lunch (str): Type of lunch the student receives (e.g., "standard", 
            "free/reduced").
        test_preparation_course (str): Whether the student completed a test 
            preparation course (e.g., "completed", "none").
        reading_score (int): Student's score in the reading section.
        writing_score (int): Student's score in the writing section.
    """
    gender: str
    race_ethnicity: str
    parental_level_of_education: str
    lunch: str
    test_preparation_course: str
    reading_score: int
    writing_score: int

    def to_dict(self) -> Dict[str, Any]:
        """
        Converts the dataclass instance to a dictionary for easier preprocessing.

        Returns:
            Dict[str, Any]: A dictionary representation of the prediction input.
        """
        return {
            'gender': self.gender,
            'race_ethnicity': self.race_ethnicity,
            'parental_level_of_education': self.parental_level_of_education,
            'lunch': self.lunch,
            'test_preparation_course': self.test_preparation_course,
            'reading_score': self.reading_score,
            'writing_score': self.writing_score
        }

@dataclass
class PredictionOutput:
    """
    Standardized output data structure for predictions.
    
    Represents the output of a prediction, including the predicted value, 
    optional probabilities for each class, and an optional explanation.

    Attributes:
        prediction (int): The predicted value or label (e.g., target score or class).
    """
    prediction: int