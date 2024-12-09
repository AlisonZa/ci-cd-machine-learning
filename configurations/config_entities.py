import os
from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    """
    Configuration for data ingestion settings.

    Attributes:
        raw_data_path (str): Path to the raw input data file. 
            Defaults to "raw_data/students.csv".
    """
    raw_data_path: str = os.path.join("raw_data", "students.csv")

@dataclass
class DataValidationConfig:
    """
    Configuration for data validation settings.

    Attributes:
        numerical_tolerance (float): Maximum allowable deviation for numerical 
            feature comparisons. Defaults to 1.3.
        categorical_tolerance (float): Maximum allowable difference for 
            categorical feature proportions. Defaults to 0.2.
        reference_statistics (str): Path to the file containing reference 
            dataset statistics. Defaults to "schemas/reference_stats.json".
    """
    numerical_tolerance: float = 1.3
    categorical_tolerance: float = 0.2
    reference_statistics: str = os.path.join("schemas", "reference_stats.json")

@dataclass
class ModelTrainingParams:
    """
    Configuration for model training parameters.

    Attributes:
        main_scoring_criteria (str): The primary metric used to evaluate models. 
            Defaults to "r2_score".
        number_of_folds_kfold (int): Number of folds for K-fold cross-validation.
            Defaults to 5.
    """
    main_scoring_criteria: str = 'r2_score'
    number_of_folds_kfold: int = 5
