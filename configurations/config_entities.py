import os
from dataclasses import dataclass, field

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
class FeatureDefinition:
    """
    Definition of dataset features including target, categorical, and numerical attributes.

    Attributes:
        target_column (str): Name of the target column. Defaults to "math_score".
        categorical_ordinals (list[str]): List of ordinal categorical feature names.
            Defaults to ["parental_level_of_education"].
        categorical_nominals (list[str]): List of nominal categorical feature names.
            Defaults to ["gender", "race_ethnicity", "lunch", "test_preparation_course"].
        numeric_scalars (list[str]): List of numerical scalar feature names. 
            Defaults to ["reading_score", "writing_score"].
    """
    target_column: str = "math_score"
    categorical_ordinals: list[str] = field(default_factory=lambda: ["parental_level_of_education"])
    categorical_nominals: list[str] = field(default_factory=lambda: ["gender", "race_ethnicity", "lunch", "test_preparation_course"])
    numeric_scalars: list[str] = field(default_factory=lambda: ["reading_score", "writing_score"])


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
