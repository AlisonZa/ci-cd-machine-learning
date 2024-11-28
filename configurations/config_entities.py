import os
from dataclasses import dataclass, field

@dataclass
class DataIngestionConfig:
    raw_data_path: str = os.path.join("raw_data", "students.csv")

@dataclass
class DataValidationConfig:
    numerical_tolerance: float = 1.3
    categorical_tolerance: float = 0.2
    reference_statistics: str = os.path.join("schemas", "reference_stats.json")

@dataclass
class FeatureDefinition:
    target_column: str = "math_score"
    categorical_ordinals: list[str] = field(default_factory=lambda: ["parental_level_of_education"])
    categorical_nominals: list[str] = field(default_factory=lambda: ["gender", "race_ethnicity", "lunch", "test_preparation_course"])
    numeric_scalars: list[str] = field(default_factory=lambda: ["reading_score", "writing_score"])

@dataclass
class ModelTrainingParams:
    main_scoring_criteria: str = 'r2_score'
    number_of_folds_kfold: int = 5