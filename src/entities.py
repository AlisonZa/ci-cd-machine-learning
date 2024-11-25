import os
from dataclasses import dataclass


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

@dataclass
class DataIngestionConfig:
    raw_data_path: str = os.path.join( "raw_data", "students.csv")

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

@dataclass
class DataValidationArtifacts:
    data_validation_root_folder: str = os.path.join( "artifacts", "data_validation") # Creates the step folder
    validated_data_path: str = os.path.join(data_validation_root_folder, 'validated_data.csv') 
    data_validation_report_folder: str = os.path.join(data_validation_root_folder, 'data_validation_report.json') 






    