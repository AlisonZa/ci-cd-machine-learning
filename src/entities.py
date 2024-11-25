import os
from dataclasses import dataclass


@dataclass
class EmailMessages:
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
    reference_statistics: str = os.path.join("schemas", "reference_stats.json") 
    numerical_tolerance: float = 3.0
    categorical_tolerance: float = 1.0 

@dataclass
class DataValidationArtifacts:
    data_validation_root_folder: str = os.path.join( "artifacts", "data_validation") # Creates the step folder
    validated_data_path: str = os.path.join(data_validation_root_folder, 'validated_data.csv') 




    