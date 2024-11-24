import os
from dataclasses import dataclass

# Emulating the root folder
# PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))

@dataclass
class EmailMessages:
    data_ingestion_error_email_subject: str = f"""Error Data Ingestion"""
    data_ingestion_success_email_subject: str = f"""Data Ingestion Completed"""

    data_ingestion_error_message: str = f"""Error during Data Ingestion, please check the logs"""
    data_ingestion_success_message: str = f"""Successfully performed Data Ingestion, starting next step"""




@dataclass
class DataIngestionConfig:
    raw_data_path: str = os.path.join(PROJECT_ROOT, "raw_data", "students.csv")

@dataclass
class DataIngestionArtifacts:
    data_ingestion_root_folder: str = os.path.join(PROJECT_ROOT, "artifacts", "data_ingestion") # Creates the step folder
    saved_raw_data_path: str = os.path.join(data_ingestion_root_folder, 'students.csv') 
