import os, sys
from src.utils import logger_obj, e_mail_obj ,CustomException
from src.entities import DataIngestionConfig, DataIngestionArtifacts, EmailMessages
import pandas as pd


# Emulating the root folder
# PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

class DataIngestion:
    def __init__(self):
        self.data_ingestion_config= DataIngestionConfig()
        self.data_ingestion_artifacts = DataIngestionArtifacts()

        try:
            logger_obj.info(f"Creating the data_ingestion folder")
            os.makedirs(self.data_ingestion_artifacts.data_ingestion_root_folder, exist_ok= True)
            logger_obj.info(f"Succesfully created the data_ingestion folder at: \n{self.data_ingestion_artifacts.data_ingestion_root_folder}")
        
        except Exception as e:
            logger_obj.error(f"Error during creating the data_ingestion folder, Error:\n{CustomException(e ,sys)}")
            raise CustomException(e ,sys)

    def perform_data_ingestion_csv_files(self):
        try:
            logger_obj.info(f"Loading the data from:\n{self.data_ingestion_config.raw_data_path}")
            raw_dataframe = pd.read_csv(self.data_ingestion_config.raw_data_path)

            raw_dataframe.to_csv(self.data_ingestion_artifacts.saved_raw_data_path, index= False)
            logger_obj.info(f"Succesfully saved the data to the artifacts folder:\n{self.data_ingestion_artifacts.saved_raw_data_path}")

        except Exception as e:
            logger_obj.error(f"Error during data_ingestion :\n{CustomException(e ,sys)}")
            raise CustomException(e ,sys)

#Testing entrypoint        
if __name__ == "__main__":

    e_mail_messages = EmailMessages()

    data_ingestion_obj = DataIngestion()

    if __name__ == "__main__":
        try:
            data_ingestion_obj.perform_data_ingestion_csv_files()
            e_mail_obj.send_email(e_mail_messages.data_ingestion_success_email_subject, e_mail_messages.data_ingestion_success_message)
        except:
            e_mail_obj.send_email(e_mail_messages.data_ingestion_error_email_subject, e_mail_messages.data_ingestion_error_message)






