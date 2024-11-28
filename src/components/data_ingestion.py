import os, sys
from src.utils import logger_obj,CustomException
from src.entities import DataIngestionConfig, DataIngestionArtifacts
import pandas as pd

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

if __name__ == "__main__":
    data_ingestion_obj = DataIngestion()
    data_ingestion_obj.perform_data_ingestion_csv_files()





