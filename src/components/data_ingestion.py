import os, sys
from src.utils import logger_obj,CustomException
import pandas as pd

from src.entities import DataIngestionArtifacts
from configurations import pipeline_config_obj


class DataIngestion:
    """
    A class to handle the ingestion of raw data for processing and storage.

    Attributes:
        data_ingestion_config: Configuration object containing paths related to data ingestion.
        data_ingestion_artifacts: Object to handle paths where data will be saved after ingestion.

    Methods:
        __init__: Initializes the DataIngestion object and creates necessary folders for data storage.
        perform_data_ingestion_csv_files: Loads raw CSV data and saves it to an artifacts folder.
    """

    def __init__(self):
        """
        Initializes the DataIngestion object and creates a folder to store the ingested data.

        The folder is created using the path defined in the data_ingestion_artifacts object.
        If the folder creation fails, an exception is logged and raised.
        """
        # Load configuration settings for data ingestion
        self.data_ingestion_config = pipeline_config_obj.data_ingestion
        self.data_ingestion_artifacts = DataIngestionArtifacts()

        try:
            # Attempt to create the data_ingestion folder if it doesn't exist
            logger_obj.info("Creating the data_ingestion folder")
            os.makedirs(self.data_ingestion_artifacts.data_ingestion_root_folder, exist_ok=True)
            logger_obj.info(f"Successfully created the data_ingestion folder at: \n{self.data_ingestion_artifacts.data_ingestion_root_folder}")
        
        except Exception as e:
            # Log any error that occurs during folder creation
            logger_obj.error(f"Error during creating the data_ingestion folder, Error:\n{CustomException(e, sys)}")
            raise CustomException(e, sys)

    def perform_data_ingestion_csv_files(self):
        """
        Loads raw CSV data from a specified path and saves it to a designated location.

        This method reads the raw CSV file, processes it, and saves it to the artifacts 
        folder for further use in the pipeline.

        Raises:
            CustomException: If there is an error while reading or saving the CSV file.
        """
        try:
            # Log the loading of the raw data
            logger_obj.info(f"Loading the data from:\n{self.data_ingestion_config.raw_data_path}")
            raw_dataframe = pd.read_csv(self.data_ingestion_config.raw_data_path)

            # Save the raw data to the specified path in the artifacts folder
            raw_dataframe.to_csv(self.data_ingestion_artifacts.saved_raw_data_path, index=False)
            logger_obj.info(f"Successfully saved the data to the artifacts folder:\n{self.data_ingestion_artifacts.saved_raw_data_path}")

        except Exception as e:
            # Log any error that occurs during data ingestion
            logger_obj.error(f"Error during data_ingestion:\n{CustomException(e, sys)}")
            raise CustomException(e, sys)


if __name__ == "__main__":
    data_ingestion_obj = DataIngestion()
    data_ingestion_obj.perform_data_ingestion_csv_files()





