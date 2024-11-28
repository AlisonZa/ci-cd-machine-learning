from src.utils import e_mail_obj, logger_obj, CustomException
from src.entities import EmailMessages
import sys

from src.components.data_ingestion import DataIngestion
from src.components.data_validation import DataValidation
from src.components.data_preprocessing import DataPreprocessing
from src.components.model_training import ModelTraining

from configurations import pipeline_config_obj

def run_training_pipeline():
    logger_obj.info(f"{'*'*10}Entering the Training Pipeline{'*'*10}")
    logger_obj.info(f"Pipeline Configurations: {pipeline_config_obj}")

    e_mail_messages = EmailMessages()
    
    # Data Ingestion:
    data_ingestion_obj = DataIngestion()
    logger_obj.info(f"{'*'*5}Entering the Data Ingestion Step{'*'*5}")
    try:
        data_ingestion_obj.perform_data_ingestion_csv_files()
        # e_mail_obj.send_email(e_mail_messages.data_ingestion_success_email_subject, e_mail_messages.data_ingestion_success_message)
        logger_obj.info(f"{'*'*5}Successfully ran the Data Ingestion Step{'*'*5}")
 
    except Exception as e: 
        # e_mail_obj.send_email(e_mail_messages.data_ingestion_error_email_subject, e_mail_messages.data_ingestion_error_message)
        logger_obj.exception(f"Exception During Data Ingestion Step\n:{e}")
        raise CustomException(e ,sys)

    
    # Data Validation:
    logger_obj.info(f"{'*'*5}Entering the Data Validation Step{'*'*5}")
    data_validation_obj = DataValidation()    
    
    try:
        data_validation_obj.run_data_validation()
        # e_mail_obj.send_email(e_mail_messages.data_validation_success_email_subject, e_mail_messages.data_validation_success_message)
        logger_obj.info(f"{'*'*5}Successfully ran the Data Validation Step{'*'*5}")

    except Exception as e:
        # e_mail_obj.send_email(e_mail_messages.data_validation_error_email_subject, e_mail_messages.data_validation_error_message)
        logger_obj.exception(f"Exception During Data Validation Step\n:{e}")
        raise CustomException(e ,sys)


    # Data Preprocessing:
    logger_obj.info(f"{'*'*5}Entering the Data Preprocessing Step{'*'*5}")
    data_preprocessing_obj = DataPreprocessing()    
    
    try:
        data_preprocessing_obj.run_data_preprocessing()
        # e_mail_obj.send_email(e_mail_messages.data_preprocessing_success_email_subject, e_mail_messages.data_preprocessing_success_message)
        logger_obj.info(f"{'*'*5}Successfully ran the Data Preprocessing Step{'*'*5}")

    except Exception as e:
        # e_mail_obj.send_email(e_mail_messages.data_preprocessing_error_email_subject, e_mail_messages.data_preprocessing_error_message)
        logger_obj.exception(f"Exception During Data Preprocessing Step\n:{e}")
        raise CustomException(e ,sys)

    # Model Training:
    logger_obj.info(f"{'*'*5}Entering the Model Training Step{'*'*5}")
    model_training_obj = ModelTraining()    
    
    try:
        model_training_obj.run_model_training_and_evaluation()
        # e_mail_obj.send_email(e_mail_messages.model_training_success_email_subject, e_mail_messages.model_training_success_message)
        logger_obj.info(f"{'*'*5}Successfully ran the Model Training Step{'*'*5}")

    except Exception as e:
        # e_mail_obj.send_email(e_mail_messages.model_training_error_email_subject, e_mail_messages.model_training_error_message)
        logger_obj.exception(f"Exception During Model Training Step\n:{e}")
        raise CustomException(e ,sys)

if __name__ == "__main__":
    run_training_pipeline()
