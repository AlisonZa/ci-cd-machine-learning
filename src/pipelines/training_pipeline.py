from src.utils import e_mail_obj
from src.entities import EmailMessages
from src.components.data_ingestion import DataIngestion
from src.components.data_validation import DataValidation
from src.utils import logger_obj, CustomException
import sys



def run_training_pipeline():
    logger_obj.info(f"{'*'*10}Entering the Training Pipeline{'*'*10}")
    
    e_mail_messages = EmailMessages()
    
    # Data Ingestion:
    data_ingestion_obj = DataIngestion()
    logger_obj.info(f"{'*'*5}Entering the Data Ingestion Step{'*'*5}")
    try:
        data_ingestion_obj.perform_data_ingestion_csv_files()
        e_mail_obj.send_email(e_mail_messages.data_ingestion_success_email_subject, e_mail_messages.data_ingestion_success_message)
        logger_obj.info(f"{'*'*5}Successfully ran the Data Ingestion Step{'*'*5}")

    except Exception as e:
        e_mail_obj.send_email(e_mail_messages.data_ingestion_error_email_subject, e_mail_messages.data_ingestion_error_message)
        logger_obj.exception(f"Exception During Data Ingestion Step\n:{e}")
        raise CustomException(e ,sys)

    
    # Data Validation:
    logger_obj.info(f"{'*'*5}Entering the Data Validation Step{'*'*5}")
    data_validation_obj = DataValidation()    
    
    try:
        data_validation_obj.run_data_validation()
        e_mail_obj.send_email(e_mail_messages.data_validation_success_email_subject, e_mail_messages.data_validation_success_message)
        logger_obj.info(f"{'*'*5}Successfully ran the Data Validation Step{'*'*5}")

    except Exception as e:
        e_mail_obj.send_email(e_mail_messages.data_validation_error_email_subject, e_mail_messages.data_validation_error_message)
        logger_obj.exception(f"Exception During Data Validation Step\n:{e}")
        raise CustomException(e ,sys)






    # Data Validation:


    # Data Preprocessing:

if __name__ == "__main__":
    run_training_pipeline()
