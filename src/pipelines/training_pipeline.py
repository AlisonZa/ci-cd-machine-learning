from src.utils import e_mail_obj
from src.entities import EmailMessages
from src.components.data_ingestion import DataIngestion
from src.components.data_validation import DataValidation
from src.utils import logger_obj




def run_training_pipeline():
    
    e_mail_messages = EmailMessages()
    
    # Data Ingestion:
    logger_obj.info(f'')
    data_ingestion_obj = DataIngestion()
    try:
        data_ingestion_obj.perform_data_ingestion_csv_files()
        e_mail_obj.send_email(e_mail_messages.data_ingestion_success_email_subject, e_mail_messages.data_ingestion_success_message)
    except:
        e_mail_obj.send_email(e_mail_messages.data_ingestion_error_email_subject, e_mail_messages.data_ingestion_error_message)

    
    # Data Validation:
    data_validation_obj = DataValidation()
    
    try:
        data_validation_obj.run_data_validation()
        e_mail_obj.send_email(e_mail_messages.data_validation_success_email_subject, e_mail_messages.data_validation_success_message)
    except:
        e_mail_obj.send_email(e_mail_messages.data_validation_error_email_subject, e_mail_messages.data_validation_error_message)




    # Data Validation:


    # Data Preprocessing:

if __name__ == "__main__":
    run_training_pipeline()
