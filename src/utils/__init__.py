import os
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

from src.utils.exception import CustomException
from src.utils.logging import Logger
from src.utils.e_mail import EmailConfig

logging_folder = Path("logs")

# You can change the method according to your desired logger format
logger_obj = Logger(logging_folder= logging_folder).create_daily_folder_logger()

# E-mail configuration object:
e_mail_obj = EmailConfig(sender_email=os.getenv("SENDER_E_MAIL"), 
                         password= os.getenv("SENDER_E_MAIL_PASSWORD"),
                         recipient_email= os.getenv("RECEPIENT_E_MAIL"), 
                         subject_identifier="CI-CD-Machine-Learning-Pipeline-Students")


