import os
from dotenv import load_dotenv
load_dotenv()

from src.utils.exception import CustomException
from src.utils.logging import logger_obj
from src.utils.e_mail import EmailConfig

# E-mail configuration object:
e_mail_obj = EmailConfig(sender_email=os.getenv("SENDER_E_MAIL"), 
                         password= os.getenv("SENDER_E_MAIL_PASSWORD"),
                         recipient_email= os.getenv("RECEPIENT_E_MAIL"), 
                         subject_identifier="CI-CD-Machine-Learning-Pipeline-Students")


