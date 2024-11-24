import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
import mimetypes
from pathlib import Path
from src.utils import CustomException  # Importing custom exception and logger
import sys
from src.utils.logging import logger_obj

class EmailConfig:
    """
    A class to handle email configuration and sending emails.

    Attributes:
        sender_email (str): The sender's email address, must be a @gmail.com e-mail. DO NOT HARDCODE, USE A .ENV file instead.
        password (str): The password for the sender's email account. DO NOT HARDCODE, USE A .ENV file instead.
        recipient_email (str): The default recipient's email address. DO NOT HARDCODE, USE A .ENV file instead.
        subject_identifier (str): Default identifier prefix for email subjects.
    """

    def __init__(self, 
                 sender_email: str, 
                 password: str,
                 recipient_email: str,
                 subject_identifier: str = "Machine Learning Pipeline"):
        """
        Initializes the EmailConfig object with the sender's email, password, and default recipient.
        """
        self.sender_email = sender_email
        self.password = password
        self.recipient_email = recipient_email
        self.subject_identifier = subject_identifier
        logger_obj.info("EmailConfig initialized successfully.")

    def send_email(self, subject: str, body: str, recipient_email: str = None):
        """
        Sends a plain text email to a recipient.
        """
        target_email = recipient_email or self.recipient_email
        full_subject = f"{self.subject_identifier}: {subject}"
        
        # Log the email details (without sensitive data)
        logger_obj.info(f"Preparing to send email to {target_email} with subject: {full_subject}")
        
        message = MIMEText(body, "plain")
        message["Subject"] = full_subject
        message["From"] = self.sender_email
        message["To"] = target_email

        try:
            with smtplib.SMTP("smtp.gmail.com", 587) as server:
                logger_obj.info("Connecting to SMTP server.")
                server.starttls()
                logger_obj.info("Starting TLS encryption.")
                server.login(self.sender_email, self.password)
                logger_obj.info("Logged into SMTP server successfully.")
                server.send_message(message)
                logger_obj.info(f"Email sent successfully to {target_email}.")
        except Exception as e:
            error_message = f"Error sending email to {target_email}: {str(e)}"
            logger_obj.error(error_message)
            raise CustomException(error_message, sys)

    def send_email_with_attachments(
        self, subject: str, body: str, attachments: list[Path], recipient_email: str = None
    ):
        """
        Sends an email with multiple attachments.
        """
        target_email = recipient_email or self.recipient_email
        full_subject = f"{self.subject_identifier}: {subject}"
        
        # Log the email and attachment details
        logger_obj.info(f"Preparing to send email with attachments to {target_email} with subject: {full_subject}")
        logger_obj.info(f"Attachments: {[str(attachment) for attachment in attachments]}")

        message = MIMEMultipart()
        message["From"] = self.sender_email
        message["To"] = target_email
        message["Subject"] = full_subject

        message.attach(MIMEText(body, "plain"))

        for attachment in attachments:
            if not attachment.exists():
                logger_obj.warning(f"Attachment not found: {attachment}")
                continue

            mime_type, _ = mimetypes.guess_type(attachment)
            mime_main, mime_sub = (mime_type or "application/octet-stream").split("/")

            try:
                with attachment.open("rb") as file:
                    part = MIMEBase(mime_main, mime_sub)
                    part.set_payload(file.read())
                encoders.encode_base64(part)
                part.add_header(
                    "Content-Disposition",
                    f"attachment; filename={attachment.name}",
                )
                message.attach(part)
                logger_obj.info(f"Attached file: {attachment}")
            except Exception as e:
                error_message = f"Error attaching file {attachment}: {str(e)}"
                logger_obj.error(error_message)
                raise CustomException(error_message, sys)

        try:
            with smtplib.SMTP("smtp.gmail.com", 587) as server:
                logger_obj.info("Connecting to SMTP server.")
                server.starttls()
                logger_obj.info("Starting TLS encryption.")
                server.login(self.sender_email, self.password)
                logger_obj.info("Logged into SMTP server successfully.")
                server.send_message(message)
                logger_obj.info(f"Email with attachments sent successfully to {target_email}.")
        except Exception as e:
            error_message = f"Error sending email with attachments to {target_email}: {str(e)}"
            logger_obj.error(error_message)
            raise CustomException(error_message, sys)
