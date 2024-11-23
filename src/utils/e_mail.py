import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
import mimetypes
from pathlib import Path
from src.utils import CustomException  # Importing custom exception and logger
from src.utils.logging import Logger
import sys

logging_folder = Path("logs")
e_mail_logger = Logger(logging_folder= logging_folder).create_daily_folder_logger()

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

        Args:
            sender_email (str): The sender's email address. DO NOT HARDCODE, USE A .ENV file instead.
            password (str): The password for the sender's email account. DO NOT HARDCODE, USE A .ENV file instead.
            recipient_email (str): The default recipient's email address. DO NOT HARDCODE, USE A .ENV file instead.
            subject_identifier (str): Default identifier prefix for email subjects.
        """
        self.sender_email = sender_email
        self.password = password
        self.recipient_email = recipient_email
        self.subject_identifier = subject_identifier

    def send_email(self, subject: str, body: str, recipient_email: str = None):
        """
        Sends a plain text email to a recipient.

        This method creates a plain text email with the provided subject and body, 
        then sends it to the specified recipient using the SMTP protocol.

        Args:
            subject (str): The subject line of the email (will be prefixed with subject_identifier).
            body (str): The plain text content of the email.
            recipient_email (str, optional): Override the default recipient email. 
                                          If None, uses the default recipient_email.

        Raises:
            CustomException: If there is an issue with sending the email.
        """
        # Use default recipient if none provided
        target_email = recipient_email or self.recipient_email
        
        # Create the full subject with identifier
        full_subject = f"{self.subject_identifier}: {subject}"

        # Create the email content
        message = MIMEText(body, "plain")
        message["Subject"] = full_subject
        message["From"] = self.sender_email
        message["To"] = target_email

        try:
            # Establish connection with the SMTP server
            with smtplib.SMTP("smtp.gmail.com", 587) as server:
                server.starttls()
                server.login(self.sender_email, self.password)
                server.send_message(message)
                e_mail_logger.info(f"Email sent successfully to {target_email} with subject: {full_subject}")
        except Exception as e:
            error_message = f"Error sending email to {target_email}: {str(e)}"
            e_mail_logger.error(error_message)  # Log the error using the custom logger
            raise CustomException(error_message, sys)  # Raise a custom exception with error details

    def send_email_with_attachments(
        self, subject: str, body: str, attachments: list[Path], recipient_email: str = None
    ):
        """
        Sends an email with multiple attachments.

        Args:
            subject (str): The subject of the email (will be prefixed with subject_identifier).
            body (str): The body of the email.
            attachments: A list of file paths (Path objects) to attach.
            recipient_email (str, optional): Override the default recipient email.
                                          If None, uses the default recipient_email.

        Raises:
            CustomException: If there is an issue with sending the email.
        """
        # Use default recipient if none provided
        target_email = recipient_email or self.recipient_email
        
        # Create the full subject with identifier
        full_subject = f"{self.subject_identifier}: {subject}"

        # Create the email object
        message = MIMEMultipart()
        message["From"] = self.sender_email
        message["To"] = target_email
        message["Subject"] = full_subject

        # Attach the body
        message.attach(MIMEText(body, "plain"))

        # Attach multiple files
        for attachment in attachments:
            if not attachment.exists():
                e_mail_logger.warning(f"Attachment not found: {attachment}")
                continue

            # Detect MIME type
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
            except Exception as e:
                error_message = f"Error attaching file {attachment}: {str(e)}"
                e_mail_logger.error(error_message)  # Log the error using the custom logger
                raise CustomException(error_message, sys)  # Raise a custom exception with error details

        # Send the email
        try:
            with smtplib.SMTP("smtp.gmail.com", 587) as server:
                server.starttls()
                server.login(self.sender_email, self.password)
                server.send_message(message)
                e_mail_logger.info(f"Email with attachments sent successfully to {target_email} with subject: {full_subject}")
        except Exception as e:
            error_message = f"Error sending email with attachments to {target_email}: {str(e)}"
            e_mail_logger.error(error_message)  # Log the error using the custom logger
            raise CustomException(error_message, sys)  # Raise a custom exception with error details
