import logging
import os
from datetime import datetime
from pathlib import Path

class Logger:
    """
    A class to handle the logs.

    Attributes:
        logging_folder (Path): The path in which the logging information will be stored
    """

    def __init__(self, logging_folder: Path = Path("./logs")):
        self.logging_folder = logging_folder

        # Create the logs directory if it doesn't exist
        os.makedirs(self.logging_folder, exist_ok=True)

    def create_daily_folder_logger(self):
        """
        Create a logger that logs into a folder for today's date, with a timestamped file.
        """
        # Create a folder for today's date inside the logging folder
        today_date = datetime.now().strftime('%Y-%m-%d')
        folder_path = self.logging_folder / today_date
        
        # Ensure the folder exists
        os.makedirs(folder_path, exist_ok=True)
        
        # Generate the log filename with timestamp
        log_filename = datetime.now().strftime('%H-%M-%S') + '.log'
        log_path = folder_path / log_filename
        
        # Create and configure logger
        logger = logging.getLogger('daily_folder_logger')
        logger.setLevel(logging.INFO)
        
        # Create file handler
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(logging.INFO)
        
        # Create formatter and add it to the handler
        formatter = logging.Formatter('%(asctime)s - %(message)s')
        file_handler.setFormatter(formatter)
        
        # Add handler to logger
        logger.addHandler(file_handler)
        
        return logger

    def create_run_specific_logger(self):
        """
        Create a logger that logs into a separate file for each run with a timestamped filename.
        """
        # Generate the log filename with timestamp
        log_filename = datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + '.log'
        log_path = self.logging_folder / log_filename
        
        # Create and configure logger
        logger = logging.getLogger('run_specific_logger')
        logger.setLevel(logging.INFO)
        
        # Create file handler
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(logging.INFO)
        
        # Create formatter and add it to the handler
        formatter = logging.Formatter('%(asctime)s - %(message)s')
        file_handler.setFormatter(formatter)
        
        # Add handler to logger
        logger.addHandler(file_handler)
        
        return logger

    def create_single_file_logger(self):
        """
        Create a logger that logs into a single file that accumulates all logs.
        """
        # Single log file inside the logs folder
        log_filename = 'all_logs.log'
        log_path = self.logging_folder / log_filename
        
        # Create and configure logger
        logger = logging.getLogger('single_file_logger')
        logger.setLevel(logging.INFO)
        
        # Create file handler
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(logging.INFO)
        
        # Create formatter and add it to the handler
        formatter = logging.Formatter('%(asctime)s - %(message)s')
        file_handler.setFormatter(formatter)
        
        # Add handler to logger
        logger.addHandler(file_handler)
        
        return logger
