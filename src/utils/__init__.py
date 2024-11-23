import os
from pathlib import Path

from src.utils.exception import CustomException
from src.utils.logging import Logger

logging_folder = Path("logs")  
logger_obj = Logger(logging_folder= logging_folder).create_daily_folder_logger()