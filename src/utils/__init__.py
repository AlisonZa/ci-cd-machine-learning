import os
from pathlib import Path

from src.utils.exception import CustomException
from src.utils.logging import Logger

logging_folder = Path("logs")

# You can change the method according to your desired logger format
logger_obj = Logger(logging_folder= logging_folder).create_daily_folder_logger()


