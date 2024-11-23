import logging
import os
from datetime import datetime

LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))

logs_path = os.path.join(PROJECT_ROOT, "logs", LOG_FILE )

os.makedirs(logs_path, exist_ok= True)

LOG_FILE_PATH = os.path.join(logs_path, LOG_FILE)

logging.basicConfig(
    filename= LOG_FILE_PATH,
    format = "[ %(asctime)s ] - %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level = logging.INFO,
)

if __name__ == "__main__":
    logging.info("Logging has started")