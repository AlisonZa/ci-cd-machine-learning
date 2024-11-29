import joblib
from typing import Dict, Any
from src.entities import PredictionInput, PredictionOutput, ModelTrainingArtifacts, DataPreprocessingArtifacts
from typing import List, Union
import pandas as pd
import os
import logging
from datetime import datetime
import os
from typing import Union, List

import logging
from typing import Dict, Any, List, Union
from datetime import datetime
from dotenv import load_dotenv
load_dotenv()
import psycopg2
import os
import logging
from datetime import datetime
from typing import List
import os
from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError, OperationalError
from dotenv import load_dotenv
import boto3

import pandas as pd
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Float
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.exc import SQLAlchemyError

import numpy as np

import pandas as pd
import sqlalchemy
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

from src.entities import PredictionInput, PredictionOutput, ModelTrainingArtifacts, DataPreprocessingArtifacts

# Database credentials from .env
DB_HOST = os.getenv("DB_HOST")
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_PORT = os.getenv("DB_PORT")

import os
from datetime import datetime
from src.entities import PredictionInput

class PredictionLoggerLocal:
    def __init__(self, log_dir='prediction_logs'):
        """
        Initialize the local prediction logger.
        
        Args:
            log_dir (str, optional): Directory to store prediction log files. 
                                     Defaults to 'prediction_logs'.
        """ 
        # Create log directory if it doesn't exist
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)

    def _get_timestamp(self):
        """
        Generate a timestamp for log file naming.
        
        Returns:
            str: Formatted timestamp
        """
        return datetime.now().strftime("%Y%m%d_%H%M%S")

    def log_prediction(self, 
                       input_data: Union[List[PredictionInput], PredictionInput], 
                       prediction: Union[List[PredictionOutput], PredictionOutput]):
        """
        Log a single or batch prediction to a CSV file.
        
        Args:
            input_data (Union[List[PredictionInput], PredictionInput]): Input feature(s)
            prediction (Union[List[PredictionOutput], PredictionOutput]): Prediction result(s)
        """
        # Ensure inputs are lists
        if isinstance(input_data, PredictionInput):
            input_data = [input_data]
        
        if isinstance(prediction, PredictionOutput):
            prediction = [prediction]
        
        # Validate input lengths
        if len(input_data) != len(prediction):
            raise ValueError("Input data and prediction lists must have the same length")

        # Prepare log data
        log_entries = []
        for inp, pred in zip(input_data, prediction):
            log_entry = {
                'timestamp': datetime.now().isoformat(),
                'gender': inp.gender,
                'race_ethnicity': inp.race_ethnicity,
                'parental_level_of_education': inp.parental_level_of_education,
                'lunch': inp.lunch,
                'test_preparation_course': inp.test_preparation_course,
                'reading_score': inp.reading_score,
                'writing_score': inp.writing_score,
                'prediction': pred.prediction
            }
            log_entries.append(log_entry)

        # Create DataFrame
        log_df = pd.DataFrame(log_entries)

        # Generate unique filename
        filename = f"prediction_log_{self._get_timestamp()}.csv"
        filepath = os.path.join(self.log_dir, filename)

        # Write to CSV
        log_df.to_csv(filepath, index=False)
        print(f"Prediction log saved to {filepath}")

        return filepath


############################### SQL Monitoring
# Create a base class for declarative models
Base = declarative_base()

class PredictionLog(Base):
    """
    SQLAlchemy model for storing prediction logs in PostgreSQL
    """
    __tablename__ = 'prediction_logs'

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    
    # Input features
    gender = Column(String)
    race_ethnicity = Column(String)
    parental_level_of_education = Column(String)
    lunch = Column(String)
    test_preparation_course = Column(String)
    reading_score = Column(Float)
    writing_score = Column(Float)
    
    # Prediction result
    prediction = Column(Float)

# Database Logger Class ######################################
class DatabasePredictionLogger:
    def __init__(self):
        # AWS RDS Postgres Connection Parameters
        self.endpoint = os.getenv('DB_ENDPOINT')
        self.port = os.getenv('DB_PORT', '5432')
        self.user = os.getenv('DB_USER')
        self.password = os.getenv('DB_PASSWORD')
        self.region = os.getenv('DB_REGION')
        self.dbname = os.getenv('DB_NAME')
        
        # Create SQLAlchemy engine
        self.engine = self._create_sqlalchemy_engine()
        
        # Create a session factory
        self.SessionLocal = sessionmaker(bind=self.engine)

    def _create_sqlalchemy_engine(self):
        """
        Create SQLAlchemy engine with flexible authentication
        """
        connection_string = (
            f"postgresql://{self.user}:{self.password}@"
            f"{self.endpoint}:{self.port}/{self.dbname}"
        )
        
        return create_engine(connection_string)

    def connect(self):
        """
        Establish a direct psycopg2 connection with flexible authentication
        """
        try:
            conn = psycopg2.connect(
                host=self.endpoint,
                port=self.port,
                database=self.dbname,
                user=self.user,
                password=self.password
                )
            return conn
        except Exception as e:
            print(f"Database connection failed due to {e}")
            raise

    def log_prediction(self, input_data: PredictionInput, prediction: float) -> None:
        """
        Log prediction details to the PostgreSQL database
        
        Args:
            input_data (PredictionInput): Input features used for prediction
            prediction (float): Predicted value
        """
        try:
            # Create a new session
            session = self.SessionLocal()
            
            # Create a new PredictionLog record
            prediction_log = PredictionLog(
                # Map input features from PredictionInput to PredictionLog columns
                gender=input_data.gender,
                race_ethnicity=input_data.race_ethnicity,
                parental_level_of_education=input_data.parental_level_of_education,
                lunch=input_data.lunch,
                test_preparation_course=input_data.test_preparation_course,
                reading_score=input_data.reading_score,
                writing_score=input_data.writing_score,
                
                # Store the prediction result
                prediction=prediction
            )
            
            # Add the record to the session
            session.add(prediction_log)
            
            # Commit the transaction
            session.commit()
        
        except Exception as e:
            # Rollback the transaction in case of an error
            session.rollback()
            print(f"Error logging prediction: {e}")
            # Optionally, you could log this error or raise it depending on your error handling strategy
            raise
        
        finally:
            # Always close the session
            session.close()
        

    def execute_query(self, query):
        """
        Execute a raw SQL query
        
        :param query: SQL query to execute
        :return: Query results
        """
        try:
            conn = self.connect()
            cur = conn.cursor()
            cur.execute(query)
            results = cur.fetchall()
            cur.close()
            conn.close()
            return results
        except Exception as e:
            print(f"Query execution failed: {e}")
            raise


################################################################################################################################################################

class RegressionComponent:
    def __init__(self, logger = None):
        """
        Initialize RegressionComponent with optional database logger
        
        Args:
            logger (DatabasePredictionLogger, optional): Logger to track predictions
        """
        # Model loading
        self.model = None
        self.model_best_model_path = ModelTrainingArtifacts().best_model_overall_path
        self.load_model()

        # Preprocessor loading
        self.preprocessor = None        
        self.trained_preprocessor_path = DataPreprocessingArtifacts().trained_preprocessor_path
        self.load_preprocessor()
        
        # Prediction logger
        self.prediction_logger = logger

    def load_model(self):
        """
        Load the best trained model from the specified path
        """
        try:
            self.model = joblib.load(self.model_best_model_path)
        except FileNotFoundError:
            raise ValueError(f"Model not found at {self.model_best_model_path}")
        except Exception as e:
            raise RuntimeError(f"Error loading model: {e}")

    def load_preprocessor(self):
        """
        Load the trained preprocessor from the specified path
        """
        try:
            self.preprocessor = joblib.load(self.trained_preprocessor_path)
        except FileNotFoundError:
            raise ValueError(f"Preprocessor not found at {self.trained_preprocessor_path}")
        except Exception as e:
            raise RuntimeError(f"Error loading preprocessor: {e}")

    def predict(self, input_data: Union[PredictionInput, List[PredictionInput]]) -> List[PredictionOutput]:
        
        if isinstance(input_data, PredictionInput):  # Single input
            input_data = [input_data]  # Convert to batch format
            
        if not input_data:
            raise ValueError("Input data is empty.")
        
        # Convert input data to a list of dictionaries
        features_list = [{
            'gender': item.gender,
            'race_ethnicity': item.race_ethnicity,
            'parental_level_of_education': item.parental_level_of_education,
            'lunch': item.lunch,
            'test_preparation_course': item.test_preparation_course,
            'reading_score': item.reading_score,
            'writing_score': item.writing_score
        } for item in input_data]
        
        # Preprocess inputs
        processed_features = self._preprocess(features_list)
        
        # Make predictions
        predictions = self.model.predict(processed_features)
        
        # If predictions are empty or invalid, raise an error
        if predictions is None or len(predictions) == 0:
            raise ValueError("No predictions were returned from the model.")

        # Generate PredictionOutput for each row
        results = [
            PredictionOutput(prediction=pred) for pred in predictions
        ]
        
        # Log predictions using PredictionLogger
        # self.prediction_logger.log_prediction(input_data, results)
        
        return results  # Ensure results are returned


    def _preprocess(self, features: List[Dict[str, Any]]):
        """
        Preprocessing for batch inputs.

        Args:
            features (List[Dict[str, Any]]): Batch of raw input features.

        Returns:
            np.ndarray: Processed features ready for model prediction.
        """
        import pandas as pd
        features_df = pd.DataFrame(features)  # Convert list of dicts to DataFrame
        processed_features = self.preprocessor.transform(features_df)
        return processed_features
