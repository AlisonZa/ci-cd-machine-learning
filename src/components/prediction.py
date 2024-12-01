import joblib
import os

import logging
from typing import Dict, Any, List, Union
from dotenv import load_dotenv
load_dotenv()
import os
import logging
from datetime import datetime
from typing import List
import os
from dotenv import load_dotenv

import pandas as pd
import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv
import os
from datetime import datetime
from typing import Union, List
import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from src.entities import PredictionInput, PredictionOutput, ModelTrainingArtifacts, DataPreprocessingArtifacts

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

import os
import sys
from datetime import datetime
from typing import Union, List
import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class DatabasePredictionLogger:
    def __init__(self, log_table_name='prediction_logs'):
        """
        Initialize the database prediction logger.
        
        Args:
            log_table_name (str, optional): Name of the database table to log predictions. 
                                            Defaults to 'prediction_logs'.
        """
        # Database connection parameters
        # self.DB_NAME = os.getenv('DB_NAME')
        self.DB_USER = os.getenv('DB_USER')
        self.DB_PASSWORD = os.getenv('DB_PASSWORD')
        self.DB_HOST = os.getenv('DB_ENDPOINT')
        self.DB_PORT = os.getenv('DB_PORT')
        
        # Table name
        self.log_table_name = log_table_name
        
        # Connection and cursor attributes
        self.conn = None
        self.cursor = None
        
        # Attempt to connect to database
        self._connect_to_database()

    def _connect_to_database(self):
        """
        Establish a connection to the PostgreSQL database.
        Provides detailed error logging and connection validation.
        """
        # Validate all required environment variables
        connection_params = {
            # 'DB_NAME': self.DB_NAME,
            'DB_USER': self.DB_USER,
            'DB_PASSWORD': self.DB_PASSWORD,
            'DB_HOST': self.DB_HOST,
            'DB_PORT': self.DB_PORT
        }
        
        # Check for missing or empty connection parameters
        missing_params = [k for k, v in connection_params.items() if not v]
        if missing_params:
            error_msg = f"Missing database connection parameters: {', '.join(missing_params)}"
            print(f"ERROR: {error_msg}")
            return

        try:
            # Construct connection parameters dictionary
            conn_kwargs = {
                'user': self.DB_USER,
                'password': self.DB_PASSWORD,
                'host': self.DB_HOST,
                'port': self.DB_PORT
            }
            
            # Add database name if it's not None or empty
            # if self.DB_NAME:
            #     conn_kwargs['database'] = self.DB_NAME

            # Attempt connection
            self.conn = psycopg2.connect(**conn_kwargs)
            self.conn.autocommit = True
            self.cursor = self.conn.cursor(cursor_factory=RealDictCursor)
            print("Database connection established successfully.")
            
            # Verify connection with a simple query
            self.cursor.execute("SELECT 1")
            self.cursor.fetchone()
            
        except psycopg2.Error as e:
            print(f"Database Connection Error: {e}")
            print("Detailed Connection Parameters:")
            for k, v in conn_kwargs.items():
                print(f"{k}: {'*' * len(str(v)) if 'password' in k.lower() else v}")
            
            # Optionally, you can re-raise the exception if you want it to stop execution
            # raise

    def log_prediction(self, 
                       input_data: Union[List[PredictionInput], PredictionInput], 
                       prediction: Union[List[PredictionOutput], PredictionOutput]):
        """
        Log a single or batch prediction to the database.
        
        Args:
            input_data (Union[List[PredictionInput], PredictionInput]): Input feature(s)
            prediction (Union[List[PredictionOutput], PredictionOutput]): Prediction result(s)
        """
        # Check if connection is established
        if not self.conn or not self.cursor:
            print("No active database connection. Skipping logging.")
            return

        # Ensure inputs are lists
        if isinstance(input_data, PredictionInput):
            input_data = [input_data]
        
        if isinstance(prediction, PredictionOutput):
            prediction = [prediction]
        
        # Validate input lengths
        if len(input_data) != len(prediction):
            raise ValueError("Input data and prediction lists must have the same length")

        # Prepare database entries
        database_entries = []
        for inp, pred in zip(input_data, prediction):
            timestamp = datetime.now()
            
            # Prepare database entry
            database_entry = (
                timestamp,
                inp.gender,
                inp.race_ethnicity,
                inp.parental_level_of_education,
                inp.lunch,
                inp.test_preparation_course,
                inp.reading_score,
                inp.writing_score,
                pred.prediction
            )
            database_entries.append(database_entry)

        # Insert into database
        try:
            insert_query = f"""
            INSERT INTO {self.log_table_name} (
                timestamp, gender, race_ethnicity, parental_level_of_education, 
                lunch, test_preparation_course, reading_score, writing_score, prediction
            ) VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s, %s
            )
            """
            self.cursor.executemany(insert_query, database_entries)
            print(f"Inserted {len(database_entries)} prediction log entries to database.")
        except Exception as e:
            print(f"Error inserting predictions to database: {e}")
            # Optionally log to a file or send an alert

    def close_connection(self):
        """
        Close the database connection.
        """
        try:
            if self.cursor:
                self.cursor.close()
            if self.conn:
                self.conn.close()
            print("Database connection closed.")
        except Exception as e:
            print(f"Error closing database connection: {e}")
        finally:
            self.cursor = None
            self.conn = None

    def __del__(self):
        """
        Ensure database connection is closed when object is deleted.
        """
        self.close_connection()


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

