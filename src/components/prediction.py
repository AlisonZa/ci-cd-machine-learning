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
from psycopg2.extras import RealDictCursor, execute_batch
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
        
        # Create table if not exists
        self._create_predictions_table()

    def _connect_to_database(self):
        """
        Establish a connection to the PostgreSQL database.
        """
        # Connection parameters
        connection_params = {
            'DB_USER': self.DB_USER,
            'DB_PASSWORD': self.DB_PASSWORD,
            'DB_HOST': self.DB_HOST,
            'DB_PORT': self.DB_PORT
        }
        
        # Check for missing parameters
        missing_params = [k for k, v in connection_params.items() if not v]
        if missing_params:
            raise ValueError(f"Missing database connection parameters: {', '.join(missing_params)}")

        try:
            # Construct connection parameters
            conn_kwargs = {
                'user': self.DB_USER,
                'password': self.DB_PASSWORD,
                'host': self.DB_HOST,
                'port': self.DB_PORT
            }

            # Attempt connection
            self.conn = psycopg2.connect(**conn_kwargs)
            self.conn.autocommit = True
            self.cursor = self.conn.cursor(cursor_factory=RealDictCursor)
            print("Database connection established successfully.")
            
            # Verify connection
            self.cursor.execute("SELECT 1")
            self.cursor.fetchone()
            
        except psycopg2.Error as e:
            print(f"Database Connection Error: {e}")
            raise

    def _create_predictions_table(self):
        """
        Create predictions log table if it doesn't exist.
        """
        create_table_query = f"""
        CREATE TABLE IF NOT EXISTS {self.log_table_name} (
            id SERIAL PRIMARY KEY,
            timestamp TIMESTAMP WITHOUT TIME ZONE,
            gender TEXT,
            race_ethnicity TEXT,
            parental_level_of_education TEXT,
            lunch TEXT,
            test_preparation_course TEXT,
            reading_score FLOAT,
            writing_score FLOAT,
            prediction FLOAT
        )
        """
        try:
            self.cursor.execute(create_table_query)
            print(f"Table {self.log_table_name} ensured to exist.")
        except Exception as e:
            print(f"Error creating table: {e}")
            raise

    def log_prediction(self, 
                       input_data: Union[List[PredictionInput], PredictionInput], 
                       prediction: Union[List[PredictionOutput], PredictionOutput], 
                       output_file: str = None):
        """
        Log a single or batch prediction to the database.
        
        Args:
            input_data (Union[List[PredictionInput], PredictionInput]): Input feature(s)
            prediction (Union[List[PredictionOutput], PredictionOutput]): Prediction result(s)
            output_file (str, optional): Path to save predictions CSV
        
        Returns:
            str: Path to the predictions CSV file
        """
        # Check if connection is established
        if not self.conn or not self.cursor:
            print("No active database connection. Skipping logging.")
            return output_file

        # Ensure inputs are lists
        if isinstance(input_data, PredictionInput):
            input_data = [input_data]
        
        if isinstance(prediction, PredictionOutput):
            prediction = [prediction]
        
        # Validate input lengths
        if len(input_data) != len(prediction):
            raise ValueError("Input data and prediction lists must have the same length")

        # Prepare database entries and prediction DataFrame
        timestamp = datetime.now()
        database_entries = []
        prediction_data = []

        for inp, pred in zip(input_data, prediction):
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

            # Prepare prediction data for CSV
            prediction_data.append({
                'gender': inp.gender,
                'race_ethnicity': inp.race_ethnicity,
                'parental_level_of_education': inp.parental_level_of_education,
                'lunch': inp.lunch,
                'test_preparation_course': inp.test_preparation_course,
                'reading_score': inp.reading_score,
                'writing_score': inp.writing_score,
                'prediction': int(pred.prediction)
            })

        # Insert into database using batch insert for efficiency
        try:
            insert_query = f"""
            INSERT INTO {self.log_table_name} (
                timestamp, gender, race_ethnicity, parental_level_of_education, 
                lunch, test_preparation_course, reading_score, writing_score, prediction
            ) VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s, %s
            )
            """
            execute_batch(self.cursor, insert_query, database_entries)
            print(f"Inserted {len(database_entries)} prediction log entries to database.")
        except Exception as e:
            print(f"Error inserting predictions to database: {e}")

        # Save predictions to CSV if output file is specified
        if output_file:
            predictions_df = pd.DataFrame(prediction_data)
            predictions_df.to_csv(output_file, index=False)
            print(f"Predictions saved to {output_file}")
            return output_file

        return None

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

