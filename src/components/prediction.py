import joblib
from typing import Dict, Any
from src.entities import PredictionInput, PredictionOutput, ModelTrainingArtifacts, DataPreprocessingArtifacts
from typing import List, Union
import pandas as pd
import os
import logging
from datetime import datetime
import os
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

import pandas as pd
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Float
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.exc import SQLAlchemyError



import pandas as pd
import sqlalchemy
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

from src.entities import PredictionInput, PredictionOutput, ModelTrainingArtifacts, DataPreprocessingArtifacts



class PredictionLogger:
    """
    A utility class to log model inputs and predictions, locally
    """
    def __init__(self, log_dir='prediction_logs'):
        """
        Initialize the prediction logger
        
        Args:
            log_dir (str): Directory to store prediction logs
        """
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        # Configure logging
        self.logger = logging.getLogger('prediction_logger')
        self.logger.setLevel(logging.INFO)
        
        # Create file handler
        log_file = os.path.join(log_dir, f'predictions_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(levelname)s: %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Add handlers to logger
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        # CSV log for detailed tracking
        self.csv_log_file = os.path.join(log_dir, f'prediction_details_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')
        
    def log_prediction(self, input_data: Union[PredictionInput, List[PredictionInput]], 
                       predictions: List[PredictionOutput]):
        """
        Log predictions with input details
        
        Args:
            input_data (Union[PredictionInput, List[PredictionInput]]): Input features
            predictions (List[PredictionOutput]): Prediction results
        """
        # Ensure input_data is a list
        if isinstance(input_data, PredictionInput):
            input_data = [input_data]
        
        # Prepare log data
        log_data = []
        for inp, pred in zip(input_data, predictions):
            log_entry = {
                'timestamp': datetime.now().isoformat(),
                'gender': inp.gender,
                'race_ethnicity': inp.race_ethnicity,
                'parental_level_of_education': inp.parental_level_of_education,
                'lunch': inp.lunch,
                'test_preparation_course': inp.test_preparation_course,
                'reading_score': inp.reading_score,
                'writing_score': inp.writing_score,
                'prediction': pred.prediction,
                # 'confidence': max(pred.probabilities.values()) if pred.probabilities else None # just for regression
            }
            log_data.append(log_entry)
            
            # Log to text logger
            self.logger.info(f"Prediction: Input={inp}, Prediction={pred.prediction}")
        
        # Save to CSV for detailed tracking
        log_df = pd.DataFrame(log_data)
        
        # Append to CSV (create if not exists)
        if not os.path.exists(self.csv_log_file):
            log_df.to_csv(self.csv_log_file, index=False)
        else:
            log_df.to_csv(self.csv_log_file, mode='a', header=False, index=False)



############################### SQL Monitoring
# Create a base class for declarative models


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
        self.prediction_logger = logger or DatabasePredictionLogger()

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
        """
        Make predictions using the loaded model and preprocessor for single or batch input.
        Includes logging of inputs and predictions.

        Args:
            input_data (Union[PredictionInput, List[PredictionInput]]): Input features for prediction (single or batch).

        Returns:
            List[PredictionOutput]: List of prediction results.
        """
        if isinstance(input_data, PredictionInput):  # Single input
            input_data = [input_data]  # Convert to batch format
        
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
        
        # Generate PredictionOutput for each row
        results = [
            PredictionOutput(
                prediction=pred,
                probabilities={'predicted_value': pred}
            ) for pred in predictions
        ]
        
        # Log predictions
        self.prediction_logger.log_prediction(input_data, results)
        
        return results


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
