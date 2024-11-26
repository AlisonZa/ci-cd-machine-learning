import numpy as np
from dataclasses import dataclass
from abc import ABC, abstractmethod
import joblib
from typing import Dict, Any
from src.entities import PredictionInput, PredictionOutput, ModelTrainingArtifacts, DataPreprocessingArtifacts
import pandas as pd
from typing import List, Union


class RegressionComponent:
    def __init__(self):
        # Model loading
        self.model = None
        self.model_best_model_path = ModelTrainingArtifacts().best_model_overall_path
        self.load_model()

        # Preprocessor loading
        self.preprocessor = None        
        self.trained_preprocessor_path = DataPreprocessingArtifacts().trained_preprocessor_path
        self.load_preprocessor()

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
