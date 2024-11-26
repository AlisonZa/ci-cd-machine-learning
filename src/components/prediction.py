import numpy as np
from dataclasses import dataclass
from abc import ABC, abstractmethod
import joblib
from typing import Dict, Any
from src.entities import PredictionInput, PredictionOutput, ModelTrainingArtifacts, DataPreprocessingArtifacts
import pandas as pd


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


    def predict(self, input_data: PredictionInput) -> PredictionOutput:
        """
        Make predictions using the loaded model and preprocessor
        
        Args:
            input_data (PredictionInput): Input features for prediction
        
        Returns:
            PredictionOutput: Prediction results
        """
        # Convert input to dictionary
        features_dict = {
            'gender': input_data.gender,
            'race_ethnicity': input_data.race_ethnicity,
            'parental_level_of_education': input_data.parental_level_of_education,
            'lunch': input_data.lunch,
            'test_preparation_course': input_data.test_preparation_course,
            'reading_score': input_data.reading_score,
            'writing_score': input_data.writing_score
        }
        
        # Preprocess input
        processed_features = self._preprocess(features_dict)
        
        # Make prediction
        prediction = self.model.predict(processed_features)

        return PredictionOutput(
            prediction=prediction[0],
            probabilities={
                'predicted_value': prediction[0]
            }
        )

    def _preprocess(self, features: Dict[str, Any]):
        """
        Preprocessing specific to this component

        Args:
            features (Dict[str, Any]): Raw input features

        Returns:
            np.ndarray: Processed features ready for model prediction
        """
        # Convert features to DataFrame for compatibility
        features_df = pd.DataFrame([features])  # Convert single dictionary to DataFrame
        # Transform features using the preprocessor
        processed_features = self.preprocessor.transform(features_df)
        return processed_features
