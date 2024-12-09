from src.entities import PredictionInput, PredictionOutput
from src.components.prediction import RegressionComponent
from src.components.prediction_logging import PredictionLoggerLocal, DatabasePredictionLogger

import pandas as pd
from typing import Union, List

logger = PredictionLoggerLocal() # Change it to DatabasePredictionLogger if you want just the database logging

def predict_pipeline(input_data: Union[PredictionInput, List[PredictionInput]], logger = logger) -> List[PredictionOutput]:
    """
    Simple prediction pipeline function with optional logging
    
    Args:
        input_data (Union[PredictionInput, List[PredictionInput]]): Input features for prediction
        logger (PredictionLogger, optional): Logger to track predictions
    
    Returns:
        List[PredictionOutput]: Prediction results
    """
    # Ensure input_data is a list
    if isinstance(input_data, PredictionInput):
        input_data = [input_data]
    
    # Create regression component with optional logger
    component = RegressionComponent(logger)
    
    # Run prediction
    prediction_result = component.predict(input_data)
    
    # Log the prediction if logger is available
    if logger:
        try:
            logger.log_prediction(input_data, prediction_result)
        except Exception as e:
            print(f"Logging failed: {e}")
    
    return prediction_result  

def batch_predict_pipeline(input_file: str, output_file: str, logger=None):
    # Load input data
    input_data_df = pd.read_csv(input_file)

    # Convert DataFrame rows to PredictionInput instances
    input_data = [
        PredictionInput(
            gender=row['gender'],
            race_ethnicity=row['race_ethnicity'],
            parental_level_of_education=row['parental_level_of_education'],
            lunch=row['lunch'],
            test_preparation_course=row['test_preparation_course'],
            reading_score=row['reading_score'],
            writing_score=row['writing_score']
        ) for _, row in input_data_df.iterrows()
    ]

    # Create regression component and predict
    component = RegressionComponent(logger)
    prediction_results = component.predict(input_data)

    # Log predictions and save to CSV
    if logger:
        logger.log_prediction(input_data, prediction_results, output_file)
    else:
        # Fallback if no logger is provided
        predictions_df = pd.DataFrame([{
            'prediction': result.prediction
        } for result in prediction_results])
        predictions_df.to_csv(output_file, index=False)

    print(f"Predictions saved to {output_file}")
