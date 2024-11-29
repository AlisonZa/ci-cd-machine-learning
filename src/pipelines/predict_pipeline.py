from src.entities import PredictionInput, PredictionOutput
from src.components.prediction import RegressionComponent, PredictionLogger
import pandas as pd
import os

logger = PredictionLogger() # Change it to PredictionLogger if you want just the local

def predict_pipeline(input_data: PredictionInput, logger = logger) -> PredictionOutput:
    """
    Simple prediction pipeline function with optional logging
    
    Args:
        input_data (PredictionInput): Input features for prediction
        logger (PredictionLogger, optional): Logger to track predictions
    
    Returns:
        PredictionOutput: Prediction results
    """
    # Create regression component with optional logger
    component = RegressionComponent(logger)
    
    # Run prediction
    prediction_result = component.predict(input_data)
    
    return prediction_result

def batch_predict_pipeline(input_file: str, output_file: str, logger = logger):
    """
    Batch prediction pipeline to handle multiple input rows with logging.

    Args:
        input_file (str): Path to input CSV file with features.
        output_file (str): Path to save predictions as a CSV.
        logger (PredictionLogger, optional): Logger to track predictions
    """

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

    # Save predictions to a CSV
    predictions_df = pd.DataFrame([{
        'prediction': int(result.prediction)
    } for result in prediction_results])

    predictions_df.to_csv(output_file, index=False)
    print(f"Predictions saved to {output_file}")


# Entrypoint:
# if __name__ == "__main__":
    
    # # Optional: Create a custom logger with a specific log directory
    # custom_logger = PredictionLogger(log_dir='my_custom_prediction_logs')
    
    # # Single prediction with logging
    # input_data = PredictionInput(
    #     gender='female',
    #     race_ethnicity='group B',
    #     parental_level_of_education='some college',
    #     lunch='standard',
    #     test_preparation_course='completed',
    #     reading_score=75,
    #     writing_score=80
    # )

    # prediction_result = predict_pipeline(input_data, logger=custom_logger)
    # print(prediction_result[0].prediction)

    # Batch prediction with logging
    # input_file = os.path.join("batch_prediction", "x_input.csv")
    # output_file = os.path.join("batch_prediction", "y_pred.csv")
    # batch_predict_pipeline(input_file, output_file, logger=custom_logger)