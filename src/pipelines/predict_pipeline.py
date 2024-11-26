from src.entities import PredictionInput, PredictionOutput
from src.components.prediction import RegressionComponent  
import pandas as pd
import os

def predict_pipeline(input_data: PredictionInput) -> PredictionOutput:
    """
    Simple prediction pipeline function
    
    Args:
        input_data (PredictionInput): Input features for prediction
    
    Returns:
        PredictionOutput: Prediction results
    """
    # Create regression component
    component = RegressionComponent()
    
    # Run prediction
    prediction_result = component.predict(input_data)
    
    return prediction_result


def batch_predict_pipeline(input_file: str, output_file: str):
    """
    Batch prediction pipeline to handle multiple input rows.

    Args:
        input_file (str): Path to input CSV file with features.
        output_file (str): Path to save predictions as a CSV.
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
    component = RegressionComponent()
    prediction_results = component.predict(input_data)

    # Save predictions to a CSV
    predictions_df = pd.DataFrame([{
        'prediction': int(result.prediction)
    } for result in prediction_results])

    predictions_df.to_csv(output_file, index=False)
    print(f"Predictions saved to {output_file}")


# Entrypoint:
# if __name__ == "__main__":
    
#     print(f"Testing Single Prediction")
    
#     input_data = PredictionInput(
    
#     gender='female',
#     race_ethnicity='group B',
#     parental_level_of_education='some college',
#     lunch='standard',
#     test_preparation_course='completed',
#     reading_score=75,
#     writing_score=80
#     )

#     prediction_result = predict_pipeline(input_data)
#     print(prediction_result[0].prediction)

#     print(f"Testing Batch Prediction")
    
#     input_file, output_file = os.path.join("batch_prediction", "x_input.csv"), os.path.join("batch_prediction", "y_pred.csv") 
#     batch_predict_pipeline(input_file, output_file)


    