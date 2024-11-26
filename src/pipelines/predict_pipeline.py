from src.entities import PredictionInput, PredictionOutput
from src.components.prediction import RegressionComponent  

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


# Entrypoint:
if __name__ == "__main__":
    input_data = PredictionInput(
    
    gender='female',
    race_ethnicity='group B',
    parental_level_of_education='some college',
    lunch='standard',
    test_preparation_course='completed',
    reading_score=75,
    writing_score=80
    )
    
    prediction_result = predict_pipeline(input_data)
    print(prediction_result.prediction) 
