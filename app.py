from flask import Flask, request, render_template, send_file
import pandas as pd

from src.pipelines.predict_pipeline import predict_pipeline
from src.entities import PredictionInput
import os
from flask import Flask, jsonify
from src.pipelines.training_pipeline import run_training_pipeline
from src.utils import logger_obj, CustomException


application = Flask(__name__)
app = application

## Route for a home page
@app.route('/')
def index():
    return render_template('index.html') 

@app.route('/predict', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        # Create PredictionInput instance using form data
        input_data = PredictionInput(
            gender=request.form.get('gender'),
            race_ethnicity=request.form.get('ethnicity'),
            parental_level_of_education=request.form.get('parental_level_of_education'),
            lunch=request.form.get('lunch'),
            test_preparation_course=request.form.get('test_preparation_course'),
            reading_score=int(request.form.get('reading_score')),
            writing_score=int(request.form.get('writing_score'))
        )

        # Use the predict_pipeline function
        prediction_output = predict_pipeline(input_data)  # Pass input_data directly

        # Since it's returning a list, take the first element
        results = prediction_output[0].prediction

        return render_template('home.html', results=results)
    




# Ensure a directory exists for batch predictions
UPLOAD_FOLDER = 'batch_predictions'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@app.route('/batch_predict', methods=['GET', 'POST'])
def batch_predict():
    if request.method == 'GET':
        return render_template('batch_predict.html')
    
    if 'file' not in request.files:
        return render_template('batch_predict.html', error='No file uploaded')
    
    file = request.files['file']
    
    if file.filename == '':
        return render_template('batch_predict.html', error='No selected file')
    
    if file and file.filename.endswith('.csv'):
        # Save uploaded file
        input_filepath = os.path.join(UPLOAD_FOLDER, 'input_batch.csv')
        output_filepath = os.path.join(UPLOAD_FOLDER, 'predictions.csv')
        
        file.save(input_filepath)
        
        # Load input data
        input_data_df = pd.read_csv(input_filepath)

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

        # Predict
        prediction_results = predict_pipeline(input_data)

        # Save predictions to a CSV
        predictions_df = pd.DataFrame([{
            'prediction': int(result.prediction)
        } for result in prediction_results])

        predictions_df.to_csv(output_filepath, index=False)
        
        # Send file for download
        return send_file(output_filepath, as_attachment=True)
    
    return render_template('batch_predict.html', error='Invalid file type. Please upload a CSV.')

# Commented for AWS deployment
# @app.route('/train', methods=['GET','POST'])
# def train_pipeline():
#     """
#     Endpoint to trigger the training pipeline.
#     """
#     try:
#         logger_obj.info("Training pipeline endpoint triggered.")
#         run_training_pipeline()
#         return jsonify({"message": "Training pipeline ran successfully."}), 200
#     except CustomException as ce:
#         logger_obj.error(f"CustomException occurred: {ce}")
#         return jsonify({"error": "An error occurred during the training pipeline.", "details": str(ce)}), 500
#     except Exception as e:
#         logger_obj.error(f"Unhandled exception occurred: {e}")
#         return jsonify({"error": "An unhandled error occurred.", "details": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port = 8080) # port 8080 = for AWS