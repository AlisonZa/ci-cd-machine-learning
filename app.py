from flask import Flask, request, render_template
import pandas as pd

from src.pipelines.predict_pipeline import predict_pipeline
from src.entities import PredictionInput


application = Flask(__name__)
app = application

## Route for a home page
@app.route('/')
def index():
    return render_template('index.html') 

@app.route('/predic', methods=['GET','POST'])
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
        prediction_output = predict_pipeline([input_data])
        
        # Since it's returning a list, take the first element
        results = prediction_output[0].prediction

        return render_template('home.html', results=results)
    
if __name__ == "__main__":
    app.run(host="0.0.0.0")