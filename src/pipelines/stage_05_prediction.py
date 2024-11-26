import numpy as np
import os
from cnnClassifier import logger
from keras.models import load_model
from keras.utils import load_img, img_to_array




STAGE_NAME = "Perform the Predictions"

class PredictionPipeline():
    def __init__(self, filename):
        self.filename =  filename


    def predict(self):
        # load model (alterado para a fase docker, se quiser puxe pelo que foi gerado pela pipeline pode descomentar isso daqui)
        # model = load_model(os.path.join(
        #     "artifacts", "training", "trained_model.h5"
        # )) 
        model = load_model(os.path.join(
            "model", "model.h5"
        ))
        image_name = self.filename
        test_image = load_img(image_name, target_size= (224, 224))
        test_image = img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis = 0)
        result = np.argmax(model.predict(test_image), axis = 1)
        print(result)

        if result[0] == 1:
            prediction = "Tumor"
            return [{"image": prediction}]
        else:
            prediction = "Normal"
            return [{"image": prediction}]

    # The source cames from the component file
#     def main(self):
#         config = ConfigurationManager()
#         eval_config = config.get_evaluation_config()
#         evaluation = Evaluation(eval_config)
#         evaluation.evaluation()
#         evaluation.log_into_mlflow() 
#         # uncomment to log into mlflow
     
# if __name__ == "__main__":
#     try:
#         logger.info(f">>>> stage: {STAGE_NAME} started <<<<")
#         obj = PredictionPipeline()
#         obj.main()
#         logger.info(f">>>> stage: {STAGE_NAME} Completed <<<<\n\n x================x")
#     except Exception as e:
#         logger.exception(e)
#         raise e