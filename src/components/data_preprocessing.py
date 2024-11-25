import os, sys
from src.utils import logger_obj,CustomException
from src.entities import DataValidationArtifacts, DataPreprocessingArtifacts, FeatureDefinition
import pandas as pd
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import  OneHotEncoder, OrdinalEncoder, FunctionTransformer
from sklearn.pipeline import Pipeline
import joblib
import numpy as np

class DataPreprocessing:
    def __init__(self):
        self.data_validation_artifacts = DataValidationArtifacts()
        self.data_preprocessing_artifacts = DataPreprocessingArtifacts()
        self.feature_definition = FeatureDefinition()

        try:
            logger_obj.info(f"Creating the data_preprocessing folder")

            os.makedirs(self.data_preprocessing_artifacts.data_preprocessing_root_folder, exist_ok= True)
            os.makedirs(self.data_preprocessing_artifacts.preprocessed_arrays_folder, exist_ok= True)

            logger_obj.info(f"Succesfully created the data_preprocessing folder at: \n{self.data_preprocessing_artifacts.data_preprocessing_root_folder}")
        
        except Exception as e:
            logger_obj.error(f"Error during creating the data_preprocessing folder, Error:\n{CustomException(e ,sys)}")
            raise CustomException(e ,sys)

    # def perform_data_ingestion_csv_files(self):
    #     try: 
    #         logger_obj.info(f"Loading the data from:\n{self.data_ingestion_config.raw_data_path}")
    #         raw_dataframe = pd.read_csv(self.data_ingestion_config.raw_data_path)

    #         raw_dataframe.to_csv(self.data_ingestion_artifacts.saved_raw_data_path, index= False)
    #         logger_obj.info(f"Succesfully saved the data to the artifacts folder:\n{self.data_ingestion_artifacts.saved_raw_data_path}")

    #     except Exception as e:
    #         logger_obj.error(f"Error during data_ingestion :\n{CustomException(e ,sys)}")
    #         raise CustomException(e ,sys) 


    def train_test_split_for_regresison(self, test_size = 0.2):

        logger_obj.info(f"Sarting the train_test_split")
        validated_df = pd.read_csv(self.data_validation_artifacts.validated_data_path)

        target_column = self.feature_definition.target_column
        
        X, y = validated_df.drop(columns=[target_column], axis = 1), validated_df[target_column]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= test_size, random_state= 42)
        
        # Assign the values to our artifact class
        self.data_preprocessing_artifacts.X_train = X_train
        self.data_preprocessing_artifacts.X_test = X_test
        self.data_preprocessing_artifacts.y_train = y_train
        self.data_preprocessing_artifacts.y_test = y_test


    def train_preprocessing_pipeline(self):

        # Define your transformers
        categorical_nominal_transformer = Pipeline(steps=[
            ('onehot', OneHotEncoder(drop='first'))  # drop='first' to avoid multicollinearity
        ])

        categorical_ordinal_transformer = Pipeline(steps=[
            ('ordinal', OrdinalEncoder(categories=[["some high school", "high school", "some college", "associate's degree", "bachelor's degree", "master's degree"]]))
        ])

        # Define the column transformer for preprocessing
        preprocessor = ColumnTransformer(
            transformers=[
                ('cat_nom', categorical_nominal_transformer, self.feature_definition.categorical_nominals),
                ('cat_ord', categorical_ordinal_transformer, self.feature_definition.categorical_ordinals),
                ('num', FunctionTransformer(), self.feature_definition.numeric_scalars)  # Keep numeric features unchanged, they are with the same scale
            ])

        # Apply the transformation to the data
        trained_preprocessor = preprocessor.fit(self.data_preprocessing_artifacts.X_train)
        joblib.dump(trained_preprocessor, self.data_preprocessing_artifacts.trained_preprocessor_path)

        self.data_preprocessing_artifacts.trained_preprocessor = trained_preprocessor

    def preprocess_data(self):
        trained_preprocessor = self.data_preprocessing_artifacts.trained_preprocessor
        X_train_transformed = trained_preprocessor.transform(self.data_preprocessing_artifacts.X_train)
        X_test_transformed = trained_preprocessor.transform(self.data_preprocessing_artifacts.X_test)

        self.data_preprocessing_artifacts.X_train_transformed = X_train_transformed
        self.data_preprocessing_artifacts.X_test_transformed = X_test_transformed    

    def save_files(self):
        variables = [self.data_preprocessing_artifacts.X_train_transformed,
                 self.data_preprocessing_artifacts.X_test_transformed,
                 self.data_preprocessing_artifacts.y_train,
                 self.data_preprocessing_artifacts.y_test]
        
        paths = [self.data_preprocessing_artifacts.X_train_transformed_path,
                 self.data_preprocessing_artifacts.X_test_transformed_path,
                 self.data_preprocessing_artifacts.y_train_path,
                 self.data_preprocessing_artifacts.y_test_path]
        
        for var, path in zip(variables, paths):
        
            if isinstance(var, (pd.DataFrame, pd.Series)):
                numpy_array = var.to_numpy()
            elif isinstance(var, np.ndarray):
                numpy_array = var

            np.save(path, numpy_array)


    def run_data_preprocessing(self):
        try:
            self.train_test_split_for_regresison()
            self.train_preprocessing_pipeline()
            self.preprocess_data()
            self.save_files()
        
        except Exception as e:
            logger_obj.error(f"Error during data_preprocessing :\n{CustomException(e ,sys)}")
            raise CustomException(e ,sys)



