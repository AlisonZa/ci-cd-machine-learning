import os, sys
import pandas as pd
import numpy as np
import joblib

from src.utils import logger_obj,CustomException

from configurations import pipeline_config_obj
from src.entities import DataValidationArtifacts, DataPreprocessingArtifacts

from sklearn.model_selection import train_test_split

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import  OneHotEncoder, OrdinalEncoder, FunctionTransformer
from sklearn.pipeline import Pipeline

class DataPreprocessing:
    """
    A class to handle data preprocessing tasks such as splitting the data into training and testing sets, 
    creating and applying preprocessing pipelines, and saving the transformed data for model training.

    Attributes:
        data_validation_artifacts: An object that contains paths for validated data, the trained_preprocessor, 
        and some variables that are going to be shared across different methods
        data_preprocessing_artifacts: An object that contains paths for saving preprocessed data.
        feature_definition: Configuration for the features, including categorical and numerical features.

    Methods:
        __init__: Initializes the DataPreprocessing object and creates necessary directories for storing preprocessed data.
        train_test_split_for_regresison: Splits the validated data into training and testing sets.
        train_preprocessing_pipeline: Trains a preprocessing pipeline to handle categorical and numerical features.
        preprocess_data: Applies the trained preprocessing pipeline to the training and testing data.
        save_files: Saves the preprocessed data to disk as numpy arrays.
        run_data_preprocessing: Runs the entire preprocessing workflow.
    """

    def __init__(self):
        """
        Initializes the DataPreprocessing object and creates necessary directories for preprocessed data.

        The method creates directories to store preprocessed data arrays and logs the creation process.
        If any error occurs, it logs the error and raises a CustomException.
        """
        self.data_validation_artifacts = DataValidationArtifacts()
        self.data_preprocessing_artifacts = DataPreprocessingArtifacts()
        self.feature_definition = pipeline_config_obj.feature_definition

        try:
            # Create directories for storing preprocessed data
            logger_obj.info("Creating the data_preprocessing folder")
            os.makedirs(self.data_preprocessing_artifacts.data_preprocessing_root_folder, exist_ok=True)
            os.makedirs(self.data_preprocessing_artifacts.preprocessed_arrays_folder, exist_ok=True)
            logger_obj.info(f"Successfully created the data_preprocessing folder at: {self.data_preprocessing_artifacts.data_preprocessing_root_folder}")
        
        except Exception as e:
            # Log the error and raise an exception if directory creation fails
            logger_obj.error(f"Error during creating the data_preprocessing folder, Error:\n{CustomException(e, sys)}")
            raise CustomException(e, sys)

    def train_test_split_for_regresison(self, test_size=0.2):
        """
        Splits the validated data into training and testing sets.

        The function reads the validated data, separates features (X) and target variable (y),
        and splits the data into training and testing sets.

        Parameters:
        -----------
        test_size : float, optional, default=0.2
            The proportion of the dataset to include in the test split.

        Updates:
        --------
        The `X_train`, `X_test`, `y_train`, and `y_test` attributes of `data_preprocessing_artifacts`.
        """
        try:
            logger_obj.info("Starting the train_test_split")
            validated_df = pd.read_csv(self.data_validation_artifacts.validated_data_path)

            target_column = self.feature_definition.target_column
            X, y = validated_df.drop(columns=[target_column], axis=1), validated_df[target_column]

            # Perform the train-test split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        
            # Store the split data in the artifact class
            self.data_preprocessing_artifacts.X_train = X_train
            self.data_preprocessing_artifacts.X_test = X_test
            self.data_preprocessing_artifacts.y_train = y_train
            self.data_preprocessing_artifacts.y_test = y_test

            logger_obj.info(f"Successfully performed train-test split with test_size={test_size}")
        
        except Exception as e:
            logger_obj.error(f"Error during train_test_split: {CustomException(e, sys)}")
            raise CustomException(e, sys)

    def train_preprocessing_pipeline(self):
        """
        Trains a preprocessing pipeline to handle both categorical and numerical features.

        This method defines and fits transformers for nominal and ordinal categorical features,
        as well as a function transformer for numerical features. The trained pipeline is then saved
        to disk using `joblib`.

        Updates:
        --------
        The `trained_preprocessor` is saved and stored in the `data_preprocessing_artifacts`.
        """
        try:
            # Define the transformers for different feature types
            categorical_nominal_transformer = Pipeline(steps=[
                ('onehot', OneHotEncoder(drop='first'))  # Drop the first category to avoid multicollinearity
            ])

            categorical_ordinal_transformer = Pipeline(steps=[
                ('ordinal', OrdinalEncoder(categories=[["some high school", "high school", "some college", 
                                                        "associate's degree", "bachelor's degree", "master's degree"]]))
            ])

            # Create a column transformer to apply preprocessing steps to each feature group
            preprocessor = ColumnTransformer(
                transformers=[
                    ('cat_nom', categorical_nominal_transformer, self.feature_definition.categorical_nominals),
                    ('cat_ord', categorical_ordinal_transformer, self.feature_definition.categorical_ordinals),
                    ('num', FunctionTransformer(), self.feature_definition.numeric_scalars)  # No change for numeric features
                ])

            # Fit the preprocessor to the training data
            trained_preprocessor = preprocessor.fit(self.data_preprocessing_artifacts.X_train)
            
            # Save the trained preprocessor to disk
            joblib.dump(trained_preprocessor, self.data_preprocessing_artifacts.trained_preprocessor_path)
            self.data_preprocessing_artifacts.trained_preprocessor = trained_preprocessor

            logger_obj.info("Successfully trained and saved the preprocessing pipeline")
        
        except Exception as e:
            logger_obj.error(f"Error during training preprocessing pipeline: {CustomException(e, sys)}")
            raise CustomException(e, sys)

    def preprocess_data(self):
        """
        Applies the trained preprocessing pipeline to the training and testing data.

        This method uses the preprocessor that was trained and saved earlier to transform both the 
        training and testing data.

        Updates:
        --------
        The transformed data is stored in `X_train_transformed` and `X_test_transformed` attributes.
        """
        try:
            logger_obj.info("Applying preprocessing pipeline to data")
            trained_preprocessor = self.data_preprocessing_artifacts.trained_preprocessor
            
            # Transform the training and testing data
            X_train_transformed = trained_preprocessor.transform(self.data_preprocessing_artifacts.X_train)
            X_test_transformed = trained_preprocessor.transform(self.data_preprocessing_artifacts.X_test)

            # Store the transformed data in the artifacts
            self.data_preprocessing_artifacts.X_train_transformed = X_train_transformed
            self.data_preprocessing_artifacts.X_test_transformed = X_test_transformed

            logger_obj.info("Successfully transformed the data using the preprocessing pipeline")
        
        except Exception as e:
            logger_obj.error(f"Error during preprocessing data: {CustomException(e, sys)}")
            raise CustomException(e, sys)

    def save_files(self):
        """
        Saves the preprocessed data to disk as numpy arrays.

        The transformed data (X_train, X_test, y_train, y_test) are saved as numpy arrays to the
        specified paths in the `data_preprocessing_artifacts`.

        Updates:
        --------
        The transformed data is saved as numpy arrays in the appropriate files.
        """
        try:
            logger_obj.info("Saving preprocessed data to disk")
            variables = [self.data_preprocessing_artifacts.X_train_transformed,
                         self.data_preprocessing_artifacts.X_test_transformed,
                         self.data_preprocessing_artifacts.y_train,
                         self.data_preprocessing_artifacts.y_test]
            
            paths = [self.data_preprocessing_artifacts.X_train_transformed_path,
                     self.data_preprocessing_artifacts.X_test_transformed_path,
                     self.data_preprocessing_artifacts.y_train_path,
                     self.data_preprocessing_artifacts.y_test_path]
            
            # Save each variable to its corresponding file path
            for var, path in zip(variables, paths):
                # Convert DataFrames/Series to numpy arrays
                if isinstance(var, (pd.DataFrame, pd.Series)):
                    numpy_array = var.to_numpy()
                elif isinstance(var, np.ndarray):
                    numpy_array = var

                np.save(path, numpy_array)
                logger_obj.info(f"Successfully saved {path}")
        
        except Exception as e:
            logger_obj.error(f"Error during saving preprocessed data: {CustomException(e, sys)}")
            raise CustomException(e, sys)

    def run_data_preprocessing(self):
        """
        Executes the entire data preprocessing pipeline.

        This method runs the train-test split, trains the preprocessing pipeline, transforms the data, 
        and saves the preprocessed data to disk. If any step fails, it logs the error and raises a CustomException.
        """
        try:
            logger_obj.info("Starting data preprocessing")
            self.train_test_split_for_regresison()
            self.train_preprocessing_pipeline()
            self.preprocess_data()
            self.save_files()
            logger_obj.info("Data preprocessing completed successfully")
        
        except Exception as e:
            logger_obj.error(f"Error during data preprocessing: {CustomException(e, sys)}")
            raise CustomException(e, sys)


if __name__ == "__main__":
    data_preprocessing_obj = DataPreprocessing()
    data_preprocessing_obj.run_data_preprocessing()