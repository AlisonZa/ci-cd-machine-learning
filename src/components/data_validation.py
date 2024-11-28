import os, sys
import pandas as pd
import pandera as pa
import json

from schemas.data_structure import data_frame_schema
from src.utils import logger_obj, CustomException

from configurations import pipeline_config_obj
from src.entities import DataValidationArtifacts, DataIngestionArtifacts


class DataValidation:
    """
    A class to validate ingested data, checking its schema and ensuring it conforms to statistical 
    reference data to detect any data drift.

    Attributes:
        data_ingestion_artifacts: An object that contains paths related to the ingested raw data.
        data_validation_artifacts: An object that contains paths for saving validated data.
        data_validation_config: Configuration object containing paths and parameters for data validation.

    Methods:
        __init__: Initializes the DataValidation object and creates necessary folders for storing validated data.
        validate_data_schema: Validates the schema of the ingested data against the expected schema.
        statistical_dataframe_validation: Validates the ingested data against reference statistics for data drift.
        run_data_validation: Runs the complete data validation process and saves the validated data if successful.
    """

    def __init__(self):
        """
        Initializes the DataValidation object and creates the necessary folder to store validated data.

        The folder is created using the path defined in the data_validation_artifacts object.
        If the folder creation fails, an exception is logged and raised.
        """
        self.data_ingestion_artifacts = DataIngestionArtifacts()
        self.data_validation_artifacts = DataValidationArtifacts()
        self.data_validation_config = pipeline_config_obj.data_validation

        try:
            # Attempt to create the data_validation folder if it doesn't exist
            logger_obj.info("Creating the data_validation folder")
            os.makedirs(self.data_validation_artifacts.data_validation_root_folder, exist_ok=True)
            logger_obj.info(f"Successfully created the data_validation folder at: \n{self.data_validation_artifacts.data_validation_root_folder}")
        
        except Exception as e:
            # Log any error that occurs during folder creation
            logger_obj.error(f"Error during creating the data_validation folder, Error:\n{CustomException(e, sys)}")
            raise CustomException(e, sys)

    def validate_data_schema(self):
        """
        Validates the schema of the ingested data.

        Loads the raw data and checks if it matches the expected schema.

        Returns:
        --------
        pandas.DataFrame
            The validated DataFrame if the schema is correct.
        
        Raises:
        -------
        pa.errors.SchemaError
            If the schema validation fails.
        """
        logger_obj.info(f"{'*'*10}Entering the process of validating the Data Schema{'*'*10}\nDataframe path:{self.data_ingestion_artifacts.saved_raw_data_path}")
        try:
            # Load the raw data for schema validation
            data_frame = pd.read_csv(self.data_ingestion_artifacts.saved_raw_data_path)
            validated_schema_df = data_frame_schema.validate(data_frame)  # Assuming `data_frame_schema` is defined elsewhere
            logger_obj.info("Successfully validated the Dataframe Schema")
            return validated_schema_df
        
        except pa.errors.SchemaError as e:
            # Log and raise schema validation errors
            logger_obj.error(f"Error during Schema Validation:\n{e}")
            raise e

    def statistical_dataframe_validation(self, validated_schema_df) -> bool:
        """
        Validates the ingested data against reference statistics to detect data drift.

        Compares the current data's statistical properties (mean, standard deviation, value counts) 
        with the reference statistics stored in a file or provided directly.

        Parameters:
        -----------
        validated_schema_df : pandas.DataFrame
            The DataFrame that passed schema validation.

        Returns:
        --------
        bool
            True if the DataFrame passes the statistical validation, False otherwise.
        """
        # Retrieve configuration parameters for validation
        data_frame_to_validate = validated_schema_df
        reference_stats_path = self.data_validation_config.reference_statistics
        numerical_tolerance = self.data_validation_config.numerical_tolerance
        categorical_tolerance = self.data_validation_config.categorical_tolerance
        
        logger_obj.info(f"{'*'*10}Entering the process of checking the Data Drift{'*'*10}\nparameters:\nnumerical_tolerance: {numerical_tolerance}\ncategorical_tolerance: {categorical_tolerance}")
        
        # Load reference statistics for comparison
        if isinstance(reference_stats_path, str):
            with open(reference_stats_path, 'r') as f:
                reference_stats = json.load(f)
        else:
            reference_stats = reference_stats_path
        
        # Validate the features in the dataframe
        for feature, stats in reference_stats.items():
            # Skip feature if not present in the dataframe
            if feature not in data_frame_to_validate.columns:
                continue
            
            # Numerical feature validation
            if 'mean' in stats:
                feature_data = data_frame_to_validate[feature]
                
                # Check mean and standard deviation for data drift
                current_mean = feature_data.mean()
                current_std = feature_data.std()
                
                # Compare mean and check if it is within tolerance
                mean_diff = abs(current_mean - stats['mean'])
                if mean_diff > numerical_tolerance * stats['std']:
                    logger_obj.info("Error During Data Drift Validation")
                    return False
                    
                # Compare standard deviation and check if it is within acceptable ratio
                std_ratio = current_std / stats['std']
                if std_ratio < 1/1.5 or std_ratio > 1.5:
                    logger_obj.info("Error During Data Drift Validation")
                    return False
            
            # Categorical feature validation
            elif 'value_counts' in stats:
                # Calculate and compare current value counts with reference proportions
                current_value_counts = data_frame_to_validate[feature].value_counts(normalize=True)
                ref_value_counts = stats['value_counts']
                
                for category, ref_proportion in ref_value_counts.items():
                    current_proportion = current_value_counts.get(category, 0)
                    if abs(current_proportion - ref_proportion) > categorical_tolerance:
                        logger_obj.info("Error During Data Drift Validation for Categorical Features")
                        return False
        
        logger_obj.info(f"{'*'*10}Successfully Validated the Data Drift{'*'*10}")
        return True

    def run_data_validation(self):
        """
        Runs the complete data validation process: schema validation followed by 
        statistical validation.

        If the validation passes, the validated data is saved to the specified location.

        Raises:
        -------
        CustomException
            If any error occurs during the validation process.
        """
        try:
            # Perform schema validation
            validated_schema_df = self.validate_data_schema()
            # Perform statistical validation
            validation_passed = self.statistical_dataframe_validation(validated_schema_df)
            if validation_passed:
                # Save the validated dataframe to the specified path
                validated_schema_df.to_csv(self.data_validation_artifacts.validated_data_path)
                logger_obj.info(f"Saved the validated dataframe to: {self.data_validation_artifacts.validated_data_path}")

        except Exception as e:
            # Log and raise any errors encountered during validation
            logger_obj.error(f"Error during data_validation :\n{CustomException(e, sys)}")
            raise CustomException(e, sys)

if __name__ == "__main__":
    data_validation_obj = DataValidation()
    data_validation_obj.run_data_validation()