import os, sys
from src.utils import logger_obj, CustomException
from src.entities import DataValidationArtifacts, DataIngestionArtifacts, DataValidationConfig
import pandas as pd
import pandera as pa
from schemas.data_structure import data_frame_schema
import json


class DataValidation:
    def __init__(self):
        self.data_ingestion_artifacts = DataIngestionArtifacts()
        self.data_validation_artifacts = DataValidationArtifacts()
        self.data_validation_config = DataValidationConfig()

        try:
            logger_obj.info(f"Creating the data_validation folder")
            os.makedirs(self.data_validation_artifacts.data_validation_root_folder, exist_ok= True)
            logger_obj.info(f"Succesfully created the data_validation folder at: \n{self.data_validation_artifacts.data_validation_root_folder}")
        
        except Exception as e:
            logger_obj.error(f"Error during creating the data_validation folder, Error:\n{CustomException(e ,sys)}")
            raise CustomException(e ,sys)

    def validate_data_schema(self):
        logger_obj.info(f"{'*'*10}Entering the process of validating the Data Schema{'*'*10}\nDataframe path:{self.data_ingestion_artifacts.saved_raw_data_path}")
        try:
            data_frame = pd.read_csv(self.data_ingestion_artifacts.saved_raw_data_path)
            validated_schema_df = data_frame_schema.validate(data_frame)
            logger_obj.info(f"Successfully validated the Dataframe Schema")
            return validated_schema_df
        
        except pa.errors.SchemaError as e:
            logger_obj.error(f"Error during Schema Validation:\n{e}")
            raise e

    def statistical_dataframe_validation(
        self,
        validated_schema_df,
        ) -> bool:
        """
        Validate a DataFrame against reference statistics for data drift detection.
               
        Returns:
        --------
        bool
            True if DataFrame passes validation, False otherwise
        """
        data_frame_to_validate = validated_schema_df
        reference_stats_path = self.data_validation_config.reference_statistics
        numerical_tolerance = self.data_validation_config.numerical_tolerance
        categorical_tolerance = self.data_validation_config.categorical_tolerance
        
        logger_obj.info(f"{'*'*10}Entering the process of checking the Data Drift{'*'*10}\nparameters:\nnumerical_tolerance: {numerical_tolerance}\ncategorical_tolerance: {categorical_tolerance}")
        # Load reference statistics
        
        if isinstance(reference_stats_path, str):
            with open(reference_stats_path, 'r') as f:
                reference_stats = json.load(f)
        else:
            reference_stats = reference_stats_path
        
        # Validate each feature
        for feature, stats in reference_stats.items():
            # Skip if feature not in dataframe
            if feature not in data_frame_to_validate.columns:
                continue
            
            # Numerical feature validation
            if 'mean' in stats:
                feature_data = data_frame_to_validate[feature]
                
                # Check mean and standard deviation
                current_mean = feature_data.mean()
                current_std = feature_data.std()
                
                # Compare mean
                mean_diff = abs(current_mean - stats['mean'])
                if mean_diff > numerical_tolerance * stats['std']:
                    logger_obj.info(f"Error During Data Drift Validation")
                    return False
                    
                
                # Compare standard deviation
                std_ratio = current_std / stats['std']
                if std_ratio < 1/1.5 or std_ratio > 1.5:
                    logger_obj.info(f"Error During Data Drift Validation")
                    return False
            
            # Categorical feature validation
            elif 'value_counts' in stats:
                # Calculate current value counts
                current_value_counts = data_frame_to_validate[feature].value_counts(normalize=True)
                ref_value_counts = stats['value_counts']
                
                # Compare categorical distributions
                for category, ref_proportion in ref_value_counts.items():
                    current_proportion = current_value_counts.get(category, 0)
                    if abs(current_proportion - ref_proportion) > categorical_tolerance:
                        logger_obj.info(f"Error During Data Drift Validation for Categorical Features")
                        return False
        
        logger_obj.info(f"{'*'*10}Succesfully Validated the Data Drift{'*'*10}")
        return True
        
    def run_data_validation(self):
        try:
            validated_schema_df = self.validate_data_schema()
            validation_passed = self.statistical_dataframe_validation(validated_schema_df)
            if validation_passed == True:
                validated_schema_df.to_csv(self.data_validation_artifacts.validated_data_path)
                logger_obj.info(f"Saved the validated dataframe to: {self.data_validation_artifacts.validated_data_path}")

        except Exception as e:
            logger_obj.error(f"Error during data_validation :\n{CustomException(e ,sys)}")
            raise CustomException(e ,sys)

if __name__ == "__main__":
    data_validation_obj = DataValidation()
    data_validation_obj.run_data_validation()