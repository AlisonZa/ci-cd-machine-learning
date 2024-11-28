from dataclasses import dataclass
from typing import Type
import json
import os

# import the classes that are going to be configured by the MLConfig class
from configurations.config_entities import DataIngestionConfig, DataValidationConfig, FeatureDefinition, ModelTrainingParams


@dataclass
class MLConfig:
    """
    Configuration class for managing machine learning pipeline settings.

    This class organizes configuration parameters for data ingestion, 
    data validation, feature definition, and model training. It also 
    provides a method to load these configurations from a JSON file.

    Attributes:
        data_ingestion (DataIngestionConfig): Configuration for data ingestion.
        data_validation (DataValidationConfig): Configuration for data validation.
        feature_definition (FeatureDefinition): Configuration for feature definitions.
        model_training (ModelTrainingParams): Configuration for model training.
    """
    data_ingestion: DataIngestionConfig = DataIngestionConfig()
    data_validation: DataValidationConfig = DataValidationConfig()
    feature_definition: FeatureDefinition = FeatureDefinition()
    model_training: ModelTrainingParams = ModelTrainingParams()

    @classmethod
    def from_json(cls: Type['MLConfig'], json_path:str = os.path.join('configurations', 'project_configurations.json')) -> 'MLConfig':
        """
        Load MLConfig from a JSON file.

        Args:
            json_path (str): Path to the JSON file containing the configuration. 
                Defaults to "configurations/project_configurations.json".

        Returns:
            MLConfig: An instance of MLConfig populated with parameters from the JSON file.

        Raises:
            FileNotFoundError: If the specified JSON file does not exist.
            JSONDecodeError: If the file content is not valid JSON.
            KeyError: If required keys are missing in the JSON configuration.
        """
        with open(json_path, 'r') as f:
            config_dict = json.load(f)

        return cls(
            data_ingestion=DataIngestionConfig(**config_dict['data_ingestion']),
            data_validation=DataValidationConfig(**config_dict['data_validation']),
            feature_definition=FeatureDefinition(**config_dict['feature_definition']),
            model_training=ModelTrainingParams(**config_dict['model_training'])
        )

# Instantiate the object that is going to be passed to the .src
pipeline_config_obj  = MLConfig.from_json()
