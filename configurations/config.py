import tkinter as tk
from tkinter import ttk, messagebox
import os
import json

from configurations.config_entities import DataIngestionConfig, DataValidationConfig, FeatureDefinition, ModelTrainingParams

from dataclasses import dataclass

@dataclass
class MLConfig:
    data_ingestion: DataIngestionConfig = DataIngestionConfig()
    data_validation: DataValidationConfig = DataValidationConfig()
    feature_definition: FeatureDefinition = FeatureDefinition()
    model_training: ModelTrainingParams = ModelTrainingParams()

    @classmethod
    def from_json(cls, json_path: str = os.path.join("configurations", "project_configurations.json")) -> 'MLConfig':
        with open(json_path, 'r') as f:
            config_dict = json.load(f)
        
        return cls(
            data_ingestion=DataIngestionConfig(**config_dict['data_ingestion']),
            data_validation=DataValidationConfig(**config_dict['data_validation']),
            feature_definition=FeatureDefinition(**config_dict['feature_definition']),
            model_training=ModelTrainingParams(**config_dict['model_training'])
        )


class DataclassConfigurationApp:
    def __init__(self, master):
        self.master = master
        master.title("Dataclass Configuration")
        master.geometry("600x800")

        # Create notebook (tabbed interface)
        self.notebook = ttk.Notebook(master)
        self.notebook.pack(expand=True, fill='both', padx=10, pady=10)

        # Create tabs
        self.data_ingestion_frame = self.create_data_ingestion_tab()
        self.data_validation_frame = self.create_data_validation_tab()
        self.feature_definition_frame = self.create_feature_definition_tab()
        self.model_training_frame = self.create_model_training_tab()

        # Add tabs to notebook
        self.notebook.add(self.data_ingestion_frame, text="Data Ingestion")
        self.notebook.add(self.data_validation_frame, text="Data Validation")
        self.notebook.add(self.feature_definition_frame, text="Feature Definition")
        self.notebook.add(self.model_training_frame, text="Model Training")

        # Save button
        self.save_button = tk.Button(master, text="Save Configurations", command=self.save_configurations)
        self.save_button.pack(pady=10)

    def create_data_ingestion_tab(self):
        frame = ttk.Frame(self.notebook)
        
        # Raw Data Path
        tk.Label(frame, text="Raw Data Path:").pack(pady=(10,0))
        self.raw_data_path_entry = tk.Entry(frame, width=50)
        self.raw_data_path_entry.insert(0, os.path.join("raw_data", "students.csv"))
        self.raw_data_path_entry.pack(pady=5)

        return frame

    def create_data_validation_tab(self):
        frame = ttk.Frame(self.notebook)
        
        # Numerical Tolerance
        tk.Label(frame, text="Numerical Tolerance:").pack(pady=(10,0))
        self.numerical_tolerance_entry = tk.Entry(frame, width=10)
        self.numerical_tolerance_entry.insert(0, "1.3")
        self.numerical_tolerance_entry.pack(pady=5)

        # Categorical Tolerance
        tk.Label(frame, text="Categorical Tolerance:").pack(pady=(10,0))
        self.categorical_tolerance_entry = tk.Entry(frame, width=10)
        self.categorical_tolerance_entry.insert(0, "0.2")
        self.categorical_tolerance_entry.pack(pady=5)

        # Reference Statistics Path
        tk.Label(frame, text="Reference Statistics Path:").pack(pady=(10,0))
        self.reference_statistics_entry = tk.Entry(frame, width=50)
        self.reference_statistics_entry.insert(0, os.path.join("schemas", "reference_stats.json"))
        self.reference_statistics_entry.pack(pady=5)

        return frame

    def create_feature_definition_tab(self):
        frame = ttk.Frame(self.notebook)
        
        # Target Column
        tk.Label(frame, text="Target Column:").pack(pady=(10,0))
        self.target_column_entry = tk.Entry(frame, width=30)
        self.target_column_entry.insert(0, "math_score")
        self.target_column_entry.pack(pady=5)

        # Categorical Ordinals
        tk.Label(frame, text="Categorical Ordinals (comma-separated):").pack(pady=(10,0))
        self.categorical_ordinals_entry = tk.Entry(frame, width=50)
        self.categorical_ordinals_entry.insert(0, "parental_level_of_education")
        self.categorical_ordinals_entry.pack(pady=5)

        # Categorical Nominals
        tk.Label(frame, text="Categorical Nominals (comma-separated):").pack(pady=(10,0))
        self.categorical_nominals_entry = tk.Entry(frame, width=50)
        self.categorical_nominals_entry.insert(0, "gender,race_ethnicity,lunch,test_preparation_course")
        self.categorical_nominals_entry.pack(pady=5)

        # Numeric Scalars
        tk.Label(frame, text="Numeric Scalars (comma-separated):").pack(pady=(10,0))
        self.numeric_scalars_entry = tk.Entry(frame, width=50)
        self.numeric_scalars_entry.insert(0, "reading_score,writing_score")
        self.numeric_scalars_entry.pack(pady=5)

        return frame

    def create_model_training_tab(self):
        frame = ttk.Frame(self.notebook)
        
        # Main Scoring Criteria
        tk.Label(frame, text="Main Scoring Criteria:").pack(pady=(10,0))
        self.main_scoring_criteria_entry = tk.Entry(frame, width=30)
        self.main_scoring_criteria_entry.insert(0, "r2_score")
        self.main_scoring_criteria_entry.pack(pady=5)

        # Number of Folds
        tk.Label(frame, text="Number of K-Fold Splits:").pack(pady=(10,0))
        self.number_of_folds_entry = tk.Entry(frame, width=10)
        self.number_of_folds_entry.insert(0, "5")
        self.number_of_folds_entry.pack(pady=5)

        return frame

    def save_configurations(self):
        try:
            # Data Ingestion Config
            data_ingestion_config = DataIngestionConfig(
                raw_data_path=os.path.normpath(self.raw_data_path_entry.get())
            )


            # Data Validation Config
            data_validation_config = DataValidationConfig(
                numerical_tolerance=float(self.numerical_tolerance_entry.get()),
                categorical_tolerance=float(self.categorical_tolerance_entry.get()),
                reference_statistics=self.reference_statistics_entry.get()
            )

            # Feature Definition
            feature_definition = FeatureDefinition(
                target_column=self.target_column_entry.get(),
                categorical_ordinals=self.categorical_ordinals_entry.get().split(','),
                categorical_nominals=self.categorical_nominals_entry.get().split(','),
                numeric_scalars=self.numeric_scalars_entry.get().split(',')
            )

            # Model Training Params
            model_training_params = ModelTrainingParams(
                main_scoring_criteria=self.main_scoring_criteria_entry.get(),
                number_of_folds_kfold=int(self.number_of_folds_entry.get())
            )

            # Create a dictionary to save all configurations
            config_dict = {
                "data_ingestion": vars(data_ingestion_config),
                "data_validation": vars(data_validation_config),
                "feature_definition": vars(feature_definition),
                "model_training": vars(model_training_params)
            }

            # Save to a JSON file
            with open('configurations/project_configurations.json', 'w') as f:
                json.dump(config_dict, f, indent=4)

            messagebox.showinfo("Success", "Configurations saved to project_configurations.json")
                    
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save configurations: {str(e)}")

def main():
    root = tk.Tk()
    app = DataclassConfigurationApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()