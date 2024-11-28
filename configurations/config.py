import tkinter as tk
from tkinter import ttk, messagebox

import os
import json

# import the classes that are going to be configured by the MLConfig class
from configurations.config_entities import DataIngestionConfig, DataValidationConfig, FeatureDefinition, ModelTrainingParams


# To create the Json File that is going to be passed to the class MLConfig, we use the following code:
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
        tk.Label(frame, text="Raw Data Path (input dataset location):").pack(pady=(10,0))
        self.raw_data_path_entry = tk.Entry(frame, width=50)
        self.raw_data_path_entry.insert(0, os.path.join("raw_data", "students.csv"))
        self.raw_data_path_entry.pack(pady=5)

        return frame

    def create_data_validation_tab(self):
        frame = ttk.Frame(self.notebook)
        
        # Numerical Tolerance with validation hint
        tk.Label(frame, text="Numerical Tolerance (0-3):").pack(pady=(10,0))
        self.numerical_tolerance_entry = tk.Entry(frame, width=10)
        self.numerical_tolerance_entry.insert(0, "1.3")
        self.numerical_tolerance_entry.pack(pady=5)
        tk.Label(frame, text="Allowed range: 0 to 3", foreground="gray").pack()

        # Categorical Tolerance with validation hint
        tk.Label(frame, text="Categorical Tolerance (0-1):").pack(pady=(10,0))
        self.categorical_tolerance_entry = tk.Entry(frame, width=10)
        self.categorical_tolerance_entry.insert(0, "0.2")
        self.categorical_tolerance_entry.pack(pady=5)
        tk.Label(frame, text="Allowed range: 0 to 1", foreground="gray").pack()

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
        tk.Label(frame, text="Allowed values: r2_score, mae, rmse, mse", foreground="gray").pack()

        # Number of Folds
        tk.Label(frame, text="Number of K-Fold Splits (5-1000):").pack(pady=(10,0))
        self.number_of_folds_entry = tk.Entry(frame, width=10)
        self.number_of_folds_entry.insert(0, "5")
        self.number_of_folds_entry.pack(pady=5)
        tk.Label(frame, text="Allowed range: 5 to 1000", foreground="gray").pack()

        return frame

    def save_configurations(self):
        try:
            # Validate inputs before saving
            validation_errors = []

            # Numerical Tolerance Validation
            numerical_tolerance = self.numerical_tolerance_entry.get()
            if not validate_numerical_tolerance(numerical_tolerance):
                validation_errors.append("Numerical Tolerance must be between 0 and 10")

            # Categorical Tolerance Validation
            categorical_tolerance = self.categorical_tolerance_entry.get()
            if not validate_categorical_tolerance(categorical_tolerance):
                validation_errors.append("Categorical Tolerance must be between 0 and 3")

            # Main Scoring Criteria Validation
            main_scoring_criteria = self.main_scoring_criteria_entry.get()
            if not validate_main_scoring_criteria(main_scoring_criteria):
                validation_errors.append("Main Scoring Criteria must be one of: r2_score, mae, rmse, mse")

            # Number of Folds Validation
            number_of_folds = self.number_of_folds_entry.get()
            if not validate_number_of_folds(number_of_folds):
                validation_errors.append("Number of Folds must be between 5 and 1000")

            # If any validation errors, show them and stop
            if validation_errors:
                error_message = "Please correct the following errors:\n\n" + "\n".join(validation_errors)
                messagebox.showerror("Validation Error", error_message)
                return

            # Data Ingestion Config
            data_ingestion_config = DataIngestionConfig(
                raw_data_path=os.path.normpath(self.raw_data_path_entry.get())
            )

            # Data Validation Config
            data_validation_config = DataValidationConfig(
                numerical_tolerance=float(numerical_tolerance),
                categorical_tolerance=float(categorical_tolerance),
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
                main_scoring_criteria=main_scoring_criteria,
                number_of_folds_kfold=int(number_of_folds)
            )

            # Create a dictionary to save all configurations
            config_dict = {
                "data_ingestion": vars(data_ingestion_config),
                "data_validation": vars(data_validation_config),
                "feature_definition": vars(feature_definition),
                "model_training": vars(model_training_params)
            }

            # Ensure configurations directory exists
            os.makedirs('configurations', exist_ok=True)

            # Save to a JSON file
            with open('configurations/project_configurations.json', 'w') as f:
                json.dump(config_dict, f, indent=4)

            messagebox.showinfo("Success", "Configurations saved to project_configurations.json")
                    
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save configurations: {str(e)}")


def validate_numerical_tolerance(value):
    """Validate numerical tolerance is between 0 and 3."""
    try:
        float_value = float(value)
        return 0 <= float_value <= 3
    except ValueError:
        return False

def validate_categorical_tolerance(value):
    """Validate categorical tolerance is between 0 and 1."""
    try:
        float_value = float(value)
        return 0 <= float_value <= 1
    except ValueError:
        return False

def validate_main_scoring_criteria(value):
    """Validate main scoring criteria is one of the allowed values."""
    allowed_criteria = ['r2_score', 'mae', 'rmse', 'mse']
    return value in allowed_criteria

def validate_number_of_folds(value):
    """Validate number of folds is between 5 and 1000."""
    try:
        int_value = int(value)
        return 5 <= int_value <= 1000
    except ValueError:
        return False

def main():
    root = tk.Tk()
    app = DataclassConfigurationApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()