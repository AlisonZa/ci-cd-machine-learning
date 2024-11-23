import sys  # Importing the sys module to access system-specific parameters and functions, like exception traceback

# Function to extract detailed error information from an exception
def error_message_detail(error, error_detail: sys):
    # Extracting the exception information (type, value, traceback) using exc_info() from sys
    _, _, exc_tb = error_detail.exc_info()

    # Extracting the filename where the error occurred from the traceback object
    file_name = exc_tb.tb_frame.f_code.co_filename
    # Extracting the line number where the error occurred from the traceback object
    line_number = exc_tb.tb_lineno
    # Converting the error object to a string to capture the error message
    error_message = str(error)

    # Formatting a detailed error message string that includes file name, line number, and the error message
    error_message = f"Error occured in Python script name {file_name} line number: {line_number} error message {error_message}"
    # Returning the formatted error message
    return error_message

# Custom Exception class to raise detailed error messages
class CustomException(Exception):
    # Constructor method for the custom exception
    def __init__(self, error_message, error_details: sys):
        # Calling the parent class constructor to initialize the error message
        super().__init__(error_message)
        # Storing the detailed error message using the error_message_detail function
        self.error_message = error_message_detail(error_message, error_details)
    
    # Overriding the __str__ method to return the custom error message when the exception is printed
    def __str__(self):
        return self.error_message
