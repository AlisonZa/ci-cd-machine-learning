# Use an official Python runtime as the base image
FROM python:3.10-slim-buster

# Set the working directory inside the container
WORKDIR /app

# Copy the entire project directory to the container's /app directory
COPY . /app

# Install the required Python packages from requirements.txt
RUN pip install -r requirements.txt

# Specify the default command to run the application
CMD ["python3", "app.py"]
