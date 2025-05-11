# backend_service/Dockerfile
FROM python:3.10-slim

WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the backend application code and model artifacts
COPY . .

# Expose the port the app runs on (Gunicorn will use this)
# Render will automatically set the PORT environment variable.
# We don't strictly need to EXPOSE here as Render handles it, but it's good practice.
# EXPOSE $PORT 

# Command to run the application using Gunicorn
# Render provides the $PORT variable.
# The number of workers can be adjusted.
CMD gunicorn --bind "0.0.0.0:$PORT" app:app
