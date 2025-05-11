# backend_service/Dockerfile
FROM python:3.10-slim

# Set a working directory inside the image
WORKDIR /deploy_space 

# Copy the backend_service's requirements.txt into the WORKDIR
COPY ./backend_service/requirements.txt /deploy_space/requirements.txt

# Install dependencies
RUN pip install --no-cache-dir -r /deploy_space/requirements.txt

# Copy the entire backend_service directory (from the build context which is project root)
# into a directory named 'backend_service' INSIDE the WORKDIR.
# So, you'll have /deploy_space/backend_service/app.py, etc.
COPY ./backend_service /deploy_space/backend_service

# Set PYTHONPATH: Add /deploy_space to Python's import search path.
# This allows Python to find the 'backend_service' package inside /deploy_space.
ENV PYTHONPATH="/deploy_space:${PYTHONPATH}"

# Command to run Gunicorn:
# Gunicorn will run from /deploy_space (due to WORKDIR).
# It needs to load the 'app' object from the 'app' module
# within the 'backend_service' package.
CMD gunicorn --bind "0.0.0.0:$PORT" --timeout 120 backend_service.app:app
