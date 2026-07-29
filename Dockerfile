# Use official Python runtime as base image
FROM python:3.14-slim

WORKDIR /app

# Install system dependencies needed for some Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip

# Install dependencies into the container environment
COPY ./requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy your source code
COPY . /app

# Copy the WebGL build (if it exists)
# This assumes your webgl_build folder is in the project root
COPY ./webgl_build /app/webgl_build

# Render passes a dynamic $PORT variable. We run shell format to evaluate it properly.
CMD uvicorn app:app --host 0.0.0.0 --port $PORT
