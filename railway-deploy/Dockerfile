FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for TensorFlow
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy and install requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Railway uses PORT env variable
CMD uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}
