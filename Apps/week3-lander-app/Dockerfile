FROM python:3.11-slim

WORKDIR /app

# Install system dependencies (swig needed for box2d, build-essential for compilation)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    swig \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY main.py .
COPY trainer.py .
COPY index.html .

# Expose default port
ENV PORT=8080

# Run with gunicorn + uvicorn worker, single worker for background training thread
# Use shell form so $PORT is expanded from the environment variable Railway injects
CMD gunicorn -w 1 -k uvicorn.workers.UvicornWorker main:app --bind 0.0.0.0:$PORT --timeout 3600
