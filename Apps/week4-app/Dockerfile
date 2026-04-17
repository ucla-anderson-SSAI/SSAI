FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY main.py .
COPY cancer.csv .
COPY index.html .

# Single worker — Cloud Run handles scaling by spinning up more containers.
# Use shell form so $PORT (injected by Cloud Run, defaults to 8080) is expanded.
CMD exec gunicorn main:app -k uvicorn.workers.UvicornWorker --workers 1 --timeout 300 --bind 0.0.0.0:${PORT:-8080}
