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
COPY app.py .
COPY index.html .

# Run with gunicorn. Cloud Run/Railway provide PORT; 8080 is the fallback.
CMD exec gunicorn app:app --bind 0.0.0.0:${PORT:-8080} --timeout 120 --workers 1 --threads 4
