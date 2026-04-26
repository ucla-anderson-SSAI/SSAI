FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies.
# This file is named Dockerfile.cloudrun (not Dockerfile) so Railway's builder
# does NOT auto-detect it. Railway falls back to Nixpacks + requirements.txt
# only (no TensorFlow, no Pillow). deploy-cloudrun.sh renames this to
# Dockerfile at deploy time so `gcloud run deploy --source .` picks it up.
COPY requirements.txt requirements-ml.txt ./
RUN pip install --no-cache-dir -r requirements.txt -r requirements-ml.txt

# Cache MobileNetV2 ImageNet weights at build time so the transfer-learning
# endpoint does not have to download them during a Cloud Run request/cold start.
ENV KERAS_HOME=/app/.keras
RUN mkdir -p "$KERAS_HOME" && python - <<'PYDOCKER'
from tensorflow import keras
keras.applications.MobileNetV2(
    input_shape=(96, 96, 3),
    include_top=False,
    weights="imagenet",
)
print("MobileNetV2 ImageNet weights cached")
PYDOCKER

# Copy application
COPY app.py .
COPY index.html .

# Single worker — Cloud Run handles scaling by spinning up more containers.
# Use shell form so $PORT (injected by Cloud Run, defaults to 8080) is expanded.
CMD exec gunicorn app:app -k uvicorn.workers.UvicornWorker --workers 1 --timeout 600 --bind 0.0.0.0:${PORT:-8080}
