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

# Default port (Railway overrides via $PORT env var)
ENV PORT=8080

# Use python main.py which reads PORT from env natively — no shell expansion issues
CMD ["python", "main.py"]
