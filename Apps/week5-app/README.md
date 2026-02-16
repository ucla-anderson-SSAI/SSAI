# Week 6 CNN Backend (Python/Keras)

Real CNN training on CIFAR-10 using Keras/TensorFlow.

## Local Development

```bash
cd backend
pip install -r requirements.txt
python app.py
```

Server runs on http://localhost:5000

## Deploy to Railway

1. Create a new Railway project
2. Connect your GitHub repo or use `railway up` CLI
3. Railway will auto-detect the Python app and use the Procfile
4. Get the deployment URL and update `API_BASE_URL` in index.html

## API Endpoints

- `GET /api/health` - Health check
- `POST /api/train` - Start training (body: config JSON)
- `GET /api/train/<session_id>` - Get training status/results
- `POST /api/predict` - Make prediction with trained model

## Config Options

```json
{
  "convBlocks": 3,
  "filters": 32,
  "kernelSize": 3,
  "batchNorm": true,
  "dropout": 0.25,
  "epochs": 20,
  "numSamples": 1000
}
```

`numSamples` can be 100, 1000, or 10000 (balanced across all 10 classes).
