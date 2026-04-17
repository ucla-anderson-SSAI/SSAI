# Week 4 Frontend (Railway)

Thin static-file server. Serves `index.html` only. No TensorFlow, no training.

All API calls go to the Cloud Run backend — `window.API_BASE` is hardcoded in `index.html`.

## Architecture

```
User → Railway (this)  →  serves index.html
                             ↓ (browser fetches)
                         Cloud Run (Apps/week4-app)  →  /train, /compare, /dataset_info, ...
```

## Deploy to Railway

1. Point your Railway service's root/source directory at `Apps/week4-frontend`.
2. Railway auto-detects Python via `requirements.txt` (Nixpacks).
3. `railway.json` provides the start command and `/health` healthcheck.
4. Push; Railway builds and deploys.

## Keeping index.html in sync

If you edit `Apps/week4-app/index.html` (e.g. to change UI or update `API_BASE`), copy it here too:

```bash
cp Apps/week4-app/index.html Apps/week4-frontend/index.html
```

Or symlink if you prefer (may not survive all git/deploy flows).

## Why separate from week4-app?

`week4-app` installs TensorFlow + XGBoost (~1GB) and loads them at startup — too heavy for Railway's standard tier. This folder stays tiny so Railway can host it trivially while Cloud Run handles the ML work.
