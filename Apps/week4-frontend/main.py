"""
Week 4 Frontend — Railway deployment.

This is a tiny static-file server that serves index.html.
All ML/API work lives on Cloud Run (see Apps/week4-app).
index.html already has `window.API_BASE` pointing at the Cloud Run URL.
"""

import os

from fastapi import FastAPI
from fastapi.responses import FileResponse

app = FastAPI(title="Week 4 Frontend")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


@app.get("/")
async def serve_frontend():
    return FileResponse(os.path.join(BASE_DIR, "index.html"))


@app.get("/health")
async def health():
    return {"status": "healthy", "role": "frontend-only"}


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
