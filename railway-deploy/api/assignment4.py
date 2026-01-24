"""
Assignment 4: RL Dynamic Pricing
Minimal FastAPI Backend - Port 8103

This backend simply serves the index.html file.
All simulation logic runs client-side in JavaScript.
"""

from fastapi import APIRouter, HTTPException
import os

router = APIRouter()

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


@router.get("/")
async def root():
    """Serve the main index.html file"""
    return FileResponse(os.path.join(SCRIPT_DIR, "index.html"))


@router.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "healthy", "assignment": 4, "title": "RL Dynamic Pricing"}


