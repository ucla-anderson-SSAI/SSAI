"""
MGMT298D: Science and Strategy of AI
Unified FastAPI Backend for Railway Deployment
UCLA Anderson School of Management
"""

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse
import os

# Create main app
app = FastAPI(
    title="MGMT298D: Science and Strategy of AI",
    description="Interactive ML/AI Learning Platform - UCLA Anderson",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Import and mount week routers
from api.week1 import router as week1_router
from api.week2 import router as week2_router
from api.week3 import router as week3_router
from api.week4 import router as week4_router
from api.week5 import router as week5_router
from api.week6 import router as week6_router
from api.week7 import router as week7_router
from api.week8 import router as week8_router

# Import and mount assignment routers
from api.assignment1 import router as assignment1_router
from api.assignment2 import router as assignment2_router
from api.assignment3 import router as assignment3_router
from api.assignment4 import router as assignment4_router
from api.assignment5 import router as assignment5_router
from api.assignment6 import router as assignment6_router
from api.assignment7 import router as assignment7_router
from api.assignment8 import router as assignment8_router

# Mount API routers
app.include_router(week1_router, prefix="/api/week1", tags=["Week 1: Linear Regression"])
app.include_router(week2_router, prefix="/api/week2", tags=["Week 2: Tree Models"])
app.include_router(week3_router, prefix="/api/week3", tags=["Week 3: Clustering"])
app.include_router(week4_router, prefix="/api/week4", tags=["Week 4: Reinfortic Learning"])
app.include_router(week5_router, prefix="/api/week5", tags=["Week 5: Neural Networks"])
app.include_router(week6_router, prefix="/api/week6", tags=["Week 6: CNNs"])
app.include_router(week7_router, prefix="/api/week7", tags=["Week 7: Transformers"])
app.include_router(week8_router, prefix="/api/week8", tags=["Week 8: LLMs & Agents"])

app.include_router(assignment1_router, prefix="/api/assignment1", tags=["Assignment 1"])
app.include_router(assignment2_router, prefix="/api/assignment2", tags=["Assignment 2"])
app.include_router(assignment3_router, prefix="/api/assignment3", tags=["Assignment 3"])
app.include_router(assignment4_router, prefix="/api/assignment4", tags=["Assignment 4"])
app.include_router(assignment5_router, prefix="/api/assignment5", tags=["Assignment 5"])
app.include_router(assignment6_router, prefix="/api/assignment6", tags=["Assignment 6"])
app.include_router(assignment7_router, prefix="/api/assignment7", tags=["Assignment 7"])
app.include_router(assignment8_router, prefix="/api/assignment8", tags=["Assignment 8"])

# Serve static files (frontends)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Root endpoint - course homepage
@app.get("/", response_class=HTMLResponse)
async def root():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>MGMT298D: Science and Strategy of AI</title>
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
        <style>
            * { margin: 0; padding: 0; box-sizing: border-box; }
            body { font-family: 'Inter', sans-serif; background: #f5f5f5; min-height: 100vh; }
            .header { background: #2774AE; border-bottom: 3px solid #FFD100; padding: 2rem; text-align: center; }
            .header h1 { color: white; font-size: 1.8rem; margin-bottom: 0.5rem; }
            .header p { color: #b0d4f1; font-size: 0.9rem; }
            .container { max-width: 1000px; margin: 2rem auto; padding: 0 1rem; }
            .section { background: white; border-radius: 12px; padding: 1.5rem; margin-bottom: 1.5rem; box-shadow: 0 2px 8px rgba(0,0,0,0.08); }
            .section h2 { color: #005587; font-size: 1.1rem; margin-bottom: 1rem; border-bottom: 2px solid #FFD100; padding-bottom: 0.5rem; }
            .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 1rem; }
            .card { background: #f8f9fa; border: 1px solid #e0e0e0; border-radius: 8px; padding: 1rem; text-decoration: none; color: inherit; transition: all 0.2s; }
            .card:hover { border-color: #2774AE; box-shadow: 0 4px 12px rgba(39,116,174,0.15); transform: translateY(-2px); }
            .card h3 { color: #2774AE; font-size: 0.9rem; margin-bottom: 0.25rem; }
            .card p { color: #666; font-size: 0.75rem; }
            .footer { text-align: center; padding: 2rem; color: #888; font-size: 0.8rem; }
        </style>
    </head>
    <body>
        <div class="header">
            <h1>MGMT298D: Science and Strategy of AI</h1>
            <p>UCLA Anderson School of Management</p>
        </div>
        <div class="container">
            <div class="section">
                <h2>Weekly Modules</h2>
                <div class="grid">
                    <a href="/week1" class="card"><h3>Week 1</h3><p>Linear Regression & Feature Engineering</p></a>
                    <a href="/week2" class="card"><h3>Week 2</h3><p>Tree Models & XGBoost</p></a>
                    <a href="/week3" class="card"><h3>Week 3</h3><p>Clustering & Collaborative Filtering</p></a>
                    <a href="/week4" class="card"><h3>Week 4</h3><p>Reinforcement Learning</p></a>
                    <a href="/week5" class="card"><h3>Week 5</h3><p>Neural Networks</p></a>
                    <a href="/week6" class="card"><h3>Week 6</h3><p>CNNs & Image Classification</p></a>
                    <a href="/week7" class="card"><h3>Week 7</h3><p>Transformers & Attention</p></a>
                    <a href="/week8" class="card"><h3>Week 8</h3><p>LLMs & AI Agents</p></a>
                </div>
            </div>
            <div class="section">
                <h2>Assignments</h2>
                <div class="grid">
                    <a href="/assignment1" class="card"><h3>Assignment 1</h3><p>Airbnb Price Prediction</p></a>
                    <a href="/assignment2" class="card"><h3>Assignment 2</h3><p>XGBoost Grid Search</p></a>
                    <a href="/assignment3" class="card"><h3>Assignment 3</h3><p>Movie Clustering & CF</p></a>
                    <a href="/assignment4" class="card"><h3>Assignment 4</h3><p>Dynamic Pricing RL</p></a>
                    <a href="/assignment5" class="card"><h3>Assignment 5</h3><p>Fashion-MNIST NN</p></a>
                    <a href="/assignment6" class="card"><h3>Assignment 6</h3><p>CNN Architecture</p></a>
                    <a href="/assignment7" class="card"><h3>Assignment 7</h3><p>Attention Mechanisms</p></a>
                    <a href="/assignment8" class="card"><h3>Assignment 8</h3><p>LLM Parameters</p></a>
                </div>
            </div>
        </div>
        <div class="footer">
            <p>API Documentation: <a href="/docs" style="color: #2774AE;">/docs</a></p>
        </div>
    </body>
    </html>
    """

# Serve individual week/assignment frontends
@app.get("/week{week_num}")
async def serve_week(week_num: int):
    return FileResponse(f"static/week{week_num}/index.html")

@app.get("/assignment{assignment_num}")
async def serve_assignment(assignment_num: int):
    return FileResponse(f"static/assignment{assignment_num}/index.html")

# Health check
@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "app": "MGMT298D",
        "modules": {
            "weeks": 8,
            "assignments": 8
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
