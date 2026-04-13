"""
Week 3 (Lunar Lander): Live PPO training with Stable-Baselines3.

Endpoints:
  POST /train/start    : start training in a background thread
  POST /train/stop     : request stop (graceful, on next callback tick)
  POST /train/reset    : wipe state + model so user can start fresh
  GET  /train/status   : current state for polling (curve, timesteps, etc.)
  POST /rollout        : greedy episode against current in-memory model
  GET  /                : health
  GET  /app             : frontend
"""

import os
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from trainer import (
    ACTION_NAMES,
    ENV_ID,
    STATE_DIMS,
    TrainConfig,
    get_manager,
)


app = FastAPI(
    title="Week 3 Lunar Lander · Live PPO",
    description="Stable-Baselines3 PPO with live training and rollout",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============== Models ==============
class StartRequest(BaseModel):
    target_episodes: int = Field(default=2000, ge=50, le=20000)
    n_envs: int = Field(default=8, ge=1, le=32)
    learning_rate: float = Field(default=3e-4, ge=1e-5, le=1e-2)
    n_steps: int = Field(default=1024, ge=64, le=4096)
    batch_size: int = Field(default=64, ge=16, le=512)
    n_epochs: int = Field(default=4, ge=1, le=20)
    gamma: float = Field(default=0.999, ge=0.5, le=0.9999)
    gae_lambda: float = Field(default=0.98, ge=0.5, le=1.0)
    ent_coef: float = Field(default=0.01, ge=0.0, le=0.5)
    seed: int = Field(default=42)


class StatusResponse(BaseModel):
    is_running: bool
    is_done: bool
    is_paused: bool = False
    error: Optional[str]
    timesteps: int
    target_episodes: int
    episode_rewards: List[float]
    rolling_100: List[float]
    last_mean_reward: float
    solved: bool
    solved_at_episode: Optional[int]
    elapsed_seconds: Optional[float]
    model_ready: bool
    action_names: List[str] = ACTION_NAMES


class RolloutRequest(BaseModel):
    seed: Optional[int] = None
    max_steps: int = Field(default=1000, ge=50, le=2000)


class RolloutResponse(BaseModel):
    xs: List[float]
    ys: List[float]
    angles: List[float]
    actions: List[int]
    rewards: List[float]
    total_reward: float
    landed: bool
    outcome: str
    action_names: List[str] = ACTION_NAMES


# ============== Endpoints ==============
@app.get("/health")
def health():
    return {
        "status": "healthy",
        "application": "Week 3 Lunar Lander · Live PPO",
        "env": ENV_ID,
        "action_names": ACTION_NAMES,
    }


@app.post("/train/start")
def train_start(req: StartRequest, session_id: str = Query(...)):
    cfg = TrainConfig(**req.model_dump())
    started = get_manager(session_id).start(cfg)
    if not started:
        raise HTTPException(status_code=409, detail="Training already running.")
    return {"started": True, "config": req.model_dump()}


@app.post("/train/pause")
def train_pause(session_id: str = Query(...)):
    ok = get_manager(session_id).pause()
    return {"pause_requested": ok}


@app.get("/state_dims")
def state_dims():
    return {"dims": STATE_DIMS, "action_names": ACTION_NAMES}


@app.post("/train/reset")
def train_reset(session_id: str = Query(...)):
    ok = get_manager(session_id).reset()
    if not ok:
        raise HTTPException(status_code=409, detail="Cannot reset while training is running.")
    return {"reset": True}


@app.get("/train/status", response_model=StatusResponse)
def train_status(session_id: str = Query(...)):
    return StatusResponse(**get_manager(session_id).state_dict())


@app.post("/rollout", response_model=RolloutResponse)
def rollout(req: RolloutRequest, session_id: str = Query(...)):
    try:
        result = get_manager(session_id).rollout(max_steps=req.max_steps, seed=req.seed)
        return RolloutResponse(**result)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Serve frontend
app.mount("/static", StaticFiles(directory="."), name="static")


@app.get("/")
async def serve_frontend():
    return FileResponse("index.html")


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
