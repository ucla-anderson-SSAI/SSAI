# Week 3: Lunar Lander · Live PPO

A FastAPI + vanilla-JS app that trains a Stable-Baselines3 **PPO** agent on
Gymnasium's `LunarLander-v3` **live in the browser**. You click Start and
watch the learning curve climb in real time, then click Play to see the
current policy fly.

## What's here

```
week3-lander-app/
├── trainer.py      # TrainingManager: background thread, live state, rollout
├── main.py         # FastAPI: /train/start /train/stop /train/status /rollout
├── index.html      # Frontend with live-polling chart + lander animation
├── requirements.txt
├── Procfile
└── railway.json
```

## Setup (one-time)

`gymnasium[box2d]` needs swig + a C++ compiler:

- **macOS:** `brew install swig`
- **Ubuntu/Debian:** `sudo apt-get install swig build-essential`
- **Windows:** WSL or `conda install -c conda-forge box2d-py`

Then:

```bash
cd Apps/week3-lander-app
pip install -r requirements.txt
```

## Run

```bash
uvicorn main:app --reload --port 8000
# open http://localhost:8000/app
```

Click **Start training**. Expect to see the rolling-100 average climb past
the "solved" line (200) in **roughly 1 to 3 minutes** on an M-series Mac with
the default 8 parallel envs. While it's training, you can hit **Play one
episode** at any time to roll out the current model and watch it improve.

## Why is this fast?

- **PPO instead of DQN**: on-policy actor-critic with much better sample
  efficiency on LunarLander than vanilla DQN.
- **8 vectorized envs**: every neural-net forward pass batches 8 states
  at once, so framework overhead is amortized roughly 8×.
- **Stable-Baselines3**: battle-tested PPO implementation in PyTorch
  with no per-step Python overhead.
- **CPU-only**: the model is too small to benefit from GPU; CPU avoids
  data-transfer overhead and is actually faster here.

## API

| Method | Path             | Notes                                       |
|--------|------------------|---------------------------------------------|
| GET    | `/`              | Health                                       |
| POST   | `/train/start`   | Start background training (returns immediately) |
| POST   | `/train/stop`    | Request graceful stop                        |
| POST   | `/train/reset`   | Wipe state + model (when not running)        |
| GET    | `/train/status`  | Poll for live state, episode rewards, rolling avg |
| POST   | `/rollout`       | Greedy episode against the current model     |
| GET    | `/app`           | Frontend                                     |

## Notes

- The frontend polls `/train/status` every 1 second while training is
  running and stops polling once training finishes.
- "Solved" = rolling-100 average reward ≥ 200 (the official Gymnasium
  threshold for LunarLander).
- The model lives in memory only; there's no checkpoint file. If you
  want one, add `manager.model.save("ppo_lander.zip")` after training.
