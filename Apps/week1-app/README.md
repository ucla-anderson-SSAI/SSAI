# Week 1: Linear Regression Interactive Dashboard

A publication-quality web application for teaching linear regression and feature engineering using H&M sales data.

## Architecture

```
┌─────────────────────────┐         ┌─────────────────────────┐
│   React Frontend        │  ←───→  │   FastAPI Backend       │
│   (index.html)          │  JSON   │   (main.py)             │
│                         │         │                         │
│   • Tailwind CSS        │         │   • scikit-learn        │
│   • Chart.js            │         │   • pandas              │
│   • Inter font          │         │   • LassoCV models      │
└─────────────────────────┘         └─────────────────────────┘
```

## Quick Start (Local Development)

### 1. Start the Backend

```bash
cd backend
pip install -r requirements.txt
python main.py
```

The API will be running at `http://localhost:8000`

### 2. Open the Frontend

Simply open `frontend/index.html` in your browser.

For local development, you can just double-click the file or:
```bash
# macOS
open frontend/index.html

# Linux
xdg-open frontend/index.html

# Windows
start frontend/index.html
```

## Deployment Options

### Option A: Hugging Face (Recommended - Free)

**Backend (Hugging Face Spaces):**
1. Create a new Space at huggingface.co → SDK: "Docker"
2. Upload `backend/main.py` and `backend/requirements.txt`
3. Add a `Dockerfile`:
   ```dockerfile
   FROM python:3.10-slim
   WORKDIR /app
   COPY requirements.txt .
   RUN pip install -r requirements.txt
   COPY main.py .
   CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
   ```
4. Your API will be at: `https://YOUR_USERNAME-week1-api.hf.space`

**Frontend (GitHub Pages):**
1. Update `API_BASE` in `index.html` to your Hugging Face URL
2. Push `frontend/index.html` to a GitHub repo
3. Enable GitHub Pages in repo settings
4. Your app will be at: `https://YOUR_USERNAME.github.io/REPO_NAME`

### Option B: Railway + Vercel (Free Tiers)

**Backend (Railway):**
```bash
cd backend
railway login
railway init
railway up
```

**Frontend (Vercel):**
```bash
cd frontend
vercel
```

### Option C: Single Server (VPS/Cloud)

Run both on one server with nginx:

```nginx
server {
    listen 80;

    location / {
        root /var/www/week1-app/frontend;
        index index.html;
    }

    location /api/ {
        proxy_pass http://localhost:8000/;
    }
}
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Health check |
| `/categories` | GET | List all product categories |
| `/analyze/{product}` | GET | Run all models for a product |

### Example Response

```json
{
  "product": "Dress",
  "n_train": 3135,
  "n_test": 285,
  "models": [
    {
      "model_name": "Model A",
      "features": ["Price"],
      "mae": 39.98,
      "predictions": [...],
      "actuals": [...]
    },
    ...
  ],
  "sales_over_time": {...},
  "improvement_ab": 42.5,
  "improvement_ac": 54.2
}
```

## Customization

### Change the API URL

In `frontend/index.html`, find this line near the top of the script:
```javascript
const API_BASE = "http://localhost:8000";
```

Change it to your deployed backend URL.

### Add Your Branding

The header and footer in `index.html` can be customized. Look for:
- `<header>` section for the top banner
- `<footer>` section for the bottom text

### Modify Colors

The app uses a purple/blue gradient. To change it, find:
```css
.gradient-bg {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
}
```

## Features

- **Interactive product selector** — Choose from 29 H&M categories
- **Real-time model comparison** — See how features improve predictions
- **Publication-quality charts** — Bar charts, scatter plots, line charts
- **Responsive design** — Works on desktop and mobile
- **Animated transitions** — Smooth loading states and hover effects

## Tech Stack

**Frontend:**
- React 18 (via CDN)
- Tailwind CSS
- Chart.js
- Inter font (Google Fonts)

**Backend:**
- FastAPI
- scikit-learn
- pandas/numpy
- uvicorn

## Screenshots

The app includes:
1. **Metric cards** — Training size, test size, best MAE, improvement %
2. **Model selector** — Click to compare Model A/B/C
3. **Bar chart** — Visual MAE comparison
4. **Scatter plot** — Predicted vs actual (updates with selected model)
5. **Line chart** — Sales trajectories over time
6. **Insight boxes** — Key takeaways for students
7. **Business implications** — Real-world applications

---

*MGMT298D: Science and Strategy of AI | UCLA Anderson School of Management*
