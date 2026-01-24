# MGMT298D: Science and Strategy of AI

## Railway Deployment Guide

This is a unified FastAPI application containing all 8 weekly modules and 8 assignments for the UCLA Anderson MGMT298D course.

### Project Structure

```
railway-deploy/
├── main.py              # FastAPI entry point with all routers
├── requirements.txt     # Python dependencies
├── railway.json         # Railway configuration
├── api/                 # Backend API routers
│   ├── week1.py        # Linear Regression
│   ├── week2.py        # Tree Models
│   ├── week3.py        # Clustering
│   ├── week4.py        # Reinforcement Learning
│   ├── week5.py        # Neural Networks
│   ├── week6.py        # CNNs
│   ├── week7.py        # Transformers
│   ├── week8.py        # LLMs & Agents
│   └── assignment*.py  # Assignment backends
├── static/             # Frontend HTML files
│   ├── week1/index.html
│   ├── week2/index.html
│   └── ...
└── data/               # CSV data files
    ├── HMData.csv
    ├── netflix_ratings.csv
    └── listings.csv
```

### Deployment to Railway

#### Option 1: Deploy via GitHub (Recommended)

1. **Push to GitHub:**
   ```bash
   cd railway-deploy
   git init
   git add .
   git commit -m "Initial commit"
   git branch -M main
   git remote add origin https://github.com/YOUR_USERNAME/mgmt298d-app.git
   git push -u origin main
   ```

2. **Connect to Railway:**
   - Go to [railway.app](https://railway.app)
   - Click "New Project" → "Deploy from GitHub repo"
   - Select your repository
   - Railway auto-detects Python and deploys

3. **Set up custom domain (optional):**
   - In Railway dashboard → Settings → Domains
   - Add custom domain like `mgmt298d.yourdomain.com`

#### Option 2: Deploy via Railway CLI

```bash
# Install Railway CLI
npm install -g @railway/cli

# Login and deploy
railway login
railway init
railway up
```

### URLs After Deployment

Once deployed, your app will be available at:

- **Homepage:** `https://your-app.up.railway.app/`
- **Week 1:** `https://your-app.up.railway.app/week1`
- **Assignment 1:** `https://your-app.up.railway.app/assignment1`
- **API Docs:** `https://your-app.up.railway.app/docs`

### Environment Variables (Optional)

Railway will auto-detect most settings. If needed:

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | 8000 | Server port (Railway sets automatically) |

### Scaling for 70+ Students

The Pro plan ($20/mo) handles 70 concurrent users easily. For extra robustness:

1. **Enable horizontal scaling:**
   - Railway Dashboard → Service → Settings → Replicas
   - Set to 2-3 replicas for high availability

2. **Monitor usage:**
   - Railway Dashboard → Metrics
   - Watch CPU/Memory during class sessions

### Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run locally
uvicorn main:app --reload --port 8000

# Visit http://localhost:8000
```

### Troubleshooting

**Build fails:**
- Check `requirements.txt` for version conflicts
- TensorFlow requires Python 3.9-3.11

**Slow cold starts:**
- First request after idle may take 10-15s (TensorFlow loading)
- Pro plan keeps containers warm longer

**Memory issues:**
- If hitting limits, upgrade to Pro plan
- Consider lazy-loading TensorFlow models

### Support

- Railway docs: https://docs.railway.app
- FastAPI docs: https://fastapi.tiangolo.com
