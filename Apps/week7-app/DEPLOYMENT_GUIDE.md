# Week 7 Transformer Training App - Deployment Guide

## 🚀 Ready to Deploy to Railway!

Your transformer training app is fully configured and ready to go. Here's what you have:

## ✨ Final Configuration

### Backend (app.py)
- **Dataset**: Yelp reviews (5-star ratings)
- **Concurrent trainings**: 30 (optimized for Pro plan)
- **Training samples**: 100, 500, 1000 options
- **Embedding dimensions**: 64, 128, 256
- **Default**: 128d embeddings, 500 samples

### Frontend (index.html)
- **Beautiful 3D architecture diagram** (matching Week 6 style)
- **Mini-batch visualizer** (real-time batch processing)
- **Training & test sample displays** (8 examples each with ⭐ ratings)
- **Simplified UI** (no cluttered dimension text)

## 📋 Deployment Steps

### Option 1: Deploy via Railway Dashboard (Recommended)

1. **Go to Railway.app** → Create New Project
2. **Connect your GitHub repository** (or upload files)
3. **Railway will auto-detect** the Python app
4. **Environment variables** (Railway sets these automatically):
   - `PORT` - Railway assigns this
   - No other config needed!
5. **Click Deploy** ✅

### Option 2: Deploy via Railway CLI

```bash
# Install Railway CLI (if not already)
npm i -g @railway/cli

# Login to Railway
railway login

# Initialize project
railway init

# Deploy
railway up

# Open in browser
railway open
```

## 🎯 Expected Resource Usage

### Railway Pro Plan ($20/month)
```
30 concurrent trainings capacity
Memory: ~1.4 GB peak (17% of 8GB)
CPU: Distributed across 8 vCPUs
Cost: ~$0.60-2/month in compute
```

Perfect for classroom of 30+ students!

## 📊 What Students Will See

1. **Training Examples** - 8 random Yelp reviews with star ratings
2. **Architecture Diagram** - Beautiful 3D visualization
3. **Live Training** - Mini-batch progress bars
4. **Test Predictions** - 8 model predictions with ✓/✗ indicators

## ⚙️ Configuration Options

Students can adjust:
- Training samples: 100, 500, or 1000
- Epochs: 4, 6, 8, or 10
- Embedding dimension: 64, 128, or 256
- Attention heads: 2, 4, or 8
- Transformer blocks: 1, 2, 3, or 4

## 🔍 Expected Training Times

| Samples | Epochs | Time     |
|---------|--------|----------|
| 100     | 8      | ~20 sec  |
| 500     | 8      | ~2 min   |
| 1000    | 8      | ~3 min   |

Perfect for live demos!

## 📝 Post-Deployment Checklist

After deployment:

1. ✅ Visit your Railway URL
2. ✅ Check "Server ready" status shows green
3. ✅ Verify training examples load (8 Yelp reviews)
4. ✅ Test a quick training run (100 samples, 4 epochs)
5. ✅ Confirm predictions show star ratings with ⭐

## 🐛 Troubleshooting

**If deployment fails:**
- Check Railway logs: `railway logs`
- Verify requirements.txt has all dependencies
- Ensure Python 3.9+ is available

**If training is slow:**
- Expected: 100 samples = 20 sec
- If slower: Check server stats for queue position
- 30 concurrent limit prevents overload

**If samples don't load:**
- Backend may still be downloading Hugging Face dataset
- First load takes ~30 seconds to download Yelp data
- Subsequent loads are instant (cached)

## 🎉 Success Indicators

Your deployment is successful when:
- ✅ Homepage loads with UCLA branding
- ✅ Architecture diagram shows 3D blocks
- ✅ Training examples display 8 Yelp reviews
- ✅ Training completes in ~20 sec - 3 min
- ✅ Test predictions show ⭐ ratings

## 📦 Files Being Deployed

```
week7-app/
├── app.py              (Backend - Yelp dataset, 30 concurrent)
├── index.html          (Frontend - 3D viz, samples display)
├── requirements.txt    (Dependencies including datasets)
├── Procfile           (Railway config)
└── railway.json       (Railway settings)
```

## 💰 Cost Monitoring

After deployment, monitor in Railway dashboard:
- Memory usage (should be ~1.4 GB max)
- Active training sessions (0-30)
- Monthly spend (should be $0.60-2)

## 🎓 For Your Students

Share this URL format:
```
https://your-app-name.up.railway.app
```

Students can:
1. See live training examples
2. Configure their own model
3. Train in 20 sec - 3 min
4. View predictions on real Yelp reviews
5. Experiment with architecture changes

## 🚀 You're Ready!

Everything is configured perfectly:
- ✅ Yelp dataset integration
- ✅ 30 concurrent training capacity
- ✅ Beautiful 3D visualizations
- ✅ Mini-batch progress tracking
- ✅ Star rating predictions
- ✅ Optimized for cost (~$0.60-2/month)

**Click "Deploy" in Railway and you're live! 🎉**
