# Railway Deployment Guide - MGMT298D Apps

This guide will help you deploy all 8 week apps to Railway so students can access them via web URLs.

## 🎯 Deployment Strategy

We'll deploy each week as a **separate Railway service**. This gives you:
- Independent deployments (update Week 1 without affecting Week 2)
- Easy troubleshooting
- Clear URLs for students
- Ability to scale specific weeks

## 📋 Prerequisites

1. **GitHub Account** (free)
2. **Railway Account** (free tier available)
   - Sign up at [railway.app](https://railway.app)
   - Connect your GitHub account

## 🚀 Quick Start (Recommended Method)

### Step 1: Prepare GitHub Repository

```bash
# Navigate to your Elective folder
cd /path/to/Elective/Apps

# Initialize git (if not already done)
git init

# Create a .gitignore file
cat > .gitignore << EOF
__pycache__/
*.pyc
.DS_Store
*.log
.env
EOF

# Add all apps
git add .
git commit -m "Add all 8 week apps for Railway deployment"

# Push to GitHub
git remote add origin https://github.com/YOUR_USERNAME/mgmt298d-apps.git
git push -u origin main
```

### Step 2: Deploy Each Week to Railway

For each week (repeat 8 times):

1. **Go to Railway Dashboard**
   - Visit [railway.app](https://railway.app)
   - Click "New Project"

2. **Deploy from GitHub**
   - Select "Deploy from GitHub repo"
   - Choose your `mgmt298d-apps` repository
   - Railway will ask for the root directory

3. **Configure Root Directory**
   - Click "Configure" or "Settings"
   - Set **Root Directory**: `week1-app` (or `week2-app`, etc.)
   - This tells Railway to only deploy that specific week

4. **Set Environment Variables** (if needed)
   - Go to "Variables" tab
   - Add `PORT` = `8000` (Railway usually auto-detects this)

5. **Deploy**
   - Railway will auto-detect Python, install requirements, and deploy
   - Wait 2-3 minutes for first deployment

6. **Get Public URL**
   - Go to "Settings" → "Domains"
   - Click "Generate Domain"
   - You'll get a URL like: `week1-app-production.up.railway.app`

7. **Test the App**
   - Visit: `https://week1-app-production.up.railway.app/app`
   - Verify it works!

### Step 3: Organize Your URLs

Create a simple landing page or share this list with students:

```
Week 1 - Linear Regression: https://week1-app-production.up.railway.app/app
Week 2 - Tree Models: https://week2-app-production.up.railway.app/app
Week 3 - Clustering: https://week3-app-production.up.railway.app/app
Week 4 - Reinforcement Learning: https://week4-app-production.up.railway.app/app
Week 5 - Neural Networks: https://week5-app-production.up.railway.app/app
Week 6 - CNNs: https://week6-app-production.up.railway.app/app
Week 7 - Transformers: https://week7-app-production.up.railway.app/app
Week 8 - LLMs: https://week8-app-production.up.railway.app/app
```

## 💰 Pricing & Cost Estimation

**Free Tier:**
- $5 credit/month
- Good for 1-2 apps with light usage
- Apps sleep after inactivity

**Hobby Plan: $5/month**
- $5 credit + $5 usage included
- Good for 2-4 apps
- No sleeping

**Pro Plan: $20/month**
- $20 credit included
- Good for all 8 apps with moderate usage
- Priority support
- Better performance

**For 70 students across 8 apps:**
- Estimated cost: **$20-30/month** (Pro plan recommended)
- Each app uses ~$2-4/month depending on traffic

## 🔧 Alternative: Railway CLI Method

If you prefer command-line deployment:

```bash
# Install Railway CLI
npm install -g @railway/cli

# Login
railway login

# For each week:
cd week1-app
railway init
railway up
railway domain  # Generate public URL

cd ../week2-app
railway init
railway up
railway domain

# Repeat for all 8 weeks...
```

## 📦 Configuration Files (Optional but Recommended)

Add these files to each week app for better control:

### `railway.toml` (in each week folder)
```toml
[build]
builder = "NIXPACKS"

[deploy]
startCommand = "uvicorn main:app --host 0.0.0.0 --port $PORT"
restartPolicyType = "ON_FAILURE"
restartPolicyMaxRetries = 10
```

### `nixpacks.toml` (in each week folder)
```toml
[phases.setup]
nixPkgs = ["python310"]

[phases.install]
cmds = ["pip install -r requirements.txt"]

[start]
cmd = "uvicorn main:app --host 0.0.0.0 --port $PORT"
```

## 🎓 Student Access Instructions

Share this with your students:

---

**How to Access Week Apps:**

1. Click on the week link provided by your instructor
2. The app will load in your browser (may take 10-15 seconds on first load)
3. No installation needed - everything runs in the browser!

**Troubleshooting:**
- If the app doesn't load, wait 30 seconds and refresh
- Clear your browser cache if you see old data
- Use Chrome or Firefox for best experience

---

## 🐛 Troubleshooting

### Deployment fails
- Check `requirements.txt` is present
- Verify Python version compatibility (3.9-3.11 recommended)
- Check Railway build logs for specific errors

### App crashes after deployment
- Check Railway logs: Dashboard → Deployments → View Logs
- Common issue: Missing environment variables
- Verify GitHub data URLs are accessible

### App is slow
- First load after idle can take 10-15 seconds (cold start)
- Upgrade to Pro plan to keep apps warm
- Consider using Render.com for less-used weeks (free tier)

### Port issues
- Railway automatically sets `$PORT` environment variable
- Make sure your code uses: `port=int(os.getenv("PORT", 8000))`

## 🔄 Updating Apps

When you make changes:

```bash
# Make your changes to any week app
cd week1-app
# Edit files...

# Commit and push to GitHub
git add .
git commit -m "Update week 1 app"
git push

# Railway auto-deploys! (if connected to GitHub)
# Check Railway dashboard for deployment status
```

## 📊 Monitoring Usage

1. **Railway Dashboard**
   - View metrics: CPU, Memory, Network
   - Check deployment logs
   - Monitor costs

2. **Set up alerts**
   - Railway → Project Settings → Notifications
   - Get notified if apps go down

## 🎯 Pro Tips

1. **Deploy in phases**
   - Start with Weeks 1-3 before the course begins
   - Add weeks 4-8 as the course progresses
   - Saves money on unused apps

2. **Use descriptive project names**
   - Name projects: "MGMT298D-Week1", "MGMT298D-Week2", etc.
   - Easy to identify in Railway dashboard

3. **Custom domains** (optional)
   - Buy a domain like `mgmt298d-ai.com`
   - Set up subdomains: `week1.mgmt298d-ai.com`
   - More professional for students

4. **Health checks**
   - Add a `/health` endpoint to each app
   - Monitor app status easily

## 📞 Support

- **Railway Docs**: https://docs.railway.app
- **Railway Discord**: https://discord.gg/railway
- **FastAPI Docs**: https://fastapi.tiangolo.com

## ✅ Checklist

- [ ] GitHub repo created with all 8 apps
- [ ] Railway account connected to GitHub
- [ ] Week 1 app deployed and tested
- [ ] Week 2 app deployed and tested
- [ ] Week 3 app deployed and tested
- [ ] Week 4 app deployed and tested
- [ ] Week 5 app deployed and tested
- [ ] Week 6 app deployed and tested
- [ ] Week 7 app deployed and tested
- [ ] Week 8 app deployed and tested
- [ ] URLs shared with students
- [ ] Monitoring set up
- [ ] Budget alerts configured

---

**Estimated Time:** 2-3 hours for all 8 apps (first time)

**Good luck with your deployment! 🚀**
