# 🚀 Railway Deployment - Quick Start

**Goal:** Get all 8 week apps live on the web in ~2 hours

## ⚡ Fast Track (Step-by-Step)

### 1️⃣ Push to GitHub (15 min)

```bash
# Navigate to Apps folder
cd /path/to/Elective/Apps

# Initialize git
git init
git add .
git commit -m "Initial commit - All 8 week apps"

# Create repo on GitHub.com
# Then push:
git remote add origin https://github.com/YOUR_USERNAME/mgmt298d-apps.git
git branch -M main
git push -u origin main
```

### 2️⃣ Sign Up for Railway (5 min)

1. Go to [railway.app](https://railway.app)
2. Click "Login" → "Login with GitHub"
3. Authorize Railway to access your GitHub

### 3️⃣ Deploy Week 1 (10 min)

1. Click "**New Project**"
2. Select "**Deploy from GitHub repo**"
3. Choose `mgmt298d-apps`
4. Click "**Add variables**" (optional, skip for now)
5. Click "**Deploy**"
6. ⚠️ **IMPORTANT:** Click "**Settings**" → Set "**Root Directory**" to `week1-app`
7. Wait 2-3 minutes for build
8. Click "**Settings**" → "**Networking**" → "**Generate Domain**"
9. Copy the URL (e.g., `week1-app-production.up.railway.app`)
10. Test: Visit `https://YOUR-URL/app`

### 4️⃣ Deploy Weeks 2-8 (Repeat 7 times)

For each remaining week:
1. "**New Project**" → "**Deploy from GitHub repo**"
2. Select `mgmt298d-apps`
3. Set "**Root Directory**" to `week2-app` (then `week3-app`, etc.)
4. Generate domain
5. Test the `/app` endpoint

**Time estimate:** ~10 min per app = 70 minutes total

### 5️⃣ Share URLs with Students

Create a document with all URLs:

```
MGMT298D - Week Apps

Week 1: https://week1-app-production.up.railway.app/app
Week 2: https://week2-app-production.up.railway.app/app
Week 3: https://week3-app-production.up.railway.app/app
...
```

## 💰 Quick Cost Check

After deploying:
1. Railway Dashboard → Click your username → "**Usage**"
2. Monitor costs (first $5 free each month)
3. Upgrade to Hobby ($5/mo) or Pro ($20/mo) as needed

## ✅ Verification Checklist

- [ ] All 8 apps pushed to GitHub
- [ ] Railway account created and connected to GitHub
- [ ] Week 1 deployed and URL tested
- [ ] Week 2 deployed and URL tested
- [ ] Week 3 deployed and URL tested
- [ ] Week 4 deployed and URL tested
- [ ] Week 5 deployed and URL tested
- [ ] Week 6 deployed and URL tested
- [ ] Week 7 deployed and URL tested
- [ ] Week 8 deployed and URL tested
- [ ] All URLs shared with students

## 🆘 Common Issues

**"Build failed"**
- Check Railway logs for specific error
- Verify `requirements.txt` exists in the week folder
- Make sure Root Directory is set correctly

**"App deployed but shows 404"**
- Add `/app` to the end of your URL
- Check if the app route is `/app` or `/`

**"Port error"**
- Apps now auto-detect Railway's PORT variable
- No action needed

**"GitHub connection failed"**
- Re-authorize Railway in GitHub settings
- Make sure repo is public (or upgrade to Railway Pro for private repos)

## 📞 Need Help?

- Full guide: See `DEPLOYMENT_GUIDE.md`
- Railway docs: https://docs.railway.app
- Railway Discord: https://discord.gg/railway

---

**You're done! Students can now access all apps via web URLs. 🎉**
