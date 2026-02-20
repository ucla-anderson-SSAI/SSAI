# MGMT298D: Building Your Own AI-Powered Web App
## Student Scaffolding Guide

---

## Overview

In this assignment, you will build a fully deployed, interactive web application powered by a live AI API. You don't need to write code by hand — you'll use an LLM (Claude or ChatGPT) to generate it. Your job is to understand the architecture, craft good prompts, and deploy the result.

**Your app will have 6 components:**

| # | Component | What it does |
|---|-----------|--------------|
| 1 | **Python (FastAPI)** | Backend server — handles logic, calls AI, returns data |
| 2 | **HTML + JavaScript** | Frontend — the UI students interact with in a browser |
| 3 | **Gemini API** | The AI brain — generates answers, classifies text, etc. |
| 4 | **GitHub** | Version control — stores your code and connects to Railway |
| 5 | **Railway** | Cloud hosting — pulls from GitHub and makes your app live |
| 6 | **LLM (Claude/ChatGPT)** | Your coding assistant — generates all the code |

---

## Step 1: Pick Your App Idea

Your app should use an LLM (via the Gemini API) to do something interesting. Here are some ideas:

- **Earnings Call Analyzer** — upload a transcript, ask questions about it
- **Resume Screener** — paste a job description, evaluate candidate resumes
- **Contract Summarizer** — summarize legal documents in plain English
- **Product Review Classifier** — categorize reviews by sentiment and topic
- **Startup Pitch Evaluator** — grade a pitch on clarity, market fit, traction
- **Interview Prep Coach** — generate practice questions for a given job role
- **Supply Chain Risk Analyzer** — highlight risk factors in supplier documents
- **Email Tone Rewriter** — make an email more professional, concise, or friendly
- **Case Study Generator** — turn a business scenario into a Harvard-style case

**Your idea:** ________________________________________________

---

## Step 2: Understand the Architecture

Before prompting an LLM to write your code, make sure you understand what each file does.

```
your-app/
├── main.py          ← Python backend (FastAPI server)
├── index.html       ← Frontend (everything the user sees)
├── requirements.txt ← Python packages to install
└── Procfile         ← Tells Railway how to start your app
```

**How the pieces connect:**

```
User's Browser  ──── HTTP request ────►  FastAPI (main.py)
                ◄─── JSON response ────        │
                                               ▼
                                       Gemini API (AI)
```

The browser never talks to Gemini directly. All AI calls happen on the server (main.py), which keeps the API key secret.

---

## Step 3: Generate Your Code with an LLM

Use the prompts below — one for the backend, one for the frontend. Copy them into Claude or ChatGPT, fill in the blanks, and paste the output into files.

---

### Prompt A: Generate `main.py` (Backend)

Copy this prompt exactly, filling in the `[ ]` sections:

```
I'm building a FastAPI web app for a UCLA Anderson MBA course on AI strategy.

My app idea: [DESCRIBE YOUR APP IN 2-3 SENTENCES]

Please generate a complete main.py file using FastAPI that:

1. Uses the Google Gemini API (model: gemini-2.0-flash) with this hardcoded API key:
   AIzaSyDybjRDGeqcDkZczBl_TDThVAibapXAeQE

2. Has CORS middleware enabled (allow_origins=["*"])

3. Serves index.html at the root route GET /

4. Has a POST endpoint called /analyze that accepts JSON with these fields:
   [LIST THE INPUTS YOUR APP NEEDS — e.g., "text: str, tone: str"]
   And returns JSON with these fields:
   [LIST THE OUTPUTS YOUR APP SHOULD RETURN — e.g., "summary: str, score: int, reasoning: str"]

5. Inside the /analyze endpoint:
   - Build a clear, specific prompt for Gemini using the user's inputs
   - Call the Gemini API with temperature=0.3 and max_output_tokens=500
   - Parse the response and return it as structured JSON

6. Has proper error handling (try/except) that returns HTTP 500 on failure

7. Reads the PORT from os.environ.get("PORT", 8000)

8. Includes a requirements.txt with: fastapi, uvicorn[standard], pydantic, google-generativeai, python-multipart

9. Includes a Procfile with: web: uvicorn main:app --host 0.0.0.0 --port $PORT

Keep the code clean and well-commented. Don't use any external databases.
```

---

### Prompt B: Generate `index.html` (Frontend)

After you have your backend, use this prompt to generate the frontend:

```
I have a FastAPI backend with one endpoint:

POST /analyze
Input JSON: [PASTE YOUR INPUT FIELDS HERE]
Output JSON: [PASTE YOUR OUTPUT FIELDS HERE]

Please generate a single index.html file for a UCLA Anderson MBA AI course app called "[YOUR APP NAME]".

Style requirements (match exactly):
- UCLA Blue header: background #2774AE, with a gold (#FFD100) 4px bottom border line
- White body background, font: Inter (Google Font)
- Section cards: white background, border-radius 12px, 1px border rgba(0,0,0,0.08), subtle box-shadow
- Section titles in #2774AE, font-weight 700
- Buttons: background #2774AE, border-radius 10px, white text, hover to #005587
- Show a loading spinner (CSS animation) while waiting for the API response
- Input fields: border 1.5px rgba(0,0,0,0.12), border-radius 10px, focus border #2774AE
- Display results in clearly labeled output cards

Functional requirements:
- A form with [DESCRIBE YOUR INPUT FIELDS — e.g., a textarea for resume text, a dropdown for job level]
- A submit button that sends a POST request to /analyze with the form data as JSON
- Displays the result from the API in a clean, readable format
- Shows an error message if the API call fails
- The page title and header should say "Week 8: [YOUR APP NAME]"
- Subtitle: "MGMT298D · Science and Strategy of AI · UCLA Anderson"

Do not use React or any npm packages. Plain HTML, CSS, and vanilla JavaScript only.
Everything must be in a single index.html file.
```

---

### Prompt C: Fix & Iterate

Once you have working code, use these prompts to improve it:

**To add input validation:**
```
Add input validation to my FastAPI endpoint so it returns a clear error message
if the user submits empty text or inputs that are too short (< 20 characters).
```

**To improve the AI prompt:**
```
I want Gemini to return more structured output. Modify the prompt inside /analyze
so that Gemini returns its response as JSON with these exact keys: [LIST KEYS].
Then parse that JSON in Python and return it as the API response.
```

**To add a second feature:**
```
Add a second endpoint POST /compare that takes two versions of [INPUT]
and returns a side-by-side comparison of how Gemini evaluates each one.
Also update the HTML to include a second tab or section for this feature.
```

---

## Step 4: Test Locally (Optional)

If you want to test before deploying:

```bash
# 1. Install Python dependencies
pip install fastapi uvicorn google-generativeai pydantic python-multipart

# 2. Run the server
python main.py

# 3. Open your browser to:
http://localhost:8000
```

If you see errors, copy the full error message and ask your LLM:
```
My FastAPI app is giving this error: [PASTE ERROR]
Here is my main.py: [PASTE CODE]
Please fix it.
```

---

## Step 5: Push to GitHub

Railway deploys directly from a GitHub repository, so you need to put your files there first.

### One-time GitHub Setup

1. Go to **github.com** and create a free account if you don't have one
2. Click **"New repository"** → give it a name (e.g., `mgmt298d-week8-app`)
3. Set it to **Public** (Railway needs to read it)
4. Click **"Create repository"**

### Upload Your Files

**Option A — GitHub website (easiest):**
1. On your new repo page, click **"uploading an existing file"**
2. Drag and drop all 4 files: `main.py`, `index.html`, `requirements.txt`, `Procfile`
3. Click **"Commit changes"**

**Option B — Command line:**
```bash
git init
git add main.py index.html requirements.txt Procfile
git commit -m "Initial app commit"
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git
git push -u origin main
```

### File Checklist

Make sure these 4 files are in your repo root before deploying:

- [ ] `main.py` — your FastAPI backend
- [ ] `index.html` — your frontend
- [ ] `requirements.txt` — your Python dependencies
- [ ] `Procfile` — contains: `web: uvicorn main:app --host 0.0.0.0 --port $PORT`

---

## Step 6: Deploy to Railway

Railway connects to your GitHub repo and hosts your app live on the internet.

### Deploy Steps

1. Go to **railway.app** and sign in with GitHub
2. Click **"New Project"** → **"Deploy from GitHub Repo"**
3. Select the repo you just created in Step 5
4. Railway auto-detects your Procfile and starts building — usually takes 1–2 minutes

### After Deploying

- Railway gives you a public URL (e.g., `https://your-app.up.railway.app`)
- Open it in a browser — your app is live!
- If something breaks, check the **Logs** tab in Railway

### Updating Your App

Whenever you push a new commit to GitHub, Railway automatically redeploys. So the workflow is:
1. Edit files
2. Commit and push to GitHub
3. Railway redeploys automatically (takes ~1 minute)

---

## Step 7: Submission

Submit the following on Canvas:

1. **Your Railway URL** — the live link to your deployed app
2. **Your GitHub repo link** — must contain main.py, index.html, requirements.txt, Procfile
3. **A 1-paragraph description** of what your app does and what AI prompt strategy you used
4. **A screenshot** of your app working with real input and output

---

## Common Issues & Fixes

| Problem | Likely Cause | Fix |
|---------|-------------|-----|
| Railway build fails | Missing requirements.txt or wrong package name | Check spelling; google-generativeai not google-generative-ai |
| App loads but API call fails | CORS issue or endpoint URL mismatch | Make sure your JS fetches `/analyze` (same-origin), not `localhost` |
| Gemini returns an error | Temperature out of range or prompt too long | Keep temperature between 0.0–1.0; shorten your prompt |
| App works locally but not on Railway | PORT not read from environment | Make sure main.py uses `os.environ.get("PORT", 8000)` |
| Index.html not found | FastAPI not serving static files | Make sure you have `return FileResponse("index.html")` at `GET /` |

---

## Grading Rubric

| Criteria | Points |
|----------|--------|
| App is live and accessible via Railway URL | 25 |
| App successfully calls Gemini and displays a real AI-generated response | 25 |
| UI is clean and matches the course style (UCLA blue, correct fonts) | 20 |
| Code is well-structured and commented | 15 |
| 1-paragraph writeup explains the prompt strategy used | 15 |
| **Total** | **100** |

---

## API Key

Use this key in your `main.py` — do not share it publicly or post it on GitHub:

```
AIzaSyDybjRDGeqcDkZczBl_TDThVAibapXAeQE
```

*This key is provided for course use only. Do not use it for personal projects.*

---

*MGMT298D — Science and Strategy of AI — UCLA Anderson School of Management*
