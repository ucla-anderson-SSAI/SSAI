# MGMT298D Apps - Quick Start Guide

All 8 weeks now have a **streamlined, consistent structure** for easy deployment and running.

## Structure

Each week app contains:
- `main.py` - FastAPI backend server
- `index.html` - Frontend interface
- `requirements.txt` - Python dependencies
- `data/` - Data folder (where applicable)

## How to Run Any Week

### Step 1: Install Dependencies
```bash
cd week1-app  # or week2-app, week3-app, etc.
pip install -r requirements.txt
```

### Step 2: Run the App
```bash
python main.py
```

### Step 3: Open in Browser
- Most apps: http://localhost:8000/app
- Week 2: http://localhost:8101/

## Data Sources

All apps now load data directly from GitHub URLs:
- **Week 1**: HMData.csv
- **Week 2**: range_rover.csv (with fallback to generated sample data)
- **Week 3**: netflix_ratings.csv

No need to download data files separately!

## Port Numbers

- Week 1: 8000
- Week 2: 8101
- Week 3-8: 8000

## Troubleshooting

**"Port already in use" error:**
```bash
# Find and kill the process using the port
lsof -ti:8000 | xargs kill -9
```

**Missing dependencies:**
```bash
pip install -r requirements.txt
```

**Internet connection required:**
Most apps load data from GitHub, so you'll need an internet connection when starting the app.
