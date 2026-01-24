from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score, cross_val_predict
from xgboost import XGBRegressor
import warnings
import os
warnings.filterwarnings('ignore')

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables to store data
df = None
X = None
y = None
encoders = {}

def generate_sample_data():
    """Generate realistic Range Rover pricing data"""
    np.random.seed(42)
    n_samples = 500

    # Years from 2015 to 2024
    years = np.random.choice(range(2015, 2025), n_samples)

    # Mileage (correlated with year - older cars have more miles)
    base_mileage = (2024 - years) * 12000  # ~12k miles per year
    mileage = base_mileage + np.random.normal(0, 5000, n_samples)
    mileage = np.maximum(mileage, 1000).astype(int)

    # Trims with different base prices
    trims = np.random.choice(['Base', 'HSE', 'Autobiography', 'SVR', 'Sport'], n_samples,
                             p=[0.2, 0.3, 0.15, 0.1, 0.25])
    trim_price_effect = {'Base': 0, 'HSE': 8000, 'Autobiography': 25000, 'SVR': 35000, 'Sport': 5000}

    # States
    states = np.random.choice(['CA', 'TX', 'FL', 'NY', 'NJ', 'AZ', 'GA', 'WA', 'IL', 'PA'], n_samples)
    state_price_effect = {'CA': 2000, 'TX': 0, 'FL': 1000, 'NY': 3000, 'NJ': 2500,
                          'AZ': -500, 'GA': -1000, 'WA': 1500, 'IL': 500, 'PA': 0}

    # Colors
    colors = np.random.choice(['Black', 'White', 'Silver', 'Blue', 'Red', 'Gray', 'Green'], n_samples,
                              p=[0.25, 0.25, 0.15, 0.1, 0.1, 0.1, 0.05])
    color_price_effect = {'Black': 1000, 'White': 500, 'Silver': 0, 'Blue': 200,
                          'Red': 500, 'Gray': 0, 'Green': -200}

    # Calculate prices
    base_price = 45000  # Base MSRP
    prices = []
    for i in range(n_samples):
        price = base_price
        # Year effect (newer = more expensive)
        price += (years[i] - 2015) * 5000
        # Mileage effect (more miles = cheaper)
        price -= mileage[i] * 0.15
        # Trim effect
        price += trim_price_effect[trims[i]]
        # State effect
        price += state_price_effect[states[i]]
        # Color effect
        price += color_price_effect[colors[i]]
        # Add some noise
        price += np.random.normal(0, 3000)
        prices.append(max(price, 20000))

    data = pd.DataFrame({
        'year': years,
        'mileage': mileage,
        'trim': trims,
        'state': states,
        'color': colors,
        'price': np.array(prices).astype(int)
    })

    return data

def load_and_prepare_data():
    global df, X, y, encoders

    # Load from GitHub URL
    try:
        url = "https://raw.githubusercontent.com/ucla-anderson-SSAI/SSAI/main/range_rover.csv"
        print(f"Loading data from {url}")
        df = pd.read_csv(url)
    except Exception as e:
        print(f"Could not load remote data: {e}")
        print("Generating sample Range Rover data...")
        df = generate_sample_data()

    # Encode categorical variables
    categorical_cols = ['trim', 'state', 'color']
    for col in categorical_cols:
        if col in df.columns:
            encoders[col] = LabelEncoder()
            df[f'{col}_enc'] = encoders[col].fit_transform(df[col].astype(str))

    # Features and target
    feature_cols = ['year', 'mileage', 'trim_enc', 'state_enc', 'color_enc']
    X = df[feature_cols].values
    y = df['price'].values

    return df

# Load data on startup
@app.on_event("startup")
async def startup_event():
    load_and_prepare_data()

class AnalyzeRequest(BaseModel):
    n_estimators: int
    learning_rate: float

class GridSearchRequest(BaseModel):
    pass

@app.post("/grid_search")
async def grid_search():
    """Run grid search over learning_rate and n_estimators"""
    learning_rates = [0.01, 0.05, 0.1, 0.2, 0.3]
    n_estimators_list = [25, 50, 75, 100, 150, 200]

    results = []

    for lr in learning_rates:
        row = []
        for n_est in n_estimators_list:
            model = XGBRegressor(
                learning_rate=lr,
                n_estimators=n_est,
                random_state=42,
                n_jobs=-1
            )
            scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_absolute_error')
            mae = -scores.mean()
            row.append(round(mae, 2))
        results.append(row)

    # Find best combination
    min_mae = float('inf')
    best_lr = learning_rates[0]
    best_n_est = n_estimators_list[0]

    for i, lr in enumerate(learning_rates):
        for j, n_est in enumerate(n_estimators_list):
            if results[i][j] < min_mae:
                min_mae = results[i][j]
                best_lr = lr
                best_n_est = n_est

    # Get default model performance
    default_model = XGBRegressor(random_state=42, n_jobs=-1)
    default_scores = cross_val_score(default_model, X, y, cv=5, scoring='neg_mean_absolute_error')
    default_mae = -default_scores.mean()

    return {
        "heatmap_data": results,
        "learning_rates": learning_rates,
        "n_estimators_list": n_estimators_list,
        "best_lr": best_lr,
        "best_n_estimators": best_n_est,
        "best_mae": round(min_mae, 2),
        "default_mae": round(default_mae, 2),
        "default_params": {
            "learning_rate": 0.3,
            "n_estimators": 100
        }
    }

@app.post("/analyze")
async def analyze(request: AnalyzeRequest):
    """Analyze model with specific hyperparameters"""
    model = XGBRegressor(
        learning_rate=request.learning_rate,
        n_estimators=request.n_estimators,
        random_state=42,
        n_jobs=-1
    )

    # Cross-validation scores
    scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_absolute_error')
    mae = -scores.mean()
    std = scores.std()

    # Get predictions using cross_val_predict
    predictions = cross_val_predict(model, X, y, cv=5)

    # Fit model for feature importances
    model.fit(X, y)
    feature_names = ['year', 'mileage', 'trim_enc', 'state_enc', 'color_enc']
    feature_importances = dict(zip(feature_names, model.feature_importances_.tolist()))

    # Sample data for visualization (limit to 200 points for performance)
    sample_size = min(200, len(y))
    indices = np.random.choice(len(y), sample_size, replace=False)

    return {
        "mae": round(mae, 2),
        "std": round(std, 2),
        "predictions": [round(p, 2) for p in predictions[indices].tolist()],
        "actuals": [round(a, 2) for a in y[indices].tolist()],
        "feature_importances": {k: round(v, 4) for k, v in feature_importances.items()},
        "n_samples": len(y)
    }

@app.get("/data_info")
async def data_info():
    """Get information about the loaded data"""
    return {
        "n_samples": len(df),
        "n_features": 5,
        "features": ['year', 'mileage', 'trim_enc', 'state_enc', 'color_enc'],
        "target": "price",
        "price_range": [float(df['price'].min()), float(df['price'].max())],
        "price_mean": round(float(df['price'].mean()), 2)
    }

# Serve static files
app.mount("/static", StaticFiles(directory="."), name="static")

@app.get("/")
async def root():
    return FileResponse("index.html")

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8101))
    uvicorn.run(app, host="0.0.0.0", port=port)
