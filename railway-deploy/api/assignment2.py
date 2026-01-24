from fastapi import APIRouter,  FastAPI
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

router = APIRouter()

def load_and_prepare_data():
    global df, X, y, encoders

    # Try to load from local file first, then URL, then generate sample data
    local_path = os.path.join(os.path.dirname(__file__), 'range_rover.csv')

    if os.path.exists(local_path):
        df = pd.read_csv(local_path)
    else:
        try:
            url = "https://raw.githubusercontent.com/ucla-anderson-SSAI/SSAI/refs/heads/main/range_rover.csv"
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
@router.on_event("startup")
async def startup_event():
    load_and_prepare_data()

class AnalyzeRequest(BaseModel):
    n_estimators: int
    learning_rate: float

class GridSearchRequest(BaseModel):
    pass

@router.post("/grid_search")
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

@router.post("/analyze")
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

@router.get("/data_info")
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
# Static files handled in main.py, name="static")

@router.get("/")
async def root():
    return FileResponse("index.html")

