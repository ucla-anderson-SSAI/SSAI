"""
Assignment 1: Linear Regression with Feature Engineering
FastAPI backend comparing three models with increasing feature complexity
Using Airbnb listings data to predict reviews_per_month
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV
from sklearn.metrics import mean_absolute_error
from typing import List
import os

router = APIRouter()

# CORS handled by main app

# Global data
df = None
NEIGHBORHOODS = []

# Feature groups for three models
# Model A: Basic property features
MODEL_A_FEATURES = ["accommodates"]

# Model B: Property + availability features
MODEL_B_FEATURES = ["accommodates", "availability_365"]

# Model C: All features (property, availability, host, ratings)
MODEL_C_FEATURES = [
    "accommodates",
    "bedrooms",
    "bathrooms",
    "minimum_nights",
    "availability_365",
    "host_is_superhost",
    "instant_bookable",
    "host_identity_verified",
    "hosts_time_as_host_years",
    "calculated_host_listings_count",
    "review_scores_rating",
    "is_entire_home"
]


class CoefficientInfo(BaseModel):
    feature: str
    coefficient: float
    abs_coefficient: float


class ModelResult(BaseModel):
    model_name: str
    mae: float
    features: List[str]
    coefficients: List[CoefficientInfo]
    predictions: List[float]
    actuals: List[float]


class AnalyzeResponse(BaseModel):
    neighborhood: str
    n_train: int
    n_test: int
    model_a: ModelResult
    model_b: ModelResult
    model_c: ModelResult
    improvement_ab: float
    improvement_ac: float
    sample_data: List[dict]


# GitHub raw URL for data
DATA_URL = "https://raw.githubusercontent.com/ucla-anderson-SSAI/SSAI/main/listings.csv"


def load_data():
    """Load and preprocess Airbnb dataset."""
    global df, NEIGHBORHOODS

    if df is not None:
        return

    # Load data from GitHub
    print("Loading Airbnb listings data from GitHub...")
    df = pd.read_csv(DATA_URL, low_memory=False)

    # Preprocess boolean columns (t/f to 1/0)
    bool_cols = ['host_is_superhost', 'instant_bookable', 'host_identity_verified']
    for col in bool_cols:
        df[col] = (df[col] == 't').astype(int)

    # Create room type binary feature
    df['is_entire_home'] = (df['room_type'] == 'Entire home/apt').astype(int)

    # Fill missing values for key features
    df['bedrooms'] = df['bedrooms'].fillna(df['bedrooms'].median())
    df['bathrooms'] = df['bathrooms'].fillna(df['bathrooms'].median())
    df['hosts_time_as_host_years'] = df['hosts_time_as_host_years'].fillna(0)
    df['review_scores_rating'] = df['review_scores_rating'].fillna(df['review_scores_rating'].median())

    # Filter to rows with reviews_per_month (our target variable)
    df = df[df['reviews_per_month'].notna()].copy()

    # Get neighborhoods with enough listings
    neighborhood_counts = df['neighbourhood_cleansed'].value_counts()
    valid_neighborhoods = neighborhood_counts[neighborhood_counts >= 100].index.tolist()
    NEIGHBORHOODS = sorted(valid_neighborhoods)

    print(f"[INFO] Loaded {len(df)} listings, {len(NEIGHBORHOODS)} neighborhoods")


def train_model(X_train, y_train, X_test, feature_names):
    """Train a LassoCV model and return predictions and coefficients."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LassoCV(cv=5, random_state=42, max_iter=10000)
    model.fit(X_train_scaled, y_train)

    predictions = model.predict(X_test_scaled)
    predictions = np.maximum(predictions, 0)  # Reviews can't be negative

    coefficients = [
        CoefficientInfo(
            feature=name,
            coefficient=round(float(coef), 6),
            abs_coefficient=round(abs(float(coef)), 6)
        )
        for name, coef in zip(feature_names, model.coef_)
    ]

    return predictions, coefficients


@router.get("/")
async def root():
    """Serve the main HTML page."""
    return FileResponse(os.path.join(os.path.dirname(__file__), "index.html"))


@router.get("/api/neighborhoods")
async def get_neighborhoods():
    """List all available neighborhoods."""
    return {
        "neighborhoods": NEIGHBORHOODS,
        "count": len(NEIGHBORHOODS)
    }


@router.get("/api/analyze/{neighborhood}")
async def analyze(neighborhood: str):
    """
    Analyze a neighborhood comparing three models with increasing feature complexity.

    Model A: Basic property features (accommodates only)
    Model B: Property + availability features
    Model C: All features including host and ratings
    """
    # Use "All Neighborhoods" or validate specific neighborhood
    if neighborhood == "All":
        df_sub = df.copy()
    elif neighborhood not in NEIGHBORHOODS:
        raise HTTPException(
            status_code=404,
            detail=f"Neighborhood '{neighborhood}' not found. Available: {NEIGHBORHOODS[:5]}..."
        )
    else:
        df_sub = df[df['neighbourhood_cleansed'] == neighborhood].copy()

    if len(df_sub) < 50:
        raise HTTPException(
            status_code=400,
            detail=f"Insufficient data for neighborhood '{neighborhood}'"
        )

    # Shuffle and split 80/20
    df_sub = df_sub.sample(frac=1, random_state=42).reset_index(drop=True)
    split_idx = int(len(df_sub) * 0.8)
    train = df_sub.iloc[:split_idx]
    test = df_sub.iloc[split_idx:]

    y_train = train['reviews_per_month'].values
    y_test = test['reviews_per_month'].values

    # Model A: Basic property features
    X_train_A = train[MODEL_A_FEATURES].values
    X_test_A = test[MODEL_A_FEATURES].values
    pred_A, coef_A = train_model(X_train_A, y_train, X_test_A, MODEL_A_FEATURES)
    mae_A = mean_absolute_error(y_test, pred_A)

    model_a = ModelResult(
        model_name="Model A: Property Size Only",
        mae=round(mae_A, 4),
        features=MODEL_A_FEATURES,
        coefficients=coef_A,
        predictions=[round(float(p), 3) for p in pred_A],
        actuals=[round(float(a), 3) for a in y_test]
    )

    # Model B: Property + availability
    X_train_B = train[MODEL_B_FEATURES].values
    X_test_B = test[MODEL_B_FEATURES].values
    pred_B, coef_B = train_model(X_train_B, y_train, X_test_B, MODEL_B_FEATURES)
    mae_B = mean_absolute_error(y_test, pred_B)

    model_b = ModelResult(
        model_name="Model B: + Availability",
        mae=round(mae_B, 4),
        features=MODEL_B_FEATURES,
        coefficients=coef_B,
        predictions=[round(float(p), 3) for p in pred_B],
        actuals=[round(float(a), 3) for a in y_test]
    )

    # Model C: All features
    X_train_C = train[MODEL_C_FEATURES].values
    X_test_C = test[MODEL_C_FEATURES].values
    pred_C, coef_C = train_model(X_train_C, y_train, X_test_C, MODEL_C_FEATURES)
    mae_C = mean_absolute_error(y_test, pred_C)

    model_c = ModelResult(
        model_name="Model C: + Host & Ratings",
        mae=round(mae_C, 4),
        features=MODEL_C_FEATURES,
        coefficients=coef_C,
        predictions=[round(float(p), 3) for p in pred_C],
        actuals=[round(float(a), 3) for a in y_test]
    )

    # Calculate improvements
    improvement_ab = ((mae_A - mae_B) / mae_A) * 100 if mae_A > 0 else 0
    improvement_ac = ((mae_A - mae_C) / mae_A) * 100 if mae_A > 0 else 0

    # Sample data for visualization (first 200 test points)
    sample_size = min(200, len(test))
    sample_data = []
    for i in range(sample_size):
        sample_data.append({
            "actual": round(float(y_test[i]), 3),
            "pred_a": round(float(pred_A[i]), 3),
            "pred_b": round(float(pred_B[i]), 3),
            "pred_c": round(float(pred_C[i]), 3),
            "accommodates": int(test.iloc[i]['accommodates']),
            "availability": int(test.iloc[i]['availability_365'])
        })

    return AnalyzeResponse(
        neighborhood=neighborhood if neighborhood != "All" else "All Neighborhoods",
        n_train=len(train),
        n_test=len(test),
        model_a=model_a,
        model_b=model_b,
        model_c=model_c,
        improvement_ab=round(improvement_ab, 2),
        improvement_ac=round(improvement_ac, 2),
        sample_data=sample_data
    )


@router.get("/api/health")
async def health_check():
    """Detailed health check."""
    return {
        "status": "healthy",
        "data_loaded": df is not None,
        "neighborhoods_count": len(NEIGHBORHOODS),
        "total_listings": len(df) if df is not None else 0
    }
