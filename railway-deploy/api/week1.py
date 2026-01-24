"""
MGMT298D: Week 1 - Linear Regression API
FastAPI backend with hyperparameter controls for linear regression models
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso, Ridge, ElasticNet, LinearRegression, LassoCV, RidgeCV, ElasticNetCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from typing import Optional, List, Dict, Any
import math

router = APIRouter()

# CORS handled by main app

# Constants
# GitHub raw URL for data
DATA_URL = "https://raw.githubusercontent.com/ucla-anderson-SSAI/SSAI/main/HMData.csv"
MONTH_COLS = ["January", "February", "March", "April", "May", "June",
              "July", "August", "September", "October", "November", "December"]

# Available features with descriptions
AVAILABLE_FEATURES = {
    "price": "Current price of the product",
    "log_price": "Natural log of price",
    "price_squared": "Price squared (non-linear effect)",
    "lag_m1": "Sales from 1 month ago",
    "lag_m2": "Sales from 2 months ago",
    "lag_m3": "Sales from 3 months ago",
    "lag_m6": "Sales from 6 months ago",
    "ma_3": "3-month moving average of sales",
    "ma_6": "6-month moving average of sales",
    "price_change": "Absolute change in price from previous month",
    "price_change_pct": "Percentage change in price from previous month",
    "price_volatility": "Rolling standard deviation of price (volatility)"
}

# Model types
MODEL_TYPES = ["ols", "lasso", "ridge", "elasticnet"]

# Global data
df = None
CATEGORIES = []


# Pydantic Models
class FeatureInfo(BaseModel):
    name: str
    description: str


class AnalyzeRequest(BaseModel):
    product: str = Field(..., description="Product category to analyze")
    model_type: str = Field(default="lasso", description="Model type: ols, lasso, ridge, elasticnet")
    features: List[str] = Field(default=["price", "lag_m1"], description="Features to include in model")
    alpha: Optional[float] = Field(default=None, description="Regularization strength (auto-select via CV if not provided)")
    l1_ratio: Optional[float] = Field(default=0.5, description="ElasticNet mixing parameter (0=Ridge, 1=Lasso)")
    test_month: Optional[int] = Field(default=None, description="Month number to use as test set (1-12, default=last available)")


class CoefficientInfo(BaseModel):
    feature: str
    coefficient: float
    abs_coefficient: float


class ModelMetrics(BaseModel):
    mae: float
    rmse: float
    r2: float


class AnalyzeResponse(BaseModel):
    product: str
    model_type: str
    alpha: Optional[float]
    alpha_source: str  # "user_specified" or "cv_selected"
    features_used: List[str]
    test_month: str
    n_train: int
    n_test: int
    metrics: ModelMetrics
    coefficients: List[CoefficientInfo]
    intercept: float
    predictions: List[float]
    actuals: List[float]
    residuals: List[float]


class CompareModelResult(BaseModel):
    model_name: str
    features: List[str]
    metrics: ModelMetrics
    coefficients: List[CoefficientInfo]
    intercept: float
    predictions: List[float]
    actuals: List[float]


class CompareResponse(BaseModel):
    product: str
    test_month: str
    n_train: int
    n_test: int
    models: List[CompareModelResult]
    improvement_ab: float
    improvement_ac: float
    sales_over_time: Dict[str, Any]


def load_data():
    """Load dataset from GitHub."""
    global df, CATEGORIES
    if df is None:
        print("[INFO] Loading HMData from GitHub...")
        df = pd.read_csv(DATA_URL)
        CATEGORIES = sorted(df["name"].unique().tolist())
        print(f"[INFO] Loaded {len(df)} rows, {len(CATEGORIES)} categories")

# Load data on import
load_data()


def prepare_data(selected_product: str) -> pd.DataFrame:
    """Prepare dataset for a selected product category with all engineered features."""
    df_sub = df[df["name"] == selected_product].copy()

    # Create month_num column
    df_sub["month_num"] = df_sub[MONTH_COLS].idxmax(axis=1).map(
        {m: i+1 for i, m in enumerate(MONTH_COLS)}
    )

    # Price transformations
    df_sub["log_price"] = np.log(df_sub["price"].clip(lower=0.01))
    df_sub["price_squared"] = (df_sub["price"] ** 2) / 1000  # Scale down for numerical stability

    # Create lag features (sales)
    for i in [1, 2, 3, 6]:
        df_sub[f"lag_m{i}"] = df_sub.groupby("id")["sales"].shift(i)

    # Create moving averages
    df_sub["ma_3"] = df_sub.groupby("id")["sales"].shift(1).rolling(3, min_periods=1).mean().values
    df_sub["ma_6"] = df_sub.groupby("id")["sales"].shift(1).rolling(6, min_periods=1).mean().values

    # Price change features
    df_sub["price_change"] = df_sub.groupby("id")["price"].diff()
    df_sub["price_change_pct"] = df_sub.groupby("id")["price"].pct_change()

    # Price volatility (rolling std of price)
    df_sub["price_volatility"] = df_sub.groupby("id")["price"].transform(
        lambda x: x.rolling(3, min_periods=1).std()
    )

    # Fill NaN values
    feature_cols = ["lag_m1", "lag_m2", "lag_m3", "lag_m6", "ma_3", "ma_6",
                    "price_change", "price_change_pct", "price_volatility"]
    df_sub[feature_cols] = df_sub[feature_cols].fillna(0)

    return df_sub


def get_sales_over_time(df_sub: pd.DataFrame, n_samples: int = 10) -> Dict[str, Any]:
    """Get sales trajectories for sample products."""
    np.random.seed(42)
    unique_ids = df_sub["id"].unique()
    sample_ids = np.random.choice(unique_ids, size=min(n_samples, len(unique_ids)), replace=False)

    trajectories = []
    for pid in sample_ids:
        product_data = df_sub[df_sub["id"] == pid]
        monthly_sales = []
        for month in MONTH_COLS:
            sales = product_data[product_data[month] == 1]["sales"].values
            monthly_sales.append(float(sales[0]) if len(sales) > 0 else None)
        trajectories.append({
            "id": int(pid),
            "sales": monthly_sales
        })

    return {
        "months": MONTH_COLS,
        "trajectories": trajectories
    }


def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    model_type: str,
    alpha: Optional[float],
    l1_ratio: float = 0.5
) -> tuple:
    """
    Train a linear model with specified hyperparameters.
    Returns: (model, predictions, alpha_used, alpha_source)
    """
    if model_type == "ols":
        model = LinearRegression()
        model.fit(X_train, y_train)
        return model, model.predict(X_test), None, "not_applicable"

    elif model_type == "lasso":
        if alpha is not None:
            model = Lasso(alpha=alpha, random_state=42, max_iter=10000)
            alpha_source = "user_specified"
        else:
            model = LassoCV(cv=5, random_state=42, max_iter=10000)
            alpha_source = "cv_selected"
        model.fit(X_train, y_train)
        alpha_used = alpha if alpha is not None else model.alpha_
        return model, model.predict(X_test), alpha_used, alpha_source

    elif model_type == "ridge":
        if alpha is not None:
            model = Ridge(alpha=alpha, random_state=42, max_iter=10000)
            alpha_source = "user_specified"
        else:
            model = RidgeCV(cv=5)
            alpha_source = "cv_selected"
        model.fit(X_train, y_train)
        alpha_used = alpha if alpha is not None else model.alpha_
        return model, model.predict(X_test), alpha_used, alpha_source

    elif model_type == "elasticnet":
        if alpha is not None:
            model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42, max_iter=10000)
            alpha_source = "user_specified"
        else:
            model = ElasticNetCV(cv=5, l1_ratio=[0.1, 0.5, 0.7, 0.9, 0.95, 0.99, 1.0], random_state=42, max_iter=10000)
            alpha_source = "cv_selected"
        model.fit(X_train, y_train)
        alpha_used = alpha if alpha is not None else model.alpha_
        return model, model.predict(X_test), alpha_used, alpha_source

    else:
        raise ValueError(f"Unknown model type: {model_type}")


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> ModelMetrics:
    """Calculate regression metrics."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return ModelMetrics(
        mae=round(mae, 4),
        rmse=round(rmse, 4),
        r2=round(r2, 4)
    )


def extract_coefficients(model, feature_names: List[str]) -> List[CoefficientInfo]:
    """Extract coefficients from a trained model."""
    coefs = model.coef_ if hasattr(model, 'coef_') else model.coef_
    return [
        CoefficientInfo(
            feature=name,
            coefficient=round(float(coef), 6),
            abs_coefficient=round(abs(float(coef)), 6)
        )
        for name, coef in zip(feature_names, coefs)
    ]


# API Endpoints

@router.get("/")
async def root():
    """Health check endpoint."""
    return {
        "message": "Week 1 Linear Regression API v2.0",
        "status": "running",
        "endpoints": ["/categories", "/features", "/analyze", "/compare/{product}"]
    }


@router.get("/categories")
async def get_categories():
    """List all available product categories."""
    return {
        "categories": CATEGORIES,
        "count": len(CATEGORIES)
    }


@router.get("/features")
async def get_features():
    """List all available features with descriptions."""
    features = [
        FeatureInfo(name=name, description=desc)
        for name, desc in AVAILABLE_FEATURES.items()
    ]
    return {
        "features": features,
        "model_types": MODEL_TYPES,
        "model_descriptions": {
            "ols": "Ordinary Least Squares - no regularization",
            "lasso": "L1 regularization - can zero out coefficients for feature selection",
            "ridge": "L2 regularization - shrinks coefficients but keeps all features",
            "elasticnet": "Combination of L1 and L2 regularization"
        }
    }


@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze_custom(request: AnalyzeRequest):
    """
    Run a custom linear regression model with user-specified hyperparameters.

    - Select model type: ols, lasso, ridge, elasticnet
    - Choose features to include
    - Optionally specify alpha (regularization strength)
    - Select test month
    """
    # Validate product
    if request.product not in CATEGORIES:
        raise HTTPException(
            status_code=404,
            detail=f"Product '{request.product}' not found. Available: {CATEGORIES}"
        )

    # Validate model type
    if request.model_type not in MODEL_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid model type '{request.model_type}'. Available: {MODEL_TYPES}"
        )

    # Validate features
    invalid_features = [f for f in request.features if f not in AVAILABLE_FEATURES]
    if invalid_features:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid features: {invalid_features}. Available: {list(AVAILABLE_FEATURES.keys())}"
        )

    if len(request.features) == 0:
        raise HTTPException(
            status_code=400,
            detail="At least one feature must be selected"
        )

    # Prepare data
    df_sub = prepare_data(request.product)
    months_available = sorted(df_sub["month_num"].unique())

    # Determine test month
    if request.test_month is not None:
        if request.test_month not in months_available:
            raise HTTPException(
                status_code=400,
                detail=f"Test month {request.test_month} not available. Available months: {months_available}"
            )
        test_month_num = request.test_month
    else:
        test_month_num = months_available[-1]

    test_month_name = MONTH_COLS[test_month_num - 1]

    # Train/test split
    train = df_sub[df_sub["month_num"] < test_month_num].copy()
    test = df_sub[df_sub["month_num"] == test_month_num].copy()

    if len(train) == 0 or len(test) == 0:
        raise HTTPException(
            status_code=400,
            detail=f"Insufficient data for train/test split with test month {test_month_name}"
        )

    # Prepare features
    X_train = train[request.features].values
    X_test = test[request.features].values
    y_train = train["sales"].values
    y_test = test["sales"].values

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train model
    model, predictions, alpha_used, alpha_source = train_model(
        X_train_scaled, y_train, X_test_scaled,
        request.model_type,
        request.alpha,
        request.l1_ratio or 0.5
    )

    # Calculate metrics
    metrics = calculate_metrics(y_test, predictions)

    # Extract coefficients
    coefficients = extract_coefficients(model, request.features)

    # Calculate residuals
    residuals = (y_test - predictions).tolist()

    return AnalyzeResponse(
        product=request.product,
        model_type=request.model_type,
        alpha=alpha_used,
        alpha_source=alpha_source,
        features_used=request.features,
        test_month=test_month_name,
        n_train=len(train),
        n_test=len(test),
        metrics=metrics,
        coefficients=coefficients,
        intercept=round(float(model.intercept_), 6),
        predictions=[round(float(p), 2) for p in predictions],
        actuals=[round(float(a), 2) for a in y_test],
        residuals=[round(float(r), 2) for r in residuals]
    )


@router.get("/compare/{product}", response_model=CompareResponse)
async def compare_models(product: str, test_month: Optional[int] = None):
    """
    Run the standard A/B/C model comparison for a product.

    - Model A: Price only
    - Model B: Price + Last Month's Sales
    - Model C: All features (price, price_change, lag_m1, lag_m2, lag_m3, ma_3)
    """
    if product not in CATEGORIES:
        raise HTTPException(
            status_code=404,
            detail=f"Product '{product}' not found. Available: {CATEGORIES}"
        )

    # Prepare data
    df_sub = prepare_data(product)
    months_available = sorted(df_sub["month_num"].unique())

    # Determine test month
    if test_month is not None:
        if test_month not in months_available:
            raise HTTPException(
                status_code=400,
                detail=f"Test month {test_month} not available. Available months: {months_available}"
            )
        test_month_num = test_month
    else:
        test_month_num = months_available[-1]

    test_month_name = MONTH_COLS[test_month_num - 1]

    # Train/test split
    train = df_sub[df_sub["month_num"] < test_month_num].copy()
    test = df_sub[df_sub["month_num"] == test_month_num].copy()

    y_train = train["sales"].values
    y_test = test["sales"].values

    models_results = []

    # Model configurations
    model_configs = [
        ("Model A: Price Only", ["price"]),
        ("Model B: Price + Lag", ["price", "lag_m1"]),
        ("Model C: All Features", ["price", "price_change", "lag_m1", "lag_m2", "lag_m3", "ma_3"])
    ]

    for model_name, features in model_configs:
        # Prepare features
        X_train = train[features].values
        X_test = test[features].values

        # Scale
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Train with LassoCV (auto alpha selection)
        model, predictions, alpha_used, _ = train_model(
            X_train_scaled, y_train, X_test_scaled,
            "lasso", None, 0.5
        )

        # Calculate metrics
        metrics = calculate_metrics(y_test, predictions)

        # Extract coefficients
        coefficients = extract_coefficients(model, features)

        models_results.append(CompareModelResult(
            model_name=model_name,
            features=features,
            metrics=metrics,
            coefficients=coefficients,
            intercept=round(float(model.intercept_), 6),
            predictions=[round(float(p), 2) for p in predictions],
            actuals=[round(float(a), 2) for a in y_test]
        ))

    # Calculate improvements
    mae_a = models_results[0].metrics.mae
    mae_b = models_results[1].metrics.mae
    mae_c = models_results[2].metrics.mae

    improvement_ab = round(((mae_a - mae_b) / mae_a) * 100, 2) if mae_a > 0 else 0
    improvement_ac = round(((mae_a - mae_c) / mae_a) * 100, 2) if mae_a > 0 else 0

    # Get sales over time
    sales_data = get_sales_over_time(df_sub)

    return CompareResponse(
        product=product,
        test_month=test_month_name,
        n_train=len(train),
        n_test=len(test),
        models=models_results,
        improvement_ab=improvement_ab,
        improvement_ac=improvement_ac,
        sales_over_time=sales_data
    )


@router.get("/months/{product}")
async def get_available_months(product: str):
    """Get available months for a product category."""
    if product not in CATEGORIES:
        raise HTTPException(
            status_code=404,
            detail=f"Product '{product}' not found. Available: {CATEGORIES}"
        )

    df_sub = prepare_data(product)
    months_available = sorted(df_sub["month_num"].unique())

    return {
        "product": product,
        "available_months": [
            {"number": m, "name": MONTH_COLS[m-1]}
            for m in months_available
        ],
        "default_test_month": {
            "number": months_available[-1],
            "name": MONTH_COLS[months_available[-1] - 1]
        }
    }


@router.get("/health")
async def health_check():
    """Detailed health check."""
    return {
        "status": "healthy",
        "data_loaded": df is not None,
        "categories_count": len(CATEGORIES),
        "available_features": list(AVAILABLE_FEATURES.keys()),
        "model_types": MODEL_TYPES
    }



