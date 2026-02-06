"""
MGMT298D: Week 1 - Linear Regression API
FastAPI backend with hyperparameter controls for linear regression models
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso, Ridge, ElasticNet, LinearRegression, LassoCV, RidgeCV, ElasticNetCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from typing import Optional, List, Dict, Any
import math

app = FastAPI(
    title="Week 1: Linear Regression API",
    description="Interactive API for exploring linear regression with hyperparameter controls",
    version="2.0.0"
)

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Constants
DATA_URL = "https://raw.githubusercontent.com/ucla-anderson-SSAI/SSAI/main/HMData.csv"
MONTH_COLS = ["January", "February", "March", "April", "May", "June",
              "July", "August", "September", "October", "November", "December"]

# Available features with descriptions (matches Week 1 notebook)
AVAILABLE_FEATURES = {
    # Base
    "price": "Current price of the product",
    # Lag features
    "lag_1": "Sales from 1 month ago",
    "lag_2": "Sales from 2 months ago",
    "lag_3": "Sales from 3 months ago",
    "lag_4": "Sales from 4 months ago",
    "lag_5": "Sales from 5 months ago",
    "lag_6": "Sales from 6 months ago",
    # Rolling statistics
    "ma_3": "3-month moving average of sales",
    "ma_5": "5-month moving average of sales",
    "ma_7": "7-month moving average of sales",
    "std_3": "3-month rolling std dev of sales",
    "std_5": "5-month rolling std dev of sales",
    "std_7": "7-month rolling std dev of sales",
    # Price features
    "price_change": "Absolute change in price from previous month",
    "price_pct_change": "Percentage change in price from previous month",
    # Squared terms
    "price_sq": "Price squared (non-linear effect)",
    "lag_1_sq": "Lag 1 squared",
    "lag_2_sq": "Lag 2 squared",
    "lag_3_sq": "Lag 3 squared",
    # Price × lag interactions
    "price_x_lag_1": "Price × Lag 1 interaction",
    "price_x_lag_2": "Price × Lag 2 interaction",
    "price_x_lag_3": "Price × Lag 3 interaction",
    "price_x_lag_4": "Price × Lag 4 interaction",
    # Price × rolling interactions
    "price_x_ma_3": "Price × 3-month MA interaction",
    "price_x_ma_5": "Price × 5-month MA interaction",
    "price_x_ma_7": "Price × 7-month MA interaction",
    # Cross-lag interactions
    "lag1_x_lag2": "Lag 1 × Lag 2 interaction",
    "lag1_x_lag3": "Lag 1 × Lag 3 interaction",
    "lag2_x_lag3": "Lag 2 × Lag 3 interaction",
    # Momentum ratios
    "lag1_over_lag2": "Lag 1 / Lag 2 momentum ratio",
    "lag1_over_ma3": "Lag 1 / 3-month MA momentum ratio",
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
    features: List[str] = Field(default=["price", "lag_1"], description="Features to include in model")
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


class RegPathPoint(BaseModel):
    alpha: float
    cv_mae: float
    n_nonzero: int


class AnalyzeResponse(BaseModel):
    product: str
    model_type: str
    alpha: Optional[float]
    alpha_source: str  # "user_specified" or "cv_selected"
    l1_ratio: Optional[float]  # ElasticNet mixing parameter (CV-selected or user-specified)
    features_used: List[str]
    test_month: str
    n_train: int
    n_test: int
    train_metrics: ModelMetrics
    test_metrics: ModelMetrics
    coefficients: List[CoefficientInfo]
    intercept: float
    predictions: List[float]
    actuals: List[float]
    residuals: List[float]
    reg_path: Optional[List[RegPathPoint]]  # regularization path for CV plots


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


@app.on_event("startup")
async def load_data():
    """Load dataset at startup."""
    global df, CATEGORIES
    try:
        print(f"[INFO] Loading data from {DATA_URL}")
        # Use requests-style loading with timeout
        import io
        import urllib.request
        with urllib.request.urlopen(DATA_URL, timeout=30) as response:
            csv_data = response.read().decode('utf-8')
        df = pd.read_csv(io.StringIO(csv_data))
        CATEGORIES = sorted(df["name"].unique().tolist())
        print(f"[INFO] Loaded {len(df)} rows, {len(CATEGORIES)} categories from {DATA_URL}")
    except Exception as e:
        print(f"[ERROR] Failed to load data from {DATA_URL}: {str(e)}")
        import traceback
        traceback.print_exc()
        df = None
        CATEGORIES = []


def prepare_data(selected_product: str) -> pd.DataFrame:
    """Prepare dataset for a selected product category with all engineered features.
    Matches the feature engineering in the Week 1 notebook."""
    df_sub = df[df["name"] == selected_product].copy()

    # Create month_num column
    df_sub["month_num"] = df_sub[MONTH_COLS].idxmax(axis=1).map(
        {m: i+1 for i, m in enumerate(MONTH_COLS)}
    )

    # --- Lag features (lag_1 through lag_6) ---
    for i in range(1, 7):
        df_sub[f"lag_{i}"] = df_sub.groupby("id")["sales"].shift(i)

    # --- Rolling statistics (shifted by 1 to avoid leakage) ---
    for w in [3, 5, 7]:
        df_sub[f"ma_{w}"] = df_sub.groupby("id")["sales"].transform(
            lambda x: x.rolling(w).mean().shift(1)
        )
        df_sub[f"std_{w}"] = df_sub.groupby("id")["sales"].transform(
            lambda x: x.rolling(w).std().shift(1)
        )

    # --- Price features ---
    df_sub["price_change"] = df_sub.groupby("id")["price"].diff()
    df_sub["price_pct_change"] = df_sub.groupby("id")["price"].pct_change()

    # --- Squared terms ---
    df_sub["price_sq"] = df_sub["price"] ** 2
    for i in range(1, 4):
        df_sub[f"lag_{i}_sq"] = df_sub[f"lag_{i}"] ** 2

    # --- Interaction terms: price × lags ---
    for i in range(1, 5):
        df_sub[f"price_x_lag_{i}"] = df_sub["price"] * df_sub[f"lag_{i}"]

    # --- Interaction terms: price × rolling ---
    for w in [3, 5, 7]:
        df_sub[f"price_x_ma_{w}"] = df_sub["price"] * df_sub[f"ma_{w}"]

    # --- Cross-lag interactions ---
    df_sub["lag1_x_lag2"] = df_sub["lag_1"] * df_sub["lag_2"]
    df_sub["lag1_x_lag3"] = df_sub["lag_1"] * df_sub["lag_3"]
    df_sub["lag2_x_lag3"] = df_sub["lag_2"] * df_sub["lag_3"]

    # --- Momentum ratios (safe division) ---
    df_sub["lag1_over_lag2"] = df_sub["lag_1"] / df_sub["lag_2"].replace(0, np.nan)
    df_sub["lag1_over_ma3"] = df_sub["lag_1"] / df_sub["ma_3"].replace(0, np.nan)

    # --- Fill NaN from shifts, rolling windows, and division ---
    feature_cols = [c for c in AVAILABLE_FEATURES.keys() if c != "price"]
    df_sub[feature_cols] = df_sub[feature_cols].fillna(0)
    df_sub.replace([np.inf, -np.inf], 0, inplace=True)

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
    Returns: (model, predictions, alpha_used, alpha_source, l1_ratio_used)
    """
    if model_type == "ols":
        model = LinearRegression()
        model.fit(X_train, y_train)
        return model, model.predict(X_test), None, "not_applicable", None

    elif model_type == "lasso":
        if alpha is not None:
            model = Lasso(alpha=alpha, random_state=42, max_iter=10000)
            alpha_source = "user_specified"
        else:
            model = LassoCV(cv=5, random_state=42, max_iter=10000)
            alpha_source = "cv_selected"
        model.fit(X_train, y_train)
        alpha_used = alpha if alpha is not None else model.alpha_
        return model, model.predict(X_test), alpha_used, alpha_source, None

    elif model_type == "ridge":
        if alpha is not None:
            model = Ridge(alpha=alpha, random_state=42, max_iter=10000)
            alpha_source = "user_specified"
        else:
            model = RidgeCV(cv=5, alphas=np.logspace(-3, 3, 50))
            alpha_source = "cv_selected"
        model.fit(X_train, y_train)
        alpha_used = alpha if alpha is not None else model.alpha_
        return model, model.predict(X_test), alpha_used, alpha_source, None

    elif model_type == "elasticnet":
        if alpha is not None:
            model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42, max_iter=10000)
            alpha_source = "user_specified"
        else:
            model = ElasticNetCV(
                cv=5,
                l1_ratio=[0.1, 0.5, 0.7, 0.9, 0.95, 0.99, 1.0],
                alphas=np.logspace(-3, 2, 30),
                random_state=42,
                max_iter=10000
            )
            alpha_source = "cv_selected"
        model.fit(X_train, y_train)
        alpha_used = alpha if alpha is not None else model.alpha_
        l1_ratio_used = l1_ratio if alpha is not None else model.l1_ratio_
        return model, model.predict(X_test), alpha_used, alpha_source, l1_ratio_used

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


def compute_reg_path(
    X_train: np.ndarray,
    y_train: np.ndarray,
    model_type: str,
    l1_ratio: float = 0.5,
    cv: int = 5
) -> List[RegPathPoint]:
    """Sweep alpha values and return CV MAE + number of nonzero coefficients at each."""
    from sklearn.model_selection import cross_val_score

    alphas = np.logspace(-3, 3, 25)
    path = []

    for a in alphas:
        if model_type == "lasso":
            m = Lasso(alpha=a, random_state=42, max_iter=10000)
        elif model_type == "ridge":
            m = Ridge(alpha=a, random_state=42, max_iter=10000)
        elif model_type == "elasticnet":
            m = ElasticNet(alpha=a, l1_ratio=l1_ratio, random_state=42, max_iter=10000)
        else:
            continue

        scores = cross_val_score(m, X_train, y_train, cv=cv, scoring="neg_mean_absolute_error")
        cv_mae = -scores.mean()

        m.fit(X_train, y_train)
        n_nonzero = int(np.sum(m.coef_ != 0))

        path.append(RegPathPoint(
            alpha=round(float(a), 6),
            cv_mae=round(float(cv_mae), 4),
            n_nonzero=n_nonzero
        ))

    return path


# API Endpoints

@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "message": "Week 1 Linear Regression API v2.0",
        "status": "running",
        "data_loaded": df is not None,
        "categories_count": len(CATEGORIES),
        "endpoints": ["/categories", "/features", "/analyze", "/compare/{product}"]
    }


@app.get("/ping")
async def ping():
    """Simple ping endpoint for testing connectivity."""
    return {"pong": True}


@app.get("/categories")
async def get_categories():
    """List all available product categories."""
    global df, CATEGORIES
    # If data wasn't loaded at startup, try again
    if df is None or len(CATEGORIES) == 0:
        try:
            import io
            import urllib.request
            print(f"[INFO] Lazy-loading data from {DATA_URL}")
            with urllib.request.urlopen(DATA_URL, timeout=30) as response:
                csv_data = response.read().decode('utf-8')
            df = pd.read_csv(io.StringIO(csv_data))
            CATEGORIES = sorted(df["name"].unique().tolist())
            print(f"[INFO] Lazy-loaded {len(df)} rows, {len(CATEGORIES)} categories")
        except Exception as e:
            print(f"[ERROR] Failed to lazy-load data: {str(e)}")
            import traceback
            traceback.print_exc()
            raise HTTPException(status_code=503, detail=f"Data not available: {str(e)}")

    return {
        "categories": CATEGORIES,
        "count": len(CATEGORIES)
    }


@app.get("/features")
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


@app.post("/analyze", response_model=AnalyzeResponse)
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
    model, predictions, alpha_used, alpha_source, l1_ratio_used = train_model(
        X_train_scaled, y_train, X_test_scaled,
        request.model_type,
        request.alpha,
        request.l1_ratio or 0.5
    )

    # Calculate test metrics (out-of-sample)
    test_metrics = calculate_metrics(y_test, predictions)

    # Calculate train metrics (in-sample)
    train_predictions = model.predict(X_train_scaled)
    train_metrics = calculate_metrics(y_train, train_predictions)

    # Extract coefficients
    coefficients = extract_coefficients(model, request.features)

    # Calculate residuals
    residuals = (y_test - predictions).tolist()

    # Compute regularization path if CV was used
    reg_path = None
    if alpha_source == "cv_selected" and request.model_type != "ols":
        reg_path = compute_reg_path(
            X_train_scaled, y_train,
            request.model_type,
            l1_ratio=l1_ratio_used or 0.5,
            cv=5
        )

    return AnalyzeResponse(
        product=request.product,
        model_type=request.model_type,
        alpha=alpha_used,
        alpha_source=alpha_source,
        l1_ratio=l1_ratio_used,
        features_used=request.features,
        test_month=test_month_name,
        n_train=len(train),
        n_test=len(test),
        train_metrics=train_metrics,
        test_metrics=test_metrics,
        coefficients=coefficients,
        intercept=round(float(model.intercept_), 6),
        predictions=[round(float(p), 2) for p in predictions],
        actuals=[round(float(a), 2) for a in y_test],
        residuals=[round(float(r), 2) for r in residuals],
        reg_path=reg_path
    )


@app.get("/compare/{product}", response_model=CompareResponse)
async def compare_models(product: str, test_month: Optional[int] = None):
    """
    Run the standard A/B/C model comparison for a product.

    - Model A: Price only
    - Model B: Price + Last Month's Sales
    - Model C: All features (price, price_change, lag_1, lag_2, lag_3, ma_3)
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
        ("Model B: Price + Lag", ["price", "lag_1"]),
        ("Model C: All Features", ["price", "price_change", "lag_1", "lag_2", "lag_3", "ma_3"])
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
        model, predictions, alpha_used, _, _ = train_model(
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


@app.get("/months/{product}")
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


@app.get("/health")
async def health_check():
    """Detailed health check."""
    return {
        "status": "healthy",
        "data_loaded": df is not None,
        "categories_count": len(CATEGORIES),
        "available_features": list(AVAILABLE_FEATURES.keys()),
        "model_types": MODEL_TYPES
    }


# Serve frontend
app.mount("/static", StaticFiles(directory="."), name="static")

@app.get("/app")
async def serve_frontend():
    """Serve the frontend application."""
    return FileResponse("index.html")


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
