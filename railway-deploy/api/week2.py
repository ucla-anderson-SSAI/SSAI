"""
Week 2: Tree Models - Vehicle Pricing Backend
FastAPI application for training and comparing tree-based models on Range Rover data.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, Literal
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb

router = APIRouter()

# CORS handled by main app

# Data URL
DATA_URL = "https://raw.githubusercontent.com/ucla-anderson-SSAI/SSAI/refs/heads/main/range_rover.csv"

# Global variables for data caching
_df_cache = None
_encoders = {}


class TrainRequest(BaseModel):
    """Request model for training endpoint."""
    model_type: Literal["decision_tree", "random_forest", "xgboost"] = Field(
        default="decision_tree",
        description="Type of model to train"
    )
    max_depth: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Maximum depth of the tree (1-20)"
    )
    n_estimators: int = Field(
        default=100,
        ge=10,
        le=200,
        description="Number of estimators for RF and XGBoost (10-200)"
    )
    learning_rate: float = Field(
        default=0.1,
        ge=0.01,
        le=0.5,
        description="Learning rate for XGBoost only (0.01-0.5)"
    )
    min_samples_split: int = Field(
        default=2,
        ge=2,
        le=20,
        description="Minimum samples to split for Decision Tree and RF (2-20)"
    )
    test_size: float = Field(
        default=0.2,
        ge=0.1,
        le=0.4,
        description="Test set proportion (0.1-0.4)"
    )
    random_state: int = Field(
        default=42,
        description="Random seed for reproducibility"
    )


class EstimatorsRequest(BaseModel):
    """Request model for trees_vs_estimators endpoint."""
    model_type: Literal["random_forest", "xgboost"] = Field(
        default="random_forest",
        description="Model type (random_forest or xgboost)"
    )
    max_depth: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Maximum depth of trees"
    )
    estimator_range: list[int] = Field(
        default=[10, 25, 50, 75, 100, 125, 150, 175, 200],
        description="List of n_estimators values to test"
    )
    learning_rate: float = Field(
        default=0.1,
        ge=0.01,
        le=0.5,
        description="Learning rate for XGBoost"
    )
    random_state: int = Field(
        default=42,
        description="Random seed for reproducibility"
    )


def load_and_prepare_data():
    """Load and prepare the Range Rover dataset."""
    global _df_cache, _encoders

    if _df_cache is not None:
        return _df_cache.copy(), _encoders

    # Load data
    df = pd.read_csv(DATA_URL)

    # Encode categorical variables
    categorical_cols = ['trim', 'state', 'color']
    _encoders = {}

    for col in categorical_cols:
        if col in df.columns:
            le = LabelEncoder()
            df[f'{col}_enc'] = le.fit_transform(df[col].astype(str))
            _encoders[col] = le

    _df_cache = df
    return df.copy(), _encoders


def get_features_and_target(df):
    """Extract features and target from dataframe."""
    feature_cols = ['year', 'mileage', 'trim_enc', 'state_enc', 'color_enc']
    available_features = [col for col in feature_cols if col in df.columns]

    X = df[available_features]
    y = df['price']

    return X, y, available_features


def calculate_metrics(y_true, y_pred):
    """Calculate regression metrics."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return {
        "mae": round(mae, 2),
        "rmse": round(rmse, 2),
        "r2": round(r2, 4)
    }


def train_model(model_type, X_train, y_train, params):
    """Train a model based on type and parameters."""
    if model_type == "decision_tree":
        model = DecisionTreeRegressor(
            max_depth=params["max_depth"],
            min_samples_split=params["min_samples_split"],
            random_state=params["random_state"]
        )
    elif model_type == "random_forest":
        model = RandomForestRegressor(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            min_samples_split=params["min_samples_split"],
            random_state=params["random_state"],
            n_jobs=-1
        )
    elif model_type == "xgboost":
        model = xgb.XGBRegressor(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            learning_rate=params["learning_rate"],
            random_state=params["random_state"],
            n_jobs=-1
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    model.fit(X_train, y_train)
    return model


@router.get("/")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "message": "Week 2: Tree Models - Vehicle Pricing API",
        "endpoints": [
            "GET /sample_data - View sample data",
            "POST /train - Train model with custom hyperparameters",
            "GET /compare - Compare all three model types",
            "POST /trees_vs_estimators - Analyze n_estimators impact"
        ]
    }


@router.get("/sample_data")
async def get_sample_data(n_samples: int = 10):
    """Return a sample of the Range Rover dataset."""
    try:
        df, encoders = load_and_prepare_data()

        sample = df.head(n_samples)

        return {
            "total_rows": len(df),
            "columns": list(df.columns),
            "sample": sample.to_dict(orient="records"),
            "feature_columns": ["year", "mileage", "trim_enc", "state_enc", "color_enc"],
            "target_column": "price",
            "categorical_mappings": {
                col: dict(zip(encoders[col].classes_, range(len(encoders[col].classes_))))
                for col in encoders
            },
            "statistics": {
                "price": {
                    "mean": round(df["price"].mean(), 2),
                    "std": round(df["price"].std(), 2),
                    "min": round(df["price"].min(), 2),
                    "max": round(df["price"].max(), 2)
                },
                "year": {
                    "min": int(df["year"].min()),
                    "max": int(df["year"].max())
                },
                "mileage": {
                    "mean": round(df["mileage"].mean(), 2),
                    "min": round(df["mileage"].min(), 2),
                    "max": round(df["mileage"].max(), 2)
                }
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/train")
async def train_model_endpoint(request: TrainRequest):
    """Train a model with custom hyperparameters."""
    try:
        df, _ = load_and_prepare_data()
        X, y, feature_names = get_features_and_target(df)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=request.test_size,
            random_state=request.random_state
        )

        # Prepare parameters
        params = {
            "max_depth": request.max_depth,
            "n_estimators": request.n_estimators,
            "learning_rate": request.learning_rate,
            "min_samples_split": request.min_samples_split,
            "random_state": request.random_state
        }

        # Train model
        model = train_model(request.model_type, X_train, y_train, params)

        # Get predictions
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)

        # Calculate metrics
        train_metrics = calculate_metrics(y_train, train_pred)
        test_metrics = calculate_metrics(y_test, test_pred)

        # Get feature importance
        feature_importance = dict(zip(
            feature_names,
            [round(imp, 4) for imp in model.feature_importances_]
        ))

        # Sort by importance
        feature_importance = dict(sorted(
            feature_importance.items(),
            key=lambda x: x[1],
            reverse=True
        ))

        return {
            "model_type": request.model_type,
            "hyperparameters": {
                "max_depth": request.max_depth,
                "n_estimators": request.n_estimators if request.model_type != "decision_tree" else None,
                "learning_rate": request.learning_rate if request.model_type == "xgboost" else None,
                "min_samples_split": request.min_samples_split if request.model_type != "xgboost" else None
            },
            "train_metrics": train_metrics,
            "test_metrics": test_metrics,
            "feature_importance": feature_importance,
            "data_split": {
                "train_samples": len(X_train),
                "test_samples": len(X_test)
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/compare")
async def compare_models(
    max_depth: int = 5,
    n_estimators: int = 100,
    learning_rate: float = 0.1,
    min_samples_split: int = 2,
    test_size: float = 0.2,
    random_state: int = 42
):
    """Compare Decision Tree, Random Forest, and XGBoost models."""
    try:
        df, _ = load_and_prepare_data()
        X, y, feature_names = get_features_and_target(df)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state
        )

        results = {}

        for model_type in ["decision_tree", "random_forest", "xgboost"]:
            params = {
                "max_depth": max_depth,
                "n_estimators": n_estimators,
                "learning_rate": learning_rate,
                "min_samples_split": min_samples_split,
                "random_state": random_state
            }

            model = train_model(model_type, X_train, y_train, params)

            train_pred = model.predict(X_train)
            test_pred = model.predict(X_test)

            train_metrics = calculate_metrics(y_train, train_pred)
            test_metrics = calculate_metrics(y_test, test_pred)

            feature_importance = dict(zip(
                feature_names,
                [round(imp, 4) for imp in model.feature_importances_]
            ))

            results[model_type] = {
                "train_metrics": train_metrics,
                "test_metrics": test_metrics,
                "feature_importance": dict(sorted(
                    feature_importance.items(),
                    key=lambda x: x[1],
                    reverse=True
                ))
            }

        # Determine best model based on test R²
        best_model = max(results.keys(), key=lambda k: results[k]["test_metrics"]["r2"])

        return {
            "comparison_parameters": {
                "max_depth": max_depth,
                "n_estimators": n_estimators,
                "learning_rate": learning_rate,
                "min_samples_split": min_samples_split
            },
            "results": results,
            "best_model": {
                "name": best_model,
                "test_r2": results[best_model]["test_metrics"]["r2"],
                "test_mae": results[best_model]["test_metrics"]["mae"]
            },
            "data_split": {
                "train_samples": len(X_train),
                "test_samples": len(X_test)
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/trees_vs_estimators")
async def trees_vs_estimators(request: EstimatorsRequest):
    """Show how error changes with number of estimators."""
    try:
        df, _ = load_and_prepare_data()
        X, y, feature_names = get_features_and_target(df)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=0.2,
            random_state=request.random_state
        )

        results = []

        for n_est in request.estimator_range:
            params = {
                "max_depth": request.max_depth,
                "n_estimators": n_est,
                "learning_rate": request.learning_rate,
                "min_samples_split": 2,
                "random_state": request.random_state
            }

            model = train_model(request.model_type, X_train, y_train, params)

            train_pred = model.predict(X_train)
            test_pred = model.predict(X_test)

            train_metrics = calculate_metrics(y_train, train_pred)
            test_metrics = calculate_metrics(y_test, test_pred)

            results.append({
                "n_estimators": n_est,
                "train_mae": train_metrics["mae"],
                "test_mae": test_metrics["mae"],
                "train_rmse": train_metrics["rmse"],
                "test_rmse": test_metrics["rmse"],
                "train_r2": train_metrics["r2"],
                "test_r2": test_metrics["r2"]
            })

        # Find optimal number of estimators (best test R²)
        best_result = max(results, key=lambda x: x["test_r2"])

        return {
            "model_type": request.model_type,
            "max_depth": request.max_depth,
            "learning_rate": request.learning_rate if request.model_type == "xgboost" else None,
            "results": results,
            "optimal_n_estimators": best_result["n_estimators"],
            "best_test_metrics": {
                "mae": best_result["test_mae"],
                "rmse": best_result["test_rmse"],
                "r2": best_result["test_r2"]
            },
            "insights": {
                "diminishing_returns": "Notice how improvements slow down as n_estimators increases",
                "overfitting_check": "Compare train vs test metrics to detect overfitting"
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
