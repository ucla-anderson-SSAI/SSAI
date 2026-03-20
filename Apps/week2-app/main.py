from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, export_text
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
import warnings
import os
import json

warnings.filterwarnings('ignore')

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
df = None
X_train = None
X_test = None
y_train = None
y_test = None
feature_names = []
encoders = {}


def load_and_prepare_data():
    global df, X_train, X_test, y_train, y_test, feature_names, encoders

    # Load from local CSV file (bundled with the app)
    csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "range_rover.csv")
    print(f"Loading data from {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows")

    # Encode categorical variables — include 'interior' if present in real data
    categorical_cols = ['trim', 'state', 'color', 'interior']
    for col in categorical_cols:
        if col in df.columns:
            encoders[col] = LabelEncoder()
            df[f'{col}_enc'] = encoders[col].fit_transform(df[col].astype(str))

    # Build feature list from what's available
    possible_features = ['year', 'mileage', 'trim_enc', 'state_enc', 'color_enc', 'interior_enc', 'engine_liters', 'horsepower', 'n_owners', 'accident_history']
    feature_names = [f for f in possible_features if f in df.columns]
    X = df[feature_names].values

    # Use sellingprice if it exists (real data), otherwise fall back to price (sample data)
    price_col = 'sellingprice' if 'sellingprice' in df.columns else 'price'
    y = df[price_col].values

    # Fixed train/test split for reproducibility
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(f"Features: {feature_names}")
    print(f"Target: {price_col}")
    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
    return df


@app.on_event("startup")
async def startup_event():
    load_and_prepare_data()


class TrainRequest(BaseModel):
    model_type: str  # 'decision_tree', 'random_forest', 'xgboost'
    max_depth: int = 6
    n_estimators: int = 100
    learning_rate: float = 0.1
    subsample: float = 0.8
    max_features: Optional[float] = 0.7  # For RF: fraction of features per split


def build_tree_structure(tree, feature_names, node_id=0, depth=0, max_display_depth=4):
    """Recursively extract tree structure for visualization"""
    if depth > max_display_depth:
        return None

    t = tree.tree_
    if node_id >= t.node_count:
        return None

    node = {
        'id': int(node_id),
        'depth': depth,
        'n_samples': int(t.n_node_samples[node_id]),
        'value': round(float(t.value[node_id][0][0]), 0),
    }

    if t.children_left[node_id] == t.children_right[node_id]:
        # Leaf node
        node['is_leaf'] = True
    else:
        node['is_leaf'] = False
        feat_idx = t.feature[node_id]
        node['feature'] = feature_names[feat_idx] if feat_idx < len(feature_names) else f'feature_{feat_idx}'
        node['threshold'] = round(float(t.threshold[node_id]), 1)

        left = build_tree_structure(tree, feature_names, t.children_left[node_id], depth + 1, max_display_depth)
        right = build_tree_structure(tree, feature_names, t.children_right[node_id], depth + 1, max_display_depth)

        if left:
            node['left'] = left
        if right:
            node['right'] = right

    return node


@app.post("/train")
async def train_model(request: TrainRequest):
    """Train a model and return results"""

    # Build the model
    if request.model_type == 'decision_tree':
        model = DecisionTreeRegressor(
            max_depth=request.max_depth,
            random_state=42
        )
    elif request.model_type == 'random_forest':
        max_feat = request.max_features if request.max_features else 1.0
        model = RandomForestRegressor(
            n_estimators=request.n_estimators,
            max_depth=request.max_depth,
            max_features=max_feat,
            random_state=42,
            n_jobs=-1
        )
    else:  # xgboost
        model = XGBRegressor(
            n_estimators=request.n_estimators,
            max_depth=request.max_depth,
            learning_rate=request.learning_rate,
            subsample=request.subsample,
            random_state=42,
            n_jobs=-1
        )

    # Train
    model.fit(X_train, y_train)

    # Predictions
    train_preds = model.predict(X_train)
    test_preds = model.predict(X_test)

    train_mae = mean_absolute_error(y_train, train_preds)
    test_mae = mean_absolute_error(y_test, test_preds)

    # Feature importances
    if hasattr(model, 'feature_importances_'):
        importances = dict(zip(feature_names, model.feature_importances_.tolist()))
    else:
        importances = {}

    # Sample predictions for scatter plot (limit to 200)
    sample_size = min(200, len(y_test))
    indices = np.random.RandomState(42).choice(len(y_test), sample_size, replace=False)

    # Tree structure (for decision tree visualization)
    tree_structure = None
    if request.model_type == 'decision_tree':
        tree_structure = build_tree_structure(model, feature_names, max_display_depth=4)
    elif request.model_type == 'random_forest':
        # Return structure of first tree for visualization
        tree_structure = build_tree_structure(model.estimators_[0], feature_names, max_display_depth=3)

    # Learning curve: train models with increasing n_estimators (for ensemble methods)
    learning_curve = None
    if request.model_type in ('random_forest', 'xgboost'):
        learning_curve = []
        n_steps = min(request.n_estimators, 20)
        step_size = max(1, request.n_estimators // n_steps)
        steps = list(range(step_size, request.n_estimators + 1, step_size))
        if steps[-1] != request.n_estimators:
            steps.append(request.n_estimators)

        if request.model_type == 'xgboost':
            # XGBoost: use iteration_range on the already-trained model (no retraining)
            for n_est in steps:
                lc_train_preds = model.predict(X_train, iteration_range=(0, n_est))
                lc_test_preds = model.predict(X_test, iteration_range=(0, n_est))
                lc_train_mae = mean_absolute_error(y_train, lc_train_preds)
                lc_test_mae = mean_absolute_error(y_test, lc_test_preds)
                learning_curve.append({
                    'n_estimators': n_est,
                    'train_mae': round(lc_train_mae, 2),
                    'test_mae': round(lc_test_mae, 2)
                })
        else:
            # Random Forest: average predictions from subsets of the already-trained estimators
            all_train_preds = np.array([est.predict(X_train) for est in model.estimators_])
            all_test_preds = np.array([est.predict(X_test) for est in model.estimators_])
            for n_est in steps:
                lc_train_mae = mean_absolute_error(y_train, all_train_preds[:n_est].mean(axis=0))
                lc_test_mae = mean_absolute_error(y_test, all_test_preds[:n_est].mean(axis=0))
                learning_curve.append({
                    'n_estimators': n_est,
                    'train_mae': round(lc_train_mae, 2),
                    'test_mae': round(lc_test_mae, 2)
                })

    # Depth curve: for decision tree, show MAE at different depths
    depth_curve = None
    if request.model_type == 'decision_tree':
        depth_curve = []
        for d in range(1, min(request.max_depth + 1, 21)):
            m = DecisionTreeRegressor(max_depth=d, random_state=42)
            m.fit(X_train, y_train)
            dc_train_mae = mean_absolute_error(y_train, m.predict(X_train))
            dc_test_mae = mean_absolute_error(y_test, m.predict(X_test))
            depth_curve.append({
                'depth': d,
                'train_mae': round(dc_train_mae, 2),
                'test_mae': round(dc_test_mae, 2)
            })

    # Boosting residuals: track real test MAE after each of the first N trees (XGBoost only)
    boosting_residuals = None
    if request.model_type == 'xgboost':
        num_shown = min(request.n_estimators, 5)
        # Sample rounds: first N trees shown in the diagram
        rounds = list(range(1, num_shown + 1))
        boosting_residuals = []
        for n in rounds:
            preds_at_n = model.predict(X_test, iteration_range=(0, n))
            mae_at_n = float(np.mean(np.abs(y_test - preds_at_n)))
            boosting_residuals.append({
                'tree': n,
                'test_mae': round(mae_at_n, 2)
            })

    response = {
        "model_type": request.model_type,
        "train_mae": round(train_mae, 2),
        "test_mae": round(test_mae, 2),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "feature_importances": {k: round(v, 4) for k, v in importances.items()},
        "predictions": [round(float(p), 2) for p in test_preds[indices]],
        "actuals": [round(float(a), 2) for a in y_test[indices]],
        "tree_structure": tree_structure,
        "learning_curve": learning_curve,
        "depth_curve": depth_curve,
        "boosting_residuals": boosting_residuals,
    }

    return response


@app.get("/data_info")
async def data_info():
    """Get information about the loaded data"""
    price_col = 'sellingprice' if 'sellingprice' in df.columns else 'price'
    return {
        "n_samples": len(df),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "n_features": len(feature_names),
        "features": feature_names,
        "feature_display_names": {
            'year': 'Year',
            'mileage': 'Mileage',
            'trim_enc': 'Trim',
            'state_enc': 'State',
            'color_enc': 'Color',
            'interior_enc': 'Interior'
        },
        "target": price_col,
        "price_range": [float(df[price_col].min()), float(df[price_col].max())],
        "price_mean": round(float(df[price_col].mean()), 2),
        "price_median": round(float(df[price_col].median()), 2)
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
