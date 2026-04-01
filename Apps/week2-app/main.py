from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeRegressor, export_text
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, make_scorer
import warnings
import os
import json
import io
import urllib.request

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
X_all = None
y_all = None
X_train = None
X_test = None
y_train = None
y_test = None
feature_names = []
encoders = {}

# Scorer for cross-validation (negative MAE since sklearn maximizes scores)
mae_scorer = make_scorer(mean_absolute_error, greater_is_better=False)


def load_and_prepare_data():
    global df, X_train, X_test, y_train, y_test, feature_names, encoders

    # Load from GitHub
    DATA_URL = "https://raw.githubusercontent.com/ucla-anderson-SSAI/SSAI/main/range_rover.csv"
    print(f"Loading data from {DATA_URL}")
    with urllib.request.urlopen(DATA_URL) as response:
        csv_data = response.read().decode('utf-8')
    df = pd.read_csv(io.StringIO(csv_data))
    print(f"Loaded {len(df)} rows")

    # FIX 1: Drop missing values so scikit-learn doesn't crash during .fit()
    possible_features = ['year', 'mileage', 'trim', 'state', 'color', 'interior', 'engine_liters', 'horsepower', 'n_owners', 'accident_history']
    price_col = 'sellingprice' if 'sellingprice' in df.columns else 'price'
    cols_to_check = [c for c in possible_features + [price_col] if c in df.columns]
    df.dropna(subset=cols_to_check, inplace=True)

    # Encode categorical variables — include 'interior' if present in real data
    categorical_cols = ['trim', 'state', 'color', 'interior']
    for col in categorical_cols:
        if col in df.columns:
            encoders[col] = LabelEncoder()
            df[f'{col}_enc'] = encoders[col].fit_transform(df[col].astype(str))

    # Build feature list from what's available
    feature_names = [f for f in possible_features if f in df.columns]
    
    # Update feature names to use the encoded versions for categorical columns
    feature_names = [f"{f}_enc" if f in categorical_cols else f for f in feature_names]

    X_all = df[feature_names].values
    y_all = df[price_col].values

    # Keep a train/test split for scatter plot predictions and tree visualization
    X_train, X_test, y_train, y_test = train_test_split(
        X_all, y_all, test_size=0.2, random_state=42
    )

    print(f"Features: {feature_names}")
    print(f"Target: {price_col}")
    print(f"Total samples: {len(X_all)}, Train: {len(X_train)}, Test: {len(X_test)}")
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
            n_jobs=2  # FIX 2: Limited to 2 threads to prevent OOM kills on Railway
        )
    else:  # xgboost
        model = XGBRegressor(
            n_estimators=request.n_estimators,
            max_depth=request.max_depth,
            learning_rate=request.learning_rate,
            subsample=request.subsample,
            random_state=42,
            n_jobs=2  # FIX 2: Limited to 2 threads to prevent OOM kills on Railway
        )

    # Cross-validation error (20-fold on full data)
    cv_scores = cross_val_score(model, X_all, y_all, cv=20, scoring=mae_scorer)
    cv_mae = -cv_scores.mean()  # Negate because sklearn returns negative MAE

    # Train on full training set for predictions, tree viz, and feature importances
    model.fit(X_train, y_train)

    # Train MAE (in-sample)
    train_preds = model.predict(X_train)
    test_preds = model.predict(X_test)  # Still needed for scatter plot

    train_mae = mean_absolute_error(y_train, train_preds)

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
    # Train MAE from fit model; CV MAE via 20-fold cross-validation at each step
    learning_curve = None
    if request.model_type in ('random_forest', 'xgboost'):
        learning_curve = []
        # Limit steps for RF since 20-fold CV per step is expensive
        max_steps = 10 if request.model_type == 'random_forest' else 20
        n_steps = min(request.n_estimators, max_steps)
        step_size = max(1, request.n_estimators // n_steps)
        steps = list(range(step_size, request.n_estimators + 1, step_size))
        if steps[-1] != request.n_estimators:
            steps.append(request.n_estimators)

        if request.model_type == 'xgboost':
            # Train MAE: use iteration_range on the already-trained model
            # CV MAE: run 20-fold CV at each step
            for n_est in steps:
                lc_train_preds = model.predict(X_train, iteration_range=(0, n_est))
                lc_train_mae = mean_absolute_error(y_train, lc_train_preds)
                lc_model = XGBRegressor(
                    n_estimators=n_est, max_depth=request.max_depth,
                    learning_rate=request.learning_rate, subsample=request.subsample,
                    random_state=42, n_jobs=2
                )
                lc_cv_scores = cross_val_score(lc_model, X_all, y_all, cv=20, scoring=mae_scorer)
                lc_cv_mae = -lc_cv_scores.mean()
                learning_curve.append({
                    'n_estimators': n_est,
                    'train_mae': round(lc_train_mae, 2),
                    'cv_mae': round(lc_cv_mae, 2)
                })
        else:
            # Random Forest: train MAE from subset of estimators, CV MAE via cross-validation
            all_train_preds = np.array([est.predict(X_train) for est in model.estimators_])
            for n_est in steps:
                lc_train_mae = mean_absolute_error(y_train, all_train_preds[:n_est].mean(axis=0))
                lc_model = RandomForestRegressor(
                    n_estimators=n_est, max_depth=request.max_depth,
                    max_features=request.max_features if request.max_features else 1.0,
                    random_state=42, n_jobs=2
                )
                lc_cv_scores = cross_val_score(lc_model, X_all, y_all, cv=20, scoring=mae_scorer)
                lc_cv_mae = -lc_cv_scores.mean()
                learning_curve.append({
                    'n_estimators': n_est,
                    'train_mae': round(lc_train_mae, 2),
                    'cv_mae': round(lc_cv_mae, 2)
                })

    # Depth curve: for decision tree, show MAE at different depths
    depth_curve = None
    if request.model_type == 'decision_tree':
        depth_curve = []
        for d in range(1, min(request.max_depth + 1, 21)):
            m = DecisionTreeRegressor(max_depth=d, random_state=42)
            m.fit(X_train, y_train)
            dc_train_mae = mean_absolute_error(y_train, m.predict(X_train))
            dc_cv_scores = cross_val_score(m, X_all, y_all, cv=20, scoring=mae_scorer)
            dc_cv_mae = -dc_cv_scores.mean()
            depth_curve.append({
                'depth': d,
                'train_mae': round(dc_train_mae, 2),
                'cv_mae': round(dc_cv_mae, 2)
            })

    # Boosting residuals: track CV MAE after each of the first N trees (XGBoost only)
    boosting_residuals = None
    if request.model_type == 'xgboost':
        num_shown = min(request.n_estimators, 5)
        rounds = list(range(1, num_shown + 1))
        boosting_residuals = []
        for n in rounds:
            br_model = XGBRegressor(
                n_estimators=n, max_depth=request.max_depth,
                learning_rate=request.learning_rate, subsample=request.subsample,
                random_state=42, n_jobs=2
            )
            br_cv_scores = cross_val_score(br_model, X_all, y_all, cv=20, scoring=mae_scorer)
            br_cv_mae = -br_cv_scores.mean()
            boosting_residuals.append({
                'tree': n,
                'cv_mae': round(br_cv_mae, 2)
            })

    response = {
        "model_type": request.model_type,
        "train_mae": round(train_mae, 2),
        "cv_mae": round(cv_mae, 2),
        "n_samples": len(X_all),
        "cv_folds": 20,
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