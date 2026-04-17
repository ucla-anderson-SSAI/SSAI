import os
"""
Week 4: Neural Networks - Breast Cancer Diagnosis Classification
FastAPI Backend for training and evaluating neural networks
"""

import asyncio
import base64
import io
import time
from functools import partial
from typing import List, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
# Limit TF threads to prevent resource contention with concurrent users
tf.config.threading.set_intra_op_parallelism_threads(2)
tf.config.threading.set_inter_op_parallelism_threads(2)
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers, callbacks
from xgboost import XGBClassifier

# Suppress TensorFlow warnings
tf.get_logger().setLevel('ERROR')

app = FastAPI(
    title="Week 4: Neural Networks - Breast Cancer Diagnosis Classification",
    description="Train and evaluate neural networks on Wisconsin Breast Cancer dataset",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Class names for Breast Cancer Diagnosis
CLASS_NAMES = ["Benign", "Malignant"]

# Feature names for the dataset
FEATURE_NAMES = [
    "radius_mean", "texture_mean", "perimeter_mean", "area_mean", "smoothness_mean",
    "compactness_mean", "concavity_mean", "concave_points_mean", "symmetry_mean", "fractal_dimension_mean",
    "radius_se", "texture_se", "perimeter_se", "area_se", "smoothness_se",
    "compactness_se", "concavity_se", "concave_points_se", "symmetry_se", "fractal_dimension_se",
    "radius_worst", "texture_worst", "perimeter_worst", "area_worst", "smoothness_worst",
    "compactness_worst", "concavity_worst", "concave_points_worst", "symmetry_worst", "fractal_dimension_worst"
]

# Global cache for dataset
_dataset_cache = None


def load_cancer_data():
    """Load and preprocess Wisconsin Breast Cancer dataset with caching."""
    global _dataset_cache

    if _dataset_cache is not None:
        return _dataset_cache

    # Load dataset from CSV
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Try multiple paths for cancer.csv
    csv_path = os.path.join(script_dir, 'cancer.csv')  # Same directory (Cloud Run)
    if not os.path.exists(csv_path):
        csv_path = os.path.join(script_dir, '..', '..', 'cancer.csv')  # Relative path (Railway)
    if not os.path.exists(csv_path):
        csv_path = '/sessions/lucid-affectionate-lovelace/mnt/SSAI/cancer.csv'  # Fallback

    df = pd.read_csv(csv_path)

    # Convert diagnosis to binary (M=1, B=0)
    y = (df['diagnosis'] == 'M').astype(int).values

    # Get features (exclude id and diagnosis columns)
    feature_cols = [col for col in df.columns if col not in ['id', 'diagnosis']]
    X = df[feature_cols].values.astype('float32')

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype('float32')
    X_test = scaler.transform(X_test).astype('float32')

    _dataset_cache = {
        "x_train": X_train,
        "y_train": y_train,
        "x_test": X_test,
        "y_test": y_test,
        "scaler": scaler,
        "feature_names": feature_cols,
        "n_features": X_train.shape[1]
    }

    return _dataset_cache


# Pydantic models for request/response
class TrainRequest(BaseModel):
    hidden_layers: List[int] = Field(default=[64, 32], description="List of hidden layer sizes")
    activation: str = Field(default="relu", description="Activation function: relu, sigmoid, tanh")
    dropout_rate: float = Field(default=0.2, ge=0.0, le=0.5, description="Dropout rate")
    use_batch_norm: bool = Field(default=False, description="Whether to use batch normalization")
    learning_rate: float = Field(default=0.001, ge=0.0001, le=0.1, description="Learning rate")
    batch_size: int = Field(default=32, description="Batch size: 16, 32, 64, 128")
    epochs: int = Field(default=50, ge=10, le=100, description="Number of epochs")


class TrainResponse(BaseModel):
    train_accuracy_history: List[float]
    val_accuracy_history: List[float]
    train_loss_history: List[float]
    val_loss_history: List[float]
    test_accuracy: float
    test_loss: float
    confusion_matrix: List[List[int]]
    training_time: float
    model_summary: str
    total_params: int


class SampleData(BaseModel):
    features: List[float]
    feature_names: List[str]
    label: int
    label_name: str


class SampleDataResponse(BaseModel):
    samples: List[SampleData]
    class_names: List[str]
    feature_names: List[str]


class ModelComparison(BaseModel):
    model_name: str
    test_accuracy: float
    training_time: float
    description: str


class CompareResponse(BaseModel):
    comparisons: List[ModelComparison]
    best_model: str


class DatasetInfo(BaseModel):
    n_samples: int
    n_features: int
    n_train: int
    n_test: int
    class_distribution: dict
    feature_names: List[str]


def build_model(
    n_features: int,
    hidden_layers: List[int],
    activation: str,
    dropout_rate: float,
    use_batch_norm: bool,
    learning_rate: float
) -> keras.Model:
    """Build a neural network model with specified architecture."""

    model = models.Sequential()
    model.add(layers.Input(shape=(n_features,)))

    for i, units in enumerate(hidden_layers):
        model.add(layers.Dense(units))

        if use_batch_norm:
            model.add(layers.BatchNormalization())

        model.add(layers.Activation(activation))

        if dropout_rate > 0:
            model.add(layers.Dropout(dropout_rate))

    # Output layer - binary classification
    model.add(layers.Dense(1, activation="sigmoid"))

    # Compile model
    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model


def get_model_summary(model: keras.Model) -> str:
    """Get model summary as string."""
    string_list = []
    model.summary(print_fn=lambda x: string_list.append(x))
    return "\n".join(string_list)


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "Week 4: Neural Networks - Breast Cancer Diagnosis Classification",
        "class_names": CLASS_NAMES
    }


@app.get("/dataset_info", response_model=DatasetInfo)
async def get_dataset_info():
    """Return information about the dataset."""
    data = load_cancer_data()

    y_all = np.concatenate([data["y_train"], data["y_test"]])
    class_dist = {
        "Benign": int(np.sum(y_all == 0)),
        "Malignant": int(np.sum(y_all == 1))
    }

    return DatasetInfo(
        n_samples=len(y_all),
        n_features=data["n_features"],
        n_train=len(data["y_train"]),
        n_test=len(data["y_test"]),
        class_distribution=class_dist,
        feature_names=data["feature_names"]
    )


@app.get("/sample_data", response_model=SampleDataResponse)
async def get_sample_data():
    """Return sample data points with labels."""
    data = load_cancer_data()

    # Get 5 samples from each class (10 total)
    samples = []
    for class_idx in range(2):
        # Find indices of this class
        class_indices = np.where(data["y_test"] == class_idx)[0]
        # Take first 5 samples
        selected_indices = class_indices[:5]

        for idx in selected_indices:
            features = data["x_test"][idx]
            samples.append(SampleData(
                features=[float(f) for f in features],
                feature_names=data["feature_names"],
                label=int(class_idx),
                label_name=CLASS_NAMES[class_idx]
            ))

    return SampleDataResponse(
        samples=samples,
        class_names=CLASS_NAMES,
        feature_names=data["feature_names"]
    )


def _train_model_sync(request: TrainRequest) -> TrainResponse:
    """Synchronous training logic — runs in a thread pool to avoid blocking the event loop."""

    # Load data
    data = load_cancer_data()
    x_train = data["x_train"].copy()
    y_train = data["y_train"].copy()
    x_test = data["x_test"]
    y_test = data["y_test"]
    n_features = data["n_features"]

    # Split training data for validation
    val_split = 0.15
    val_size = int(len(x_train) * val_split)
    x_val = x_train[-val_size:]
    y_val = y_train[-val_size:]
    x_train = x_train[:-val_size]
    y_train = y_train[:-val_size]

    # Build model
    model = build_model(
        n_features=n_features,
        hidden_layers=request.hidden_layers,
        activation=request.activation,
        dropout_rate=request.dropout_rate,
        use_batch_norm=request.use_batch_norm,
        learning_rate=request.learning_rate
    )

    # Get model summary
    model_summary = get_model_summary(model)
    total_params = model.count_params()

    # Early stopping callback
    early_stop = callbacks.EarlyStopping(
        monitor="val_loss",
        patience=15,
        restore_best_weights=True
    )

    # Train model
    start_time = time.time()
    history = model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=request.epochs,
        batch_size=request.batch_size,
        callbacks=[early_stop],
        verbose=0
    )
    training_time = time.time() - start_time

    # Evaluate on test set
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)

    # Get predictions for confusion matrix
    y_pred_proba = model.predict(x_test, verbose=0)
    y_pred_classes = (y_pred_proba > 0.5).astype(int).flatten()
    cm = confusion_matrix(y_test, y_pred_classes)

    # Clear model from memory
    keras.backend.clear_session()

    return TrainResponse(
        train_accuracy_history=[float(x) for x in history.history["accuracy"]],
        val_accuracy_history=[float(x) for x in history.history["val_accuracy"]],
        train_loss_history=[float(x) for x in history.history["loss"]],
        val_loss_history=[float(x) for x in history.history["val_loss"]],
        test_accuracy=float(test_accuracy),
        test_loss=float(test_loss),
        confusion_matrix=cm.tolist(),
        training_time=float(training_time),
        model_summary=model_summary,
        total_params=total_params
    )


@app.post("/train", response_model=TrainResponse)
async def train_model(request: TrainRequest):
    """Train a custom neural network with specified hyperparameters."""

    # Validate activation function
    if request.activation not in ["relu", "sigmoid", "tanh"]:
        raise HTTPException(
            status_code=400,
            detail="Activation must be one of: relu, sigmoid, tanh"
        )

    # Validate batch size
    if request.batch_size not in [16, 32, 64, 128]:
        raise HTTPException(
            status_code=400,
            detail="Batch size must be one of: 16, 32, 64, 128"
        )

    # Run blocking TF training in a thread so health checks stay responsive
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, partial(_train_model_sync, request))


def _compare_models_sync() -> CompareResponse:
    """Synchronous compare logic — runs in a thread pool to avoid blocking the event loop."""

    data = load_cancer_data()
    x_train = data["x_train"]
    y_train = data["y_train"]
    x_test = data["x_test"]
    y_test = data["y_test"]
    n_features = data["n_features"]

    comparisons = []

    # 1. XGBoost
    print("Training XGBoost...")
    start_time = time.time()
    xgb_model = XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        n_jobs=2,
        random_state=42
    )
    xgb_model.fit(x_train, y_train)
    xgb_time = time.time() - start_time
    xgb_accuracy = xgb_model.score(x_test, y_test)
    comparisons.append(ModelComparison(
        model_name="XGBoost",
        test_accuracy=float(xgb_accuracy),
        training_time=float(xgb_time),
        description="Gradient boosted trees (100 estimators, max_depth=4)"
    ))

    # 2. Simple NN (single hidden layer)
    print("Training Simple NN...")
    keras.backend.clear_session()
    simple_nn = build_model(
        n_features=n_features,
        hidden_layers=[32],
        activation="relu",
        dropout_rate=0.0,
        use_batch_norm=False,
        learning_rate=0.001
    )
    start_time = time.time()
    simple_nn.fit(x_train, y_train, epochs=50, batch_size=32, verbose=0)
    simple_time = time.time() - start_time
    _, simple_accuracy = simple_nn.evaluate(x_test, y_test, verbose=0)
    comparisons.append(ModelComparison(
        model_name="Simple NN",
        test_accuracy=float(simple_accuracy),
        training_time=float(simple_time),
        description="Single hidden layer (32 units), no regularization"
    ))

    # 3. Deep NN (multiple hidden layers, no regularization)
    print("Training Deep NN...")
    keras.backend.clear_session()
    deep_nn = build_model(
        n_features=n_features,
        hidden_layers=[64, 32, 16],
        activation="relu",
        dropout_rate=0.0,
        use_batch_norm=False,
        learning_rate=0.001
    )
    start_time = time.time()
    deep_nn.fit(x_train, y_train, epochs=50, batch_size=32, verbose=0)
    deep_time = time.time() - start_time
    _, deep_accuracy = deep_nn.evaluate(x_test, y_test, verbose=0)
    comparisons.append(ModelComparison(
        model_name="Deep NN",
        test_accuracy=float(deep_accuracy),
        training_time=float(deep_time),
        description="Three hidden layers (64, 32, 16), no regularization"
    ))

    # 4. Regularized NN (with dropout and batch norm)
    print("Training Regularized NN...")
    keras.backend.clear_session()
    reg_nn = build_model(
        n_features=n_features,
        hidden_layers=[64, 32, 16],
        activation="relu",
        dropout_rate=0.3,
        use_batch_norm=True,
        learning_rate=0.001
    )
    start_time = time.time()
    reg_nn.fit(x_train, y_train, epochs=75, batch_size=32, verbose=0)
    reg_time = time.time() - start_time
    _, reg_accuracy = reg_nn.evaluate(x_test, y_test, verbose=0)
    comparisons.append(ModelComparison(
        model_name="Regularized NN",
        test_accuracy=float(reg_accuracy),
        training_time=float(reg_time),
        description="Three hidden layers with dropout (0.3) and batch normalization"
    ))

    # Clear memory
    keras.backend.clear_session()

    # Find best model
    best_model = max(comparisons, key=lambda x: x.test_accuracy)

    return CompareResponse(
        comparisons=comparisons,
        best_model=best_model.model_name
    )


@app.get("/compare", response_model=CompareResponse)
async def compare_models():
    """Compare XGBoost vs simple NN vs deep NN vs regularized NN."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _compare_models_sync)


# Serve frontend
app.mount("/static", StaticFiles(directory="."), name="static")

@app.get("/")
async def serve_frontend():
    """Serve the frontend application."""
    return FileResponse("index.html")


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
