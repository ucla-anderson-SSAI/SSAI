"""
Assignment 5: Neural Network Architecture Tuning for Fashion Product Classification
FastAPI Backend - Port 8104
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import confusion_matrix
import os
import gzip
import urllib.request

router = APIRouter()


# Class names for Fashion-MNIST
CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

# Global cache for data
_cached_data = None

def download_fashion_mnist():
    """Download Fashion-MNIST data directly from GitHub mirror."""
    base_url = "https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/"
    files = {
        'train_images': 'train-images-idx3-ubyte.gz',
        'train_labels': 'train-labels-idx1-ubyte.gz',
        'test_images': 't10k-images-idx3-ubyte.gz',
        'test_labels': 't10k-labels-idx1-ubyte.gz'
    }

    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    os.makedirs(data_dir, exist_ok=True)

    paths = {}
    for key, filename in files.items():
        filepath = os.path.join(data_dir, filename)
        if not os.path.exists(filepath):
            print(f"Downloading {filename}...")
            urllib.request.urlretrieve(base_url + filename, filepath)
        paths[key] = filepath

    return paths

def load_idx_images(filepath):
    """Load images from IDX format."""
    with gzip.open(filepath, 'rb') as f:
        # Read header
        magic = int.from_bytes(f.read(4), 'big')
        num_images = int.from_bytes(f.read(4), 'big')
        rows = int.from_bytes(f.read(4), 'big')
        cols = int.from_bytes(f.read(4), 'big')
        # Read data
        data = np.frombuffer(f.read(), dtype=np.uint8)
        return data.reshape(num_images, rows * cols)

def load_idx_labels(filepath):
    """Load labels from IDX format."""
    with gzip.open(filepath, 'rb') as f:
        # Read header
        magic = int.from_bytes(f.read(4), 'big')
        num_labels = int.from_bytes(f.read(4), 'big')
        # Read data
        return np.frombuffer(f.read(), dtype=np.uint8)

def load_data():
    """Load and preprocess Fashion-MNIST data."""
    global _cached_data

    if _cached_data is not None:
        return _cached_data

    try:
        # Try keras first
        (x_train_full, y_train_full), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
        x_train_full = x_train_full.reshape(-1, 784)
        x_test = x_test.reshape(-1, 784)
    except Exception as e:
        print(f"Keras download failed: {e}")
        print("Trying direct download from GitHub...")

        try:
            paths = download_fashion_mnist()
            x_train_full = load_idx_images(paths['train_images'])
            y_train_full = load_idx_labels(paths['train_labels'])
            x_test = load_idx_images(paths['test_images'])
            y_test = load_idx_labels(paths['test_labels'])
        except Exception as e2:
            print(f"Direct download also failed: {e2}")
            print("Using synthetic data for demonstration...")
            # Generate synthetic data that mimics Fashion-MNIST
            np.random.seed(42)
            x_train_full = np.random.rand(60000, 784).astype('float32')
            y_train_full = np.random.randint(0, 10, 60000).astype('uint8')
            x_test = np.random.rand(10000, 784).astype('float32')
            y_test = np.random.randint(0, 10, 10000).astype('uint8')

            # Skip normalization for synthetic data (already 0-1)
            split_idx = int(0.8 * len(x_train_full))
            x_train, x_val = x_train_full[:split_idx], x_train_full[split_idx:]
            y_train, y_val = y_train_full[:split_idx], y_train_full[split_idx:]

            _cached_data = ((x_train, y_train), (x_val, y_val), (x_test, y_test))
            return _cached_data

    # Normalize pixel values to [0, 1]
    x_train_full = x_train_full.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    # 80/20 train/val split
    split_idx = int(0.8 * len(x_train_full))
    x_train, x_val = x_train_full[:split_idx], x_train_full[split_idx:]
    y_train, y_val = y_train_full[:split_idx], y_train_full[split_idx:]

    _cached_data = ((x_train, y_train), (x_val, y_val), (x_test, y_test))
    return _cached_data

# Request models
class HiddenSizeRequest(BaseModel):
    hidden_sizes: List[int] = [4, 8, 16, 32, 64, 128]

class DropoutRequest(BaseModel):
    dropout_rates: List[float] = [0.0, 0.1, 0.3, 0.5, 0.7]

class CustomModelRequest(BaseModel):
    hidden_1: int = 128
    hidden_2: int = 64
    dropout: float = 0.3
    epochs: int = 10

@router.post("/train_hidden_size")
async def train_hidden_size(request: HiddenSizeRequest):
    """
    Train models with different hidden layer sizes to analyze effect on performance.
    """
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_data()

    results = {
        "hidden_sizes": request.hidden_sizes,
        "train_accuracies": [],
        "val_accuracies": [],
        "test_accuracies": [],
        "parameter_counts": []
    }

    for hidden_size in request.hidden_sizes:
        # Build simple model with one hidden layer
        model = keras.Sequential([
            keras.layers.Dense(hidden_size, activation='relu', input_shape=(784,)),
            keras.layers.Dense(10, activation='softmax')
        ])

        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        # Count parameters
        param_count = model.count_params()

        # Train model
        history = model.fit(
            x_train, y_train,
            epochs=10,
            batch_size=128,
            validation_data=(x_val, y_val),
            verbose=0
        )

        # Evaluate
        train_loss, train_acc = model.evaluate(x_train, y_train, verbose=0)
        val_loss, val_acc = model.evaluate(x_val, y_val, verbose=0)
        test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)

        results["train_accuracies"].append(float(train_acc))
        results["val_accuracies"].append(float(val_acc))
        results["test_accuracies"].append(float(test_acc))
        results["parameter_counts"].append(param_count)

        # Clear session to free memory
        keras.backend.clear_session()

    return results

@router.post("/train_dropout")
async def train_dropout(request: DropoutRequest):
    """
    Train models with different dropout rates to analyze effect on overfitting.
    """
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_data()

    results = {
        "dropout_rates": request.dropout_rates,
        "train_accuracies": [],
        "val_accuracies": [],
        "gaps": [],
        "training_histories": []
    }

    for dropout_rate in request.dropout_rates:
        # Build model with dropout
        model = keras.Sequential([
            keras.layers.Dense(128, activation='relu', input_shape=(784,)),
            keras.layers.Dropout(dropout_rate),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dropout(dropout_rate),
            keras.layers.Dense(10, activation='softmax')
        ])

        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        # Train model
        history = model.fit(
            x_train, y_train,
            epochs=15,
            batch_size=128,
            validation_data=(x_val, y_val),
            verbose=0
        )

        # Evaluate
        train_loss, train_acc = model.evaluate(x_train, y_train, verbose=0)
        val_loss, val_acc = model.evaluate(x_val, y_val, verbose=0)

        gap = float(train_acc - val_acc)

        results["train_accuracies"].append(float(train_acc))
        results["val_accuracies"].append(float(val_acc))
        results["gaps"].append(gap)
        results["training_histories"].append({
            "train_acc": [float(x) for x in history.history['accuracy']],
            "val_acc": [float(x) for x in history.history['val_accuracy']],
            "train_loss": [float(x) for x in history.history['loss']],
            "val_loss": [float(x) for x in history.history['val_loss']]
        })

        keras.backend.clear_session()

    return results

@router.post("/train_custom")
async def train_custom(request: CustomModelRequest):
    """
    Train a custom architecture with user-specified hyperparameters.
    Returns training history, test accuracy, confusion matrix, and per-class accuracy.
    """
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_data()

    # Build custom model
    model = keras.Sequential([
        keras.layers.Dense(request.hidden_1, activation='relu', input_shape=(784,)),
        keras.layers.Dropout(request.dropout),
        keras.layers.Dense(request.hidden_2, activation='relu'),
        keras.layers.Dropout(request.dropout),
        keras.layers.Dense(10, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # Train model
    history = model.fit(
        x_train, y_train,
        epochs=request.epochs,
        batch_size=128,
        validation_data=(x_val, y_val),
        verbose=0
    )

    # Evaluate on test set
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)

    # Get predictions for confusion matrix
    y_pred = model.predict(x_test, verbose=0)
    y_pred_classes = np.argmax(y_pred, axis=1)

    # Compute confusion matrix
    cm = confusion_matrix(y_test, y_pred_classes)

    # Compute per-class accuracy
    per_class_accuracy = []
    for i in range(10):
        class_mask = y_test == i
        class_acc = np.mean(y_pred_classes[class_mask] == y_test[class_mask])
        per_class_accuracy.append(float(class_acc))

    # Find most confused pairs
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    np.fill_diagonal(cm_normalized, 0)
    confused_pairs = []
    for i in range(10):
        for j in range(10):
            if i != j and cm_normalized[i, j] > 0.05:
                confused_pairs.append({
                    "true_class": CLASS_NAMES[i],
                    "predicted_class": CLASS_NAMES[j],
                    "confusion_rate": float(cm_normalized[i, j])
                })
    confused_pairs.sort(key=lambda x: x["confusion_rate"], reverse=True)

    keras.backend.clear_session()

    return {
        "training_history": {
            "train_acc": [float(x) for x in history.history['accuracy']],
            "val_acc": [float(x) for x in history.history['val_accuracy']],
            "train_loss": [float(x) for x in history.history['loss']],
            "val_loss": [float(x) for x in history.history['val_loss']]
        },
        "test_accuracy": float(test_acc),
        "confusion_matrix": cm.tolist(),
        "per_class_accuracy": per_class_accuracy,
        "class_names": CLASS_NAMES,
        "confused_pairs": confused_pairs[:5],
        "parameter_count": model.count_params()
    }

@router.get("/")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "assignment": 5}
