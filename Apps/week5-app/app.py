"""
Week 5: Convolutional Neural Networks – Lions, Tigers, and Bears!
FastAPI backend for CNN training on a CIFAR-100 subset.

Architecture mirrors Week 4: each training request is a single NDJSON-streamed
HTTP response. Cloud Run scales by spinning up container replicas — no shared
in-memory state, no session queues. Each request is fully self-contained.
"""

import asyncio
import json
import os
import queue
import random
import threading
import time
from typing import List, Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Force CPU and suppress TF noise BEFORE import
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# TensorFlow is LAZY-LOADED. Railway (frontend-only) never imports it.
_TF_INITIALIZED = False


def _load_tf():
    """Lazy-import TF + Keras. Cached after first call."""
    global _TF_INITIALIZED
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers

    if not _TF_INITIALIZED:
        tf.get_logger().setLevel("ERROR")
        try:
            tf.config.threading.set_intra_op_parallelism_threads(2)
            tf.config.threading.set_inter_op_parallelism_threads(2)
        except RuntimeError:
            pass
        # Fix random seeds so students get consistent results for the same config
        np.random.seed(42)
        random.seed(42)
        tf.random.set_seed(42)
        _TF_INITIALIZED = True

    return tf, keras, layers


# ============================================
# CIFAR-100 Class Configuration
# Lions, Tigers, and Bears — Oh My!
# ============================================
CIFAR100_TARGET_CLASSES = {
    3: 'bear',    # CIFAR-100 index 3  → our class 0
    43: 'lion',   # CIFAR-100 index 43 → our class 1
    88: 'tiger'   # CIFAR-100 index 88 → our class 2
}
CLASS_NAMES = ['bear', 'lion', 'tiger']
NUM_CLASSES = 3

# Training caps
DEFAULT_EPOCHS = 10
MAX_EPOCHS = 20
DEFAULT_NUM_SAMPLES = 300
MAX_SAMPLES = 500

# Transfer learning caps
TRANSFER_MAX_EPOCHS = 3
TRANSFER_IMAGE_SIZE = 96  # Upscale 32×32 → 96×96 for MobileNetV2

# Global dataset cache
_dataset_cache = None
_X_TRAIN_96 = None
_X_TEST_96 = None


app = FastAPI(
    title="Week 5: CNNs – Lions, Tigers, and Bears",
    description="Train CNNs and MobileNetV2 transfer learning on a CIFAR-100 subset",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================
# Image helpers
# ============================================
def image_array_to_base64(img_array):
    """Convert numpy array (32×32×3, normalized 0-1) to base64 data URL."""
    import base64
    from io import BytesIO
    from PIL import Image

    img_uint8 = (img_array * 255).astype(np.uint8)
    img = Image.fromarray(img_uint8, mode='RGB')
    buffer = BytesIO()
    img.save(buffer, format='PNG')
    buffer.seek(0)
    img_base64 = base64.b64encode(buffer.read()).decode('utf-8')
    return f'data:image/png;base64,{img_base64}'


# ============================================
# Data loading
# ============================================
def load_data():
    """Load and cache CIFAR-100 filtered to lions, tigers, bears."""
    global _dataset_cache
    if _dataset_cache is not None:
        return _dataset_cache

    tf, keras, layers = _load_tf()

    print("Loading CIFAR-100 dataset...")
    from tensorflow.keras.datasets import cifar100
    (X_train_full, y_train_full), (X_test_full, y_test_full) = cifar100.load_data(label_mode='fine')

    y_train_full = y_train_full.flatten()
    y_test_full = y_test_full.flatten()

    target_indices = list(CIFAR100_TARGET_CLASSES.keys())

    train_mask = np.isin(y_train_full, target_indices)
    X_train_filtered = X_train_full[train_mask].astype('float32') / 255.0
    y_train_filtered = y_train_full[train_mask]

    test_mask = np.isin(y_test_full, target_indices)
    X_test_filtered = X_test_full[test_mask].astype('float32') / 255.0
    y_test_filtered = y_test_full[test_mask]

    # Remap labels: {3: 0, 43: 1, 88: 2} → bear=0, lion=1, tiger=2
    label_map = {old_idx: new_idx for new_idx, old_idx in enumerate(sorted(target_indices))}
    Y_TRAIN = np.array([label_map[y] for y in y_train_filtered])
    Y_TEST = np.array([label_map[y] for y in y_test_filtered])

    _dataset_cache = {
        "X_train": X_train_filtered,
        "Y_train": Y_TRAIN,
        "X_test": X_test_filtered,
        "Y_test": Y_TEST,
    }

    for i, name in enumerate(CLASS_NAMES):
        train_count = np.sum(Y_TRAIN == i)
        test_count = np.sum(Y_TEST == i)
        print(f"  {name}: {train_count} train, {test_count} test")
    print(f"Total: {len(X_train_filtered)} training, {len(X_test_filtered)} test images")

    return _dataset_cache


def get_upscaled_data():
    """Lazily upscale CIFAR images to 96×96 for transfer learning."""
    global _X_TRAIN_96, _X_TEST_96
    if _X_TRAIN_96 is not None:
        return _X_TRAIN_96, _X_TEST_96

    tf, keras, layers = _load_tf()
    data = load_data()

    print("Upscaling images to 96×96 for transfer learning...")
    _X_TRAIN_96 = tf.image.resize(data["X_train"], [TRANSFER_IMAGE_SIZE, TRANSFER_IMAGE_SIZE]).numpy()
    _X_TEST_96 = tf.image.resize(data["X_test"], [TRANSFER_IMAGE_SIZE, TRANSFER_IMAGE_SIZE]).numpy()
    print(f"Upscaled: train {_X_TRAIN_96.shape}, test {_X_TEST_96.shape}")

    return _X_TRAIN_96, _X_TEST_96


# ============================================
# Pydantic models
# ============================================
class TrainRequest(BaseModel):
    convBlocks: int = Field(default=2, ge=1, le=5)
    filters: int = Field(default=32)
    kernelSize: int = Field(default=3)
    batchNorm: bool = Field(default=False)
    dropout: float = Field(default=0.0, ge=0.0, le=0.5)
    epochs: int = Field(default=10, ge=1, le=20)
    numSamples: int = Field(default=300)


class TransferRequest(BaseModel):
    strategy: str = Field(default="finetune")
    freezeLayers: int = Field(default=90, ge=0, le=100)
    numSamples: int = Field(default=300)
    epochs: int = Field(default=3, ge=1, le=3)


# ============================================
# CNN model building
# ============================================
def build_cnn_model(config: dict):
    """Build CNN model based on frontend config."""
    tf, keras, layers = _load_tf()

    model = keras.Sequential()
    for i in range(config['convBlocks']):
        filters = config['filters'] * (2 ** min(i, 2))

        if i == 0:
            model.add(layers.Conv2D(
                filters, (config['kernelSize'], config['kernelSize']),
                padding='same', activation='relu',
                input_shape=(32, 32, 3), name=f'conv2d_{i}'
            ))
        else:
            model.add(layers.Conv2D(
                filters, (config['kernelSize'], config['kernelSize']),
                padding='same', activation='relu', name=f'conv2d_{i}'
            ))

        if config.get('batchNorm', False):
            model.add(layers.BatchNormalization(name=f'bn_{i}'))

        model.add(layers.MaxPooling2D(pool_size=(2, 2), name=f'pool_{i}'))

        if config.get('dropout', 0) > 0:
            model.add(layers.Dropout(config['dropout']))

    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(NUM_CLASSES, activation='softmax'))

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


def extract_filters(model, num_blocks):
    """Extract filter weights from trained model."""
    filters = {}
    for i in range(num_blocks):
        try:
            layer = model.get_layer(f'conv2d_{i}')
            weights = layer.get_weights()[0]
            h, w, in_c, out_f = weights.shape
            layer_filters = []
            for f in range(min(out_f, 16)):
                filter_data = []
                for y in range(h):
                    row = []
                    for x in range(w):
                        channels = [float(weights[y, x, c, f]) for c in range(in_c)]
                        row.append(channels)
                    filter_data.append(row)
                layer_filters.append(filter_data)
            filters[f'layer{i + 1}'] = layer_filters
        except Exception as e:
            print(f"Error extracting layer {i}: {e}")
    return filters


# ============================================
# Transfer learning model building
# ============================================
def build_transfer_model(config: dict):
    """Build MobileNetV2 transfer learning model."""
    tf, keras, layers = _load_tf()

    freeze_pct = config.get('freezeLayers', 90) / 100.0
    strategy = config.get('strategy', 'finetune')

    base_model = keras.applications.MobileNetV2(
        input_shape=(TRANSFER_IMAGE_SIZE, TRANSFER_IMAGE_SIZE, 3),
        include_top=False,
        weights='imagenet'
    )

    total_layers = len(base_model.layers)
    freeze_count = int(total_layers * freeze_pct)

    if strategy == 'feature':
        base_model.trainable = False
    else:
        base_model.trainable = True
        for i, layer in enumerate(base_model.layers):
            layer.trainable = i >= freeze_count

    trainable_count = sum(1 for l in base_model.layers if l.trainable)
    frozen_count = total_layers - trainable_count

    inputs = keras.Input(shape=(TRANSFER_IMAGE_SIZE, TRANSFER_IMAGE_SIZE, 3))
    x = keras.applications.mobilenet_v2.preprocess_input(inputs)
    x = base_model(x, training=(strategy == 'finetune'))
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(NUM_CLASSES, activation='softmax')(x)

    model = keras.Model(inputs, outputs)

    lr = 0.0001 if strategy == 'finetune' else 0.001
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model, total_layers, frozen_count, trainable_count


# ============================================
# Streaming training (Week 4 pattern)
# ============================================
def _run_cnn_training(config: dict, q: "queue.Queue"):
    """Run CNN training in a thread; push NDJSON messages onto q."""
    try:
        tf, keras, layers = _load_tf()
        data = load_data()

        num_samples = min(config.get('numSamples', DEFAULT_NUM_SAMPLES), MAX_SAMPLES)
        epochs = min(config.get('epochs', DEFAULT_EPOCHS), MAX_EPOCHS)

        # Sample balanced across 3 classes
        samples_per_class = num_samples // NUM_CLASSES
        indices = []
        for class_idx in range(NUM_CLASSES):
            class_indices = np.where(data["Y_train"] == class_idx)[0]
            selected = np.random.choice(
                class_indices,
                size=min(samples_per_class, len(class_indices)),
                replace=False
            )
            indices.extend(selected)
        np.random.shuffle(indices)

        X_train = data["X_train"][indices]
        y_train = data["Y_train"][indices]

        # Validation set
        val_indices = np.random.choice(len(data["X_test"]), size=min(300, len(data["X_test"])), replace=False)
        X_val = data["X_test"][val_indices]
        y_val = data["Y_test"][val_indices]

        model = build_cnn_model(config)

        # Per-epoch streaming callback
        class StreamCallback(keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                logs = logs or {}
                q.put({
                    "type": "epoch",
                    "epoch": int(epoch) + 1,
                    "total_epochs": epochs,
                    "trainAcc": float(logs.get("accuracy", 0.0) * 100),
                    "valAcc": float(logs.get("val_accuracy", 0.0) * 100),
                    "trainLoss": float(logs.get("loss", 0.0)),
                    "valLoss": float(logs.get("val_loss", 0.0)),
                })

        start_time = time.time()
        model.fit(
            X_train, y_train,
            batch_size=64,
            epochs=epochs,
            validation_data=(X_val, y_val),
            callbacks=[StreamCallback()],
            verbose=0
        )
        training_time = time.time() - start_time

        # Extract learned filters
        filters = extract_filters(model, config['convBlocks'])

        # Evaluate on full test set
        test_loss, test_acc = model.evaluate(data["X_test"], data["Y_test"], verbose=0)

        # Sample predictions — a few examples per class
        sample_predictions = []
        for class_idx in range(NUM_CLASSES):
            class_test_indices = np.where(data["Y_test"] == class_idx)[0]
            if len(class_test_indices) > 0:
                selected = np.random.choice(class_test_indices, size=min(3, len(class_test_indices)), replace=False)
                for idx in selected:
                    pred = model.predict(data["X_test"][idx:idx+1], verbose=0)[0]
                    sample_predictions.append({
                        'true': int(data["Y_test"][idx]),
                        'predicted': int(np.argmax(pred)),
                        'confidence': float(np.max(pred)),
                        'probabilities': [float(p) for p in pred],
                        'imageData': image_array_to_base64(data["X_test"][idx])
                    })

        keras.backend.clear_session()

        q.put({
            "type": "final",
            "test_accuracy": float(test_acc * 100),
            "training_time": float(training_time),
            "filters": filters,
            "sample_predictions": sample_predictions[:8],
            "classes": CLASS_NAMES,
        })

    except Exception as exc:
        q.put({"type": "error", "detail": str(exc)})
    finally:
        q.put(None)  # sentinel


def _run_transfer_training(config: dict, q: "queue.Queue"):
    """Run MobileNetV2 transfer learning in a thread; push NDJSON messages onto q."""
    try:
        tf, keras, layers = _load_tf()
        X_train_96, X_test_96 = get_upscaled_data()
        data = load_data()

        epochs = min(config.get('epochs', TRANSFER_MAX_EPOCHS), TRANSFER_MAX_EPOCHS)
        num_samples = min(config.get('numSamples', 300), MAX_SAMPLES)

        # Sample balanced training data
        samples_per_class = num_samples // NUM_CLASSES
        indices = []
        for class_idx in range(NUM_CLASSES):
            class_indices = np.where(data["Y_train"] == class_idx)[0]
            selected = np.random.choice(
                class_indices,
                size=min(samples_per_class, len(class_indices)),
                replace=False
            )
            indices.extend(selected)
        np.random.shuffle(indices)

        X_train = X_train_96[indices]
        y_train = data["Y_train"][indices]

        val_indices = np.random.choice(len(X_test_96), size=min(300, len(X_test_96)), replace=False)
        X_val = X_test_96[val_indices]
        y_val = data["Y_test"][val_indices]

        model, total_layers, frozen_count, trainable_count = build_transfer_model(config)

        model_info = {
            'total_layers': total_layers,
            'frozen_layers': frozen_count,
            'trainable_layers': trainable_count,
            'total_params': int(model.count_params()),
            'trainable_params': int(sum(
                keras.backend.count_params(w) for w in model.trainable_weights
            ))
        }

        # Send model info immediately so frontend can display it during training
        q.put({
            "type": "model_info",
            "model_info": model_info,
        })

        class StreamCallback(keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                logs = logs or {}
                q.put({
                    "type": "epoch",
                    "epoch": int(epoch) + 1,
                    "total_epochs": epochs,
                    "trainAcc": float(logs.get("accuracy", 0.0) * 100),
                    "valAcc": float(logs.get("val_accuracy", 0.0) * 100),
                    "trainLoss": float(logs.get("loss", 0.0)),
                    "valLoss": float(logs.get("val_loss", 0.0)),
                })

        start_time = time.time()
        model.fit(
            X_train, y_train,
            batch_size=32,
            epochs=epochs,
            validation_data=(X_val, y_val),
            callbacks=[StreamCallback()],
            verbose=0
        )
        training_time = time.time() - start_time

        # Evaluate on full test set
        test_loss, test_acc = model.evaluate(X_test_96, data["Y_test"], verbose=0)

        # Sample predictions
        sample_predictions = []
        for class_idx in range(NUM_CLASSES):
            class_test_indices = np.where(data["Y_test"] == class_idx)[0]
            if len(class_test_indices) > 0:
                selected = np.random.choice(class_test_indices, size=min(3, len(class_test_indices)), replace=False)
                for idx in selected:
                    pred = model.predict(X_test_96[idx:idx+1], verbose=0)[0]
                    sample_predictions.append({
                        'true': int(data["Y_test"][idx]),
                        'predicted': int(np.argmax(pred)),
                        'confidence': float(np.max(pred)),
                        'probabilities': [float(p) for p in pred],
                        'imageData': image_array_to_base64(data["X_test"][idx])
                    })

        keras.backend.clear_session()

        q.put({
            "type": "final",
            "test_accuracy": float(test_acc * 100),
            "training_time": float(training_time),
            "model_info": model_info,
            "sample_predictions": sample_predictions[:8],
            "classes": CLASS_NAMES,
        })

    except Exception as exc:
        q.put({"type": "error", "detail": str(exc)})
    finally:
        q.put(None)  # sentinel


async def _stream_training(run_fn, config: dict):
    """Async generator yielding NDJSON lines as training progresses."""
    loop = asyncio.get_event_loop()
    q: "queue.Queue" = queue.Queue()

    thread = threading.Thread(target=run_fn, args=(config, q), daemon=True)
    thread.start()

    while True:
        item = await loop.run_in_executor(None, q.get)
        if item is None:
            break
        yield (json.dumps(item) + "\n").encode("utf-8")


# ============================================
# API Endpoints
# ============================================
@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "ok",
        "tf_loaded": _TF_INITIALIZED,
        "dataset": "Lions, Tigers, and Bears (CIFAR-100 subset)",
        "classes": CLASS_NAMES,
        "architecture": "stateless-streaming",
        "optimized_for": "100+ students via Cloud Run autoscaling",
    }


@app.post("/api/warmup")
async def warmup():
    """Pre-load TensorFlow and dataset."""
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, load_data)
    tf, keras, layers = _load_tf()
    data = load_data()
    return {
        "status": "ok",
        "tf_version": tf.__version__,
        "classes": CLASS_NAMES,
        "train_samples": len(data["X_train"]),
        "test_samples": len(data["X_test"]),
    }


@app.get("/api/classes")
async def get_classes():
    """Get the class names."""
    return {
        "classes": CLASS_NAMES,
        "num_classes": NUM_CLASSES,
        "dataset": "CIFAR-100 subset",
    }


@app.get("/api/sample-images")
async def get_sample_images():
    """Get sample training images for each class."""
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, load_data)
    data = load_data()

    sample_images = []
    for class_idx in range(NUM_CLASSES):
        class_train_indices = np.where(data["Y_train"] == class_idx)[0]
        if len(class_train_indices) > 0:
            selected = np.random.choice(class_train_indices, size=min(3, len(class_train_indices)), replace=False)
            for idx in selected:
                sample_images.append({
                    'imageData': image_array_to_base64(data["X_train"][idx]),
                    'className': CLASS_NAMES[class_idx],
                    'classIndex': int(class_idx),
                })

    return {"samples": sample_images[:8], "classes": CLASS_NAMES}


@app.post("/api/train")
async def train_model(request: TrainRequest):
    """Stream CNN training progress as NDJSON.

    Response body is a sequence of newline-delimited JSON objects:
      - {"type": "epoch", "epoch": N, "trainAcc": ..., "valAcc": ..., ...}
      - {"type": "final", "test_accuracy": ..., "filters": ..., ...}
      - {"type": "error", "detail": "..."}  (on failure)
    """
    config = request.model_dump()
    config['epochs'] = min(config.get('epochs', DEFAULT_EPOCHS), MAX_EPOCHS)
    config['numSamples'] = min(config.get('numSamples', DEFAULT_NUM_SAMPLES), MAX_SAMPLES)

    headers = {
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no",
    }
    return StreamingResponse(
        _stream_training(_run_cnn_training, config),
        media_type="application/x-ndjson",
        headers=headers,
    )


@app.post("/api/transfer/train")
async def train_transfer(request: TransferRequest):
    """Stream MobileNetV2 transfer learning progress as NDJSON.

    Response body is a sequence of newline-delimited JSON objects:
      - {"type": "model_info", "model_info": {...}}
      - {"type": "epoch", "epoch": N, "trainAcc": ..., "valAcc": ..., ...}
      - {"type": "final", "test_accuracy": ..., "model_info": ..., ...}
      - {"type": "error", "detail": "..."}  (on failure)
    """
    config = request.model_dump()
    config['epochs'] = min(config.get('epochs', TRANSFER_MAX_EPOCHS), TRANSFER_MAX_EPOCHS)
    config['numSamples'] = min(config.get('numSamples', 300), MAX_SAMPLES)

    headers = {
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no",
    }
    return StreamingResponse(
        _stream_training(_run_transfer_training, config),
        media_type="application/x-ndjson",
        headers=headers,
    )


BASE_DIR = os.path.dirname(os.path.abspath(__file__))


@app.get("/")
async def serve_frontend():
    """Serve the frontend application."""
    return FileResponse(os.path.join(BASE_DIR, "index.html"))


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
