import os
"""
Week 4: Neural Networks - MNIST Digit Classification
FastAPI backend for training and evaluating fully-connected networks on MNIST.

Mirrors the Notebooks/Week4.ipynb setup: a 2,000 train / 500 test subsample
of MNIST so training is fast enough to run live in the browser.
"""

import asyncio
import json
import queue
import random
import threading
import time
from functools import partial
from typing import List

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sklearn.metrics import confusion_matrix

# TensorFlow and XGBoost are LAZY-LOADED inside the functions that need them.
# Rationale: this same file runs on Cloud Run (full backend) AND on Railway
# (frontend-only serving of index.html). Railway's smaller tier can't handle
# TF at startup. By deferring the import, Railway boots in <1s and never
# touches TF unless someone actually hits /train or /compare (which they
# won't, because index.html points API_BASE at Cloud Run).
_TF_INITIALIZED = False


def _load_tf():
    """Lazy-import TF + Keras. Cached after first call. Returns a namespace."""
    global _TF_INITIALIZED
    import tensorflow as tf  # noqa: WPS433 (intentional lazy import)
    from tensorflow import keras
    from tensorflow.keras import layers, models, optimizers, callbacks

    if not _TF_INITIALIZED:
        tf.get_logger().setLevel("ERROR")
        try:
            # Limit threads to prevent resource contention with concurrent users.
            # Can only be set once per process, hence the try/except.
            tf.config.threading.set_intra_op_parallelism_threads(2)
            tf.config.threading.set_inter_op_parallelism_threads(2)
        except RuntimeError:
            pass
        _TF_INITIALIZED = True

    return tf, keras, layers, models, optimizers, callbacks


def _load_xgboost():
    """Lazy-import XGBoost."""
    from xgboost import XGBClassifier  # noqa: WPS433
    return XGBClassifier


app = FastAPI(
    title="Week 4: Neural Networks - MNIST Digit Classification",
    description="Train and evaluate fully-connected networks on a 2k/500 MNIST subsample",
    version="2.0.0",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 10 digit classes
CLASS_NAMES = [str(i) for i in range(10)]

# 784 = 28 * 28 flattened pixels
N_FEATURES = 784
FEATURE_NAMES = [f"pixel_{i}" for i in range(N_FEATURES)]

# Subsample sizes — match the Week 4 notebook
N_TRAIN = 2000
N_TEST = 500

# Global cache for dataset
_dataset_cache = None


def load_mnist_data():
    """Load and cache a deterministic 2k/500 MNIST subsample, scaled to [0, 1]."""
    global _dataset_cache

    if _dataset_cache is not None:
        return _dataset_cache

    _tf, keras, _layers, _models, _optimizers, _callbacks = _load_tf()
    (x_train_full, y_train_full), (x_test_full, y_test_full) = keras.datasets.mnist.load_data()

    # Deterministic subsample so dataset_info / sample_data / train all agree.
    rng = np.random.default_rng(42)
    train_idx = rng.choice(len(x_train_full), size=N_TRAIN, replace=False)
    test_idx = rng.choice(len(x_test_full), size=N_TEST, replace=False)

    x_train = x_train_full[train_idx].reshape(-1, N_FEATURES).astype("float32") / 255.0
    y_train = y_train_full[train_idx].astype("int32")
    x_test = x_test_full[test_idx].reshape(-1, N_FEATURES).astype("float32") / 255.0
    y_test = y_test_full[test_idx].astype("int32")

    _dataset_cache = {
        "x_train": x_train,
        "y_train": y_train,
        "x_test": x_test,
        "y_test": y_test,
        "feature_names": FEATURE_NAMES,
        "n_features": N_FEATURES,
    }

    return _dataset_cache


# Pydantic models for request/response
class TrainRequest(BaseModel):
    hidden_layers: List[int] = Field(default=[64, 32], description="List of hidden layer sizes")
    activation: str = Field(default="relu", description="Activation function: relu, sigmoid, tanh")
    learning_rate: float = Field(default=0.001, ge=0.0001, le=0.1, description="Learning rate")
    batch_size: int = Field(default=32, description="Batch size: 16, 32, 64, 128")
    epochs: int = Field(default=50, ge=10, le=200, description="Number of epochs")
    use_early_stopping: bool = Field(default=True, description="Halt when val_loss stops improving")


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
    learning_rate: float,
):
    """Build a fully-connected MNIST classifier."""
    _tf, _keras, layers, models, optimizers, _cb = _load_tf()

    model = models.Sequential()
    model.add(layers.Input(shape=(n_features,)))

    for units in hidden_layers:
        model.add(layers.Dense(units, activation=activation))

    # 10-way softmax over digit classes
    model.add(layers.Dense(10, activation="softmax"))

    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def get_model_summary(model) -> str:
    """Get model summary as string."""
    string_list = []
    model.summary(print_fn=lambda x: string_list.append(x))
    return "\n".join(string_list)


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "Week 4: Neural Networks - MNIST Digit Classification",
        "class_names": CLASS_NAMES,
    }


@app.get("/dataset_info", response_model=DatasetInfo)
async def get_dataset_info():
    """Return information about the dataset."""
    data = load_mnist_data()

    y_all = np.concatenate([data["y_train"], data["y_test"]])
    class_dist = {str(i): int(np.sum(y_all == i)) for i in range(10)}

    return DatasetInfo(
        n_samples=len(y_all),
        n_features=data["n_features"],
        n_train=len(data["y_train"]),
        n_test=len(data["y_test"]),
        class_distribution=class_dist,
        feature_names=data["feature_names"],
    )


@app.get("/sample_data", response_model=SampleDataResponse)
async def get_sample_data():
    """Return one sample image per digit (10 total) from the test set."""
    data = load_mnist_data()

    samples = []
    for class_idx in range(10):
        class_indices = np.where(data["y_test"] == class_idx)[0]
        if len(class_indices) == 0:
            continue
        idx = int(class_indices[0])
        features = data["x_test"][idx]
        samples.append(SampleData(
            features=[float(f) for f in features],
            feature_names=data["feature_names"],
            label=int(class_idx),
            label_name=CLASS_NAMES[class_idx],
        ))

    return SampleDataResponse(
        samples=samples,
        class_names=CLASS_NAMES,
        feature_names=data["feature_names"],
    )


def _run_training(request: TrainRequest, q: "queue.Queue"):
    """Run Keras training in a thread; push per-epoch + final messages onto `q`.

    Communicates with the async generator via a thread-safe queue. The final
    sentinel (None) tells the generator the stream is complete.
    """
    try:
        _tf, keras, _layers, _models, _optimizers, callbacks = _load_tf()

        # Fix all random seeds at the start of every request so every student
        # running the same hyperparameters gets identical results (test
        # accuracy, confusion matrix, epoch-stop count). Re-seeding per request
        # is essential: the global RNG state advances across requests, so
        # without this the Nth student's run would diverge from the 1st.
        # Seeds must be set BEFORE build_model (weight init) and BEFORE
        # model.fit (batch shuffling).
        seed = 42
        os.environ["PYTHONHASHSEED"] = str(seed)
        random.seed(seed)
        np.random.seed(seed)
        keras.utils.set_random_seed(seed)
        try:
            _tf.config.experimental.enable_op_determinism()
        except Exception:  # noqa: BLE001
            # Older TF versions don't have this; CPU runs are still
            # mostly deterministic with the seeds above.
            pass

        # Load data
        data = load_mnist_data()
        x_train = data["x_train"].copy()
        y_train = data["y_train"].copy()
        x_test = data["x_test"]
        y_test = data["y_test"]
        n_features = data["n_features"]

        # Split off a validation set from the training data
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
            learning_rate=request.learning_rate,
        )

        model_summary = get_model_summary(model)
        total_params = int(model.count_params())

        # Per-epoch streaming callback — enqueues a message at the end of each epoch.
        class StreamCallback(callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                logs = logs or {}
                q.put({
                    "type": "epoch",
                    "epoch": int(epoch) + 1,
                    "train_loss": float(logs.get("loss", 0.0)),
                    "val_loss": float(logs.get("val_loss", 0.0)),
                    "train_accuracy": float(logs.get("accuracy", 0.0)),
                    "val_accuracy": float(logs.get("val_accuracy", 0.0)),
                })

        # Early stopping is opt-in from the UI. Patience is hardcoded to 10:
        # training halts after 10 epochs of no val_loss improvement and the
        # best weights are restored.
        fit_callbacks = [StreamCallback()]
        if request.use_early_stopping:
            fit_callbacks.append(callbacks.EarlyStopping(
                monitor="val_loss",
                patience=10,
                restore_best_weights=True,
            ))

        start_time = time.time()
        history = model.fit(
            x_train, y_train,
            validation_data=(x_val, y_val),
            epochs=request.epochs,
            batch_size=request.batch_size,
            callbacks=fit_callbacks,
            verbose=0,
        )
        training_time = time.time() - start_time

        # Test evaluation + confusion matrix (runs after best weights are restored)
        test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
        y_pred_proba = model.predict(x_test, verbose=0)
        y_pred_classes = np.argmax(y_pred_proba, axis=1)
        cm = confusion_matrix(y_test, y_pred_classes, labels=list(range(10)))

        # Sample 5 correctly-classified and 5 misclassified test images so the
        # UI can show side-by-side examples of what the model got right vs.
        # wrong. We use a fresh RNG seeded with a fixed value so that (a) the
        # selection is reproducible across students, and (b) the picks are
        # spread across the whole test set instead of always the first few.
        sampler = np.random.default_rng(42)
        correct_indices = np.where(y_pred_classes == y_test)[0]
        misclass_indices = np.where(y_pred_classes != y_test)[0]

        def _sample_examples(indices, n):
            if len(indices) == 0:
                return []
            chosen = sampler.choice(
                indices, size=min(n, len(indices)), replace=False
            )
            out = []
            for idx in sorted(chosen):
                out.append({
                    "pixels": [float(p) for p in x_test[idx]],
                    "true_label": int(y_test[idx]),
                    "predicted_label": int(y_pred_classes[idx]),
                })
            return out

        correct_examples = _sample_examples(correct_indices, 5)
        misclassified_examples = _sample_examples(misclass_indices, 5)

        keras.backend.clear_session()

        q.put({
            "type": "final",
            "train_accuracy_history": [float(x) for x in history.history["accuracy"]],
            "val_accuracy_history": [float(x) for x in history.history["val_accuracy"]],
            "train_loss_history": [float(x) for x in history.history["loss"]],
            "val_loss_history": [float(x) for x in history.history["val_loss"]],
            "test_accuracy": float(test_accuracy),
            "test_loss": float(test_loss),
            "confusion_matrix": cm.tolist(),
            "training_time": float(training_time),
            "model_summary": model_summary,
            "total_params": total_params,
            "correct_examples": correct_examples,
            "misclassified_examples": misclassified_examples,
        })
    except Exception as exc:  # noqa: BLE001
        q.put({"type": "error", "detail": str(exc)})
    finally:
        q.put(None)  # sentinel: stream is done


async def _stream_training(request: TrainRequest):
    """Async generator yielding NDJSON lines as training progresses."""
    loop = asyncio.get_event_loop()
    q: "queue.Queue" = queue.Queue()

    thread = threading.Thread(target=_run_training, args=(request, q), daemon=True)
    thread.start()

    while True:
        # Offload the blocking q.get() to a worker thread so the event
        # loop stays responsive to health checks during long trainings.
        item = await loop.run_in_executor(None, q.get)
        if item is None:
            break
        yield (json.dumps(item) + "\n").encode("utf-8")


@app.post("/train")
async def train_model(request: TrainRequest):
    """Stream per-epoch training progress as NDJSON.

    Response body is a sequence of newline-delimited JSON objects:
      - {"type": "epoch", "epoch": N, "train_loss": ..., "val_loss": ..., ...}
      - {"type": "final", "confusion_matrix": ..., "test_accuracy": ..., ...}
      - {"type": "error", "detail": "..."}  (on failure)
    """

    # Validate activation function
    if request.activation not in ["relu", "sigmoid", "tanh"]:
        raise HTTPException(
            status_code=400,
            detail="Activation must be one of: relu, sigmoid, tanh",
        )

    # Validate batch size
    if request.batch_size not in [16, 32, 64, 128]:
        raise HTTPException(
            status_code=400,
            detail="Batch size must be one of: 16, 32, 64, 128",
        )

    # Headers hint to proxies/CDNs not to buffer the stream (important on
    # Cloud Run behind the Google Front End).
    headers = {
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no",
    }
    return StreamingResponse(
        _stream_training(request),
        media_type="application/x-ndjson",
        headers=headers,
    )


def _compare_models_sync() -> CompareResponse:
    """Compare XGBoost vs Simple NN vs Deep NN on the MNIST subsample."""
    _tf, keras, _layers, _models, _optimizers, _cb = _load_tf()
    XGBClassifier = _load_xgboost()

    data = load_mnist_data()
    x_train = data["x_train"]
    y_train = data["y_train"]
    x_test = data["x_test"]
    y_test = data["y_test"]
    n_features = data["n_features"]

    comparisons = []

    # 1. XGBoost (multi-class softprob)
    print("Training XGBoost...")
    start_time = time.time()
    xgb_model = XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        n_jobs=2,
        objective="multi:softprob",
        num_class=10,
        random_state=42,
    )
    xgb_model.fit(x_train, y_train)
    xgb_time = time.time() - start_time
    xgb_accuracy = xgb_model.score(x_test, y_test)
    comparisons.append(ModelComparison(
        model_name="XGBoost",
        test_accuracy=float(xgb_accuracy),
        training_time=float(xgb_time),
        description="Gradient boosted trees (100 estimators, max_depth=4)",
    ))

    # 2. Simple NN (single hidden layer)
    print("Training Simple NN...")
    keras.backend.clear_session()
    simple_nn = build_model(
        n_features=n_features,
        hidden_layers=[32],
        activation="relu",
        learning_rate=0.001,
    )
    start_time = time.time()
    simple_nn.fit(x_train, y_train, epochs=50, batch_size=32, verbose=0)
    simple_time = time.time() - start_time
    _, simple_accuracy = simple_nn.evaluate(x_test, y_test, verbose=0)
    comparisons.append(ModelComparison(
        model_name="Simple NN",
        test_accuracy=float(simple_accuracy),
        training_time=float(simple_time),
        description="Single hidden layer (32 units)",
    ))

    # 3. Deep NN (matches notebook: 32, 64, 32)
    print("Training Deep NN...")
    keras.backend.clear_session()
    deep_nn = build_model(
        n_features=n_features,
        hidden_layers=[32, 64, 32],
        activation="relu",
        learning_rate=0.001,
    )
    start_time = time.time()
    deep_nn.fit(x_train, y_train, epochs=50, batch_size=32, verbose=0)
    deep_time = time.time() - start_time
    _, deep_accuracy = deep_nn.evaluate(x_test, y_test, verbose=0)
    comparisons.append(ModelComparison(
        model_name="Deep NN",
        test_accuracy=float(deep_accuracy),
        training_time=float(deep_time),
        description="Three hidden layers (32, 64, 32)",
    ))

    # Clear memory
    keras.backend.clear_session()

    # Find best model
    best_model = max(comparisons, key=lambda x: x.test_accuracy)

    return CompareResponse(
        comparisons=comparisons,
        best_model=best_model.model_name,
    )


@app.get("/compare", response_model=CompareResponse)
async def compare_models():
    """Compare XGBoost vs simple NN vs deep NN on the MNIST subsample."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _compare_models_sync)


BASE_DIR = os.path.dirname(os.path.abspath(__file__))


@app.get("/")
async def serve_frontend():
    """Serve the frontend application."""
    return FileResponse(os.path.join(BASE_DIR, "index.html"))


# Serve static assets — mounted AFTER explicit routes so they take priority
app.mount("/static", StaticFiles(directory=BASE_DIR), name="static")


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
