"""
Week 5: Neural Networks - Fashion-MNIST Image Classification
FastAPI Backend for training and evaluating neural networks
"""

import base64
import io
import time
from typing import List, Optional

import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from PIL import Image
from sklearn.metrics import confusion_matrix
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers, callbacks
from xgboost import XGBClassifier

# Suppress TensorFlow warnings
tf.get_logger().setLevel('ERROR')

router = APIRouter()

# CORS middleware
# CORS handled by main app

# Class names for Fashion-MNIST
CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

# Global cache for dataset
_dataset_cache = None


def load_fashion_mnist():
    """Load and preprocess Fashion-MNIST dataset with caching."""
    global _dataset_cache

    if _dataset_cache is not None:
        return _dataset_cache

    # Load dataset (auto-downloads if not present)
    (x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()

    # Normalize pixel values to [0, 1]
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    # Reshape for neural network (flatten) - will be reshaped as needed
    x_train_flat = x_train.reshape(-1, 784)
    x_test_flat = x_test.reshape(-1, 784)

    _dataset_cache = {
        "x_train": x_train,
        "y_train": y_train,
        "x_test": x_test,
        "y_test": y_test,
        "x_train_flat": x_train_flat,
        "x_test_flat": x_test_flat
    }

    return _dataset_cache


# Pydantic models for request/response
class TrainRequest(BaseModel):
    hidden_layers: List[int] = Field(default=[128, 64], description="List of hidden layer sizes")
    activation: str = Field(default="relu", description="Activation function: relu, sigmoid, tanh")
    dropout_rate: float = Field(default=0.2, ge=0.0, le=0.5, description="Dropout rate")
    use_batch_norm: bool = Field(default=False, description="Whether to use batch normalization")
    learning_rate: float = Field(default=0.001, ge=0.0001, le=0.1, description="Learning rate")
    batch_size: int = Field(default=64, description="Batch size: 32, 64, 128, 256")
    epochs: int = Field(default=10, ge=5, le=50, description="Number of epochs")


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


class PredictRequest(BaseModel):
    image_base64: str = Field(description="Base64 encoded image")


class PredictResponse(BaseModel):
    predicted_class: int
    predicted_label: str
    confidence: float
    all_probabilities: List[float]
    class_names: List[str]


class SampleImage(BaseModel):
    image_base64: str
    label: int
    label_name: str


class SampleImagesResponse(BaseModel):
    images: List[SampleImage]
    class_names: List[str]


class ModelComparison(BaseModel):
    model_name: str
    test_accuracy: float
    training_time: float
    description: str


class CompareResponse(BaseModel):
    comparisons: List[ModelComparison]
    best_model: str


def build_model(
    hidden_layers: List[int],
    activation: str,
    dropout_rate: float,
    use_batch_norm: bool,
    learning_rate: float
) -> keras.Model:
    """Build a neural network model with specified architecture."""

    model = models.Sequential()
    model.add(layers.Input(shape=(784,)))

    for i, units in enumerate(hidden_layers):
        model.add(layers.Dense(units))

        if use_batch_norm:
            model.add(layers.BatchNormalization())

        model.add(layers.Activation(activation))

        if dropout_rate > 0:
            model.add(layers.Dropout(dropout_rate))

    # Output layer
    model.add(layers.Dense(10, activation="softmax"))

    # Compile model
    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


def get_model_summary(model: keras.Model) -> str:
    """Get model summary as string."""
    string_list = []
    model.summary(print_fn=lambda x: string_list.append(x))
    return "\n".join(string_list)


def image_to_base64(img_array: np.ndarray) -> str:
    """Convert numpy array to base64 encoded PNG."""
    # Scale to 0-255 if normalized
    if img_array.max() <= 1.0:
        img_array = (img_array * 255).astype(np.uint8)

    img = Image.fromarray(img_array)
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def base64_to_image(base64_string: str) -> np.ndarray:
    """Convert base64 encoded image to numpy array."""
    # Remove data URL prefix if present
    if "," in base64_string:
        base64_string = base64_string.split(",")[1]

    img_bytes = base64.b64decode(base64_string)
    img = Image.open(io.BytesIO(img_bytes))

    # Convert to grayscale if needed
    if img.mode != "L":
        img = img.convert("L")

    # Resize to 28x28
    img = img.resize((28, 28))

    # Convert to numpy and normalize
    img_array = np.array(img).astype("float32") / 255.0

    return img_array


@router.get("/")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "Week 5: Neural Networks - Fashion-MNIST Classification",
        "class_names": CLASS_NAMES
    }


@router.get("/sample_images", response_model=SampleImagesResponse)
async def get_sample_images():
    """Return base64 encoded sample images with labels."""
    data = load_fashion_mnist()

    # Get 2 samples from each class (20 total)
    samples = []
    for class_idx in range(10):
        # Find indices of this class
        class_indices = np.where(data["y_test"] == class_idx)[0]
        # Take first 2 samples
        selected_indices = class_indices[:2]

        for idx in selected_indices:
            img = data["x_test"][idx]
            samples.append(SampleImage(
                image_base64=image_to_base64(img),
                label=int(class_idx),
                label_name=CLASS_NAMES[class_idx]
            ))

    return SampleImagesResponse(
        images=samples,
        class_names=CLASS_NAMES
    )


@router.post("/train", response_model=TrainResponse)
async def train_model(request: TrainRequest):
    """Train a custom neural network with specified hyperparameters."""

    # Validate activation function
    if request.activation not in ["relu", "sigmoid", "tanh"]:
        raise HTTPException(
            status_code=400,
            detail="Activation must be one of: relu, sigmoid, tanh"
        )

    # Validate batch size
    if request.batch_size not in [32, 64, 128, 256]:
        raise HTTPException(
            status_code=400,
            detail="Batch size must be one of: 32, 64, 128, 256"
        )

    # Load data
    data = load_fashion_mnist()
    x_train = data["x_train_flat"]
    y_train = data["y_train"]
    x_test = data["x_test_flat"]
    y_test = data["y_test"]

    # Split training data for validation
    val_split = 0.1
    val_size = int(len(x_train) * val_split)
    x_val = x_train[-val_size:]
    y_val = y_train[-val_size:]
    x_train = x_train[:-val_size]
    y_train = y_train[:-val_size]

    # Build model
    model = build_model(
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
        patience=5,
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
    y_pred = model.predict(x_test, verbose=0)
    y_pred_classes = np.argmax(y_pred, axis=1)
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


@router.get("/compare", response_model=CompareResponse)
async def compare_models():
    """Compare XGBoost vs simple NN vs deep NN vs regularized NN."""

    data = load_fashion_mnist()
    x_train = data["x_train_flat"]
    y_train = data["y_train"]
    x_test = data["x_test_flat"]
    y_test = data["y_test"]

    comparisons = []

    # 1. XGBoost
    print("Training XGBoost...")
    start_time = time.time()
    xgb_model = XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        n_jobs=-1,
        random_state=42
    )
    # Use subset for faster training
    train_subset = 10000
    xgb_model.fit(x_train[:train_subset], y_train[:train_subset])
    xgb_time = time.time() - start_time
    xgb_accuracy = xgb_model.score(x_test, y_test)
    comparisons.append(ModelComparison(
        model_name="XGBoost",
        test_accuracy=float(xgb_accuracy),
        training_time=float(xgb_time),
        description="Gradient boosted trees (100 estimators, max_depth=6)"
    ))

    # 2. Simple NN (single hidden layer)
    print("Training Simple NN...")
    keras.backend.clear_session()
    simple_nn = build_model(
        hidden_layers=[64],
        activation="relu",
        dropout_rate=0.0,
        use_batch_norm=False,
        learning_rate=0.001
    )
    start_time = time.time()
    simple_nn.fit(x_train, y_train, epochs=10, batch_size=128, verbose=0)
    simple_time = time.time() - start_time
    _, simple_accuracy = simple_nn.evaluate(x_test, y_test, verbose=0)
    comparisons.append(ModelComparison(
        model_name="Simple NN",
        test_accuracy=float(simple_accuracy),
        training_time=float(simple_time),
        description="Single hidden layer (64 units), no regularization"
    ))

    # 3. Deep NN (multiple hidden layers, no regularization)
    print("Training Deep NN...")
    keras.backend.clear_session()
    deep_nn = build_model(
        hidden_layers=[256, 128, 64],
        activation="relu",
        dropout_rate=0.0,
        use_batch_norm=False,
        learning_rate=0.001
    )
    start_time = time.time()
    deep_nn.fit(x_train, y_train, epochs=10, batch_size=128, verbose=0)
    deep_time = time.time() - start_time
    _, deep_accuracy = deep_nn.evaluate(x_test, y_test, verbose=0)
    comparisons.append(ModelComparison(
        model_name="Deep NN",
        test_accuracy=float(deep_accuracy),
        training_time=float(deep_time),
        description="Three hidden layers (256, 128, 64), no regularization"
    ))

    # 4. Regularized NN (with dropout and batch norm)
    print("Training Regularized NN...")
    keras.backend.clear_session()
    reg_nn = build_model(
        hidden_layers=[256, 128, 64],
        activation="relu",
        dropout_rate=0.3,
        use_batch_norm=True,
        learning_rate=0.001
    )
    start_time = time.time()
    reg_nn.fit(x_train, y_train, epochs=15, batch_size=128, verbose=0)
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


@router.post("/predict", response_model=PredictResponse)
async def predict_image(request: PredictRequest):
    """Predict class for a given base64 encoded image."""

    try:
        # Decode and preprocess image
        img_array = base64_to_image(request.image_base64)

        # Flatten for model input
        img_flat = img_array.reshape(1, 784)

        # Build a simple model for prediction
        # In production, you'd load a pre-trained model
        data = load_fashion_mnist()
        x_train = data["x_train_flat"]
        y_train = data["y_train"]

        # Train a quick model
        keras.backend.clear_session()
        model = build_model(
            hidden_layers=[128, 64],
            activation="relu",
            dropout_rate=0.2,
            use_batch_norm=True,
            learning_rate=0.001
        )
        model.fit(x_train, y_train, epochs=5, batch_size=128, verbose=0)

        # Make prediction
        predictions = model.predict(img_flat, verbose=0)[0]
        predicted_class = int(np.argmax(predictions))
        confidence = float(predictions[predicted_class])

        # Clear memory
        keras.backend.clear_session()

        return PredictResponse(
            predicted_class=predicted_class,
            predicted_label=CLASS_NAMES[predicted_class],
            confidence=confidence,
            all_probabilities=[float(p) for p in predictions],
            class_names=CLASS_NAMES
        )

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing image: {str(e)}")
