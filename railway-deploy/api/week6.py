"""
Week 6: CNNs - CIFAR-10 Image Classification Backend
FastAPI backend for training and comparing CNN architectures on CIFAR-10.
"""

import base64
import io
import time
from enum import Enum
from typing import List, Optional

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from sklearn.metrics import confusion_matrix

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar10

# Initialize FastAPI app
router = APIRouter()

# Add CORS middleware

# Enums for model types
class ModelType(str, Enum):
    mlp = "mlp"
    simple_cnn = "simple_cnn"
    advanced_cnn = "advanced_cnn"

# Pydantic models for request/response
class TrainRequest(BaseModel):
    model_type: ModelType = ModelType.simple_cnn
    num_conv_blocks: int = Field(default=2, ge=1, le=3)
    filters_per_block: int = Field(default=32, ge=16, le=128)
    kernel_size: int = Field(default=3, ge=3, le=5)
    use_batch_norm: bool = True
    dropout_rate: float = Field(default=0.25, ge=0.0, le=0.5)
    learning_rate: float = Field(default=0.001, ge=0.0001, le=0.01)
    batch_size: int = Field(default=64, ge=32, le=128)
    epochs: int = Field(default=10, ge=5, le=30)

class TrainResponse(BaseModel):
    model_type: str
    accuracy_history: List[float]
    val_accuracy_history: List[float]
    loss_history: List[float]
    val_loss_history: List[float]
    test_accuracy: float
    test_loss: float
    confusion_matrix: List[List[int]]
    num_parameters: int
    training_time: float
    class_names: List[str] = CLASS_NAMES

class SampleImagesResponse(BaseModel):
    images: List[str]  # Base64 encoded images
    labels: List[str]
    label_indices: List[int]

class CompareResponse(BaseModel):
    results: List[TrainResponse]

class VisualizeFiltersRequest(BaseModel):
    model_type: ModelType = ModelType.simple_cnn
    num_conv_blocks: int = Field(default=2, ge=1, le=3)
    filters_per_block: int = Field(default=32, ge=16, le=128)
    kernel_size: int = Field(default=3, ge=3, le=5)
    use_batch_norm: bool = True
    layer_index: int = Field(default=0, ge=0, description="Index of conv layer to visualize")

class VisualizeFiltersResponse(BaseModel):
    filter_images: List[str]  # Base64 encoded filter visualizations
    layer_name: str
    num_filters: int

def build_mlp_model(input_shape=(32, 32, 3), num_classes=10,
                    dropout_rate=0.25, learning_rate=0.001):
    """Build a Multi-Layer Perceptron model."""
    model = models.Sequential([
        layers.Flatten(input_shape=input_shape),
        layers.Dense(512, activation='relu'),
        layers.Dropout(dropout_rate),
        layers.Dense(256, activation='relu'),
        layers.Dropout(dropout_rate),
        layers.Dense(128, activation='relu'),
        layers.Dropout(dropout_rate),
        layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def build_simple_cnn_model(input_shape=(32, 32, 3), num_classes=10,
                           num_conv_blocks=2, filters_per_block=32,
                           kernel_size=3, use_batch_norm=True,
                           dropout_rate=0.25, learning_rate=0.001):
    """Build a simple CNN model."""
    model = models.Sequential()
    model.add(layers.Input(shape=input_shape))

    filters = filters_per_block
    for i in range(num_conv_blocks):
        model.add(layers.Conv2D(filters, (kernel_size, kernel_size),
                                padding='same', activation='relu'))
        if use_batch_norm:
            model.add(layers.BatchNormalization())
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Dropout(dropout_rate))
        filters = min(filters * 2, 256)  # Double filters, cap at 256

    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(dropout_rate))
    model.add(layers.Dense(num_classes, activation='softmax'))

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def build_advanced_cnn_model(input_shape=(32, 32, 3), num_classes=10,
                             num_conv_blocks=3, filters_per_block=32,
                             kernel_size=3, use_batch_norm=True,
                             dropout_rate=0.25, learning_rate=0.001):
    """Build an advanced CNN model with skip connections."""
    inputs = layers.Input(shape=input_shape)
    x = inputs

    filters = filters_per_block
    for i in range(num_conv_blocks):
        # First conv layer in block
        conv1 = layers.Conv2D(filters, (kernel_size, kernel_size),
                              padding='same', activation='relu')(x)
        if use_batch_norm:
            conv1 = layers.BatchNormalization()(conv1)

        # Second conv layer in block
        conv2 = layers.Conv2D(filters, (kernel_size, kernel_size),
                              padding='same', activation='relu')(conv1)
        if use_batch_norm:
            conv2 = layers.BatchNormalization()(conv2)

        # Skip connection (adjust channels if needed)
        if x.shape[-1] != filters:
            x = layers.Conv2D(filters, (1, 1), padding='same')(x)

        x = layers.Add()([x, conv2])
        x = layers.Activation('relu')(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(dropout_rate)(x)

        filters = min(filters * 2, 256)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def train_model(model, x_train, y_train, x_test, y_test,
                epochs=10, batch_size=64):
    """Train the model and return metrics."""
    start_time = time.time()

    # Use a subset for faster training in demo
    train_samples = min(len(x_train), 25000)
    test_samples = min(len(x_test), 5000)

    history = model.fit(
        x_train[:train_samples], y_train[:train_samples],
        validation_split=0.2,
        epochs=epochs,
        batch_size=batch_size,
        verbose=1
    )

    training_time = time.time() - start_time

    # Evaluate on test set
    test_loss, test_accuracy = model.evaluate(
        x_test[:test_samples], y_test[:test_samples], verbose=0
    )

    # Get predictions for confusion matrix
    predictions = model.predict(x_test[:test_samples], verbose=0)
    pred_labels = np.argmax(predictions, axis=1)
    conf_matrix = confusion_matrix(y_test[:test_samples], pred_labels)

    return {
        'accuracy_history': [float(x) for x in history.history['accuracy']],
        'val_accuracy_history': [float(x) for x in history.history['val_accuracy']],
        'loss_history': [float(x) for x in history.history['loss']],
        'val_loss_history': [float(x) for x in history.history['val_loss']],
        'test_accuracy': float(test_accuracy),
        'test_loss': float(test_loss),
        'confusion_matrix': conf_matrix.tolist(),
        'num_parameters': int(model.count_params()),
        'training_time': float(training_time)
    }

def image_to_base64(img_array):
    """Convert numpy image array to base64 string."""
    # Ensure values are in [0, 255] range
    if img_array.max() <= 1.0:
        img_array = (img_array * 255).astype(np.uint8)
    else:
        img_array = img_array.astype(np.uint8)

    fig, ax = plt.subplots(figsize=(2, 2))
    ax.imshow(img_array)
    ax.axis('off')

    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)
    buf.seek(0)

    return base64.b64encode(buf.read()).decode('utf-8')

@router.get("/", tags=["Health"])
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "message": "Week 6: CNNs - CIFAR-10 Image Classification API",
        "tensorflow_version": tf.__version__,
        "gpu_available": len(tf.config.list_physical_devices('GPU')) > 0
    }

@router.get("/sample_images", response_model=SampleImagesResponse, tags=["Data"])
async def get_sample_images(num_samples: int = 10):
    """Get sample images from CIFAR-10 dataset."""
    if num_samples < 1 or num_samples > 100:
        raise HTTPException(status_code=400, detail="num_samples must be between 1 and 100")

    x_train, y_train, _, _ = load_cifar10_data()

    # Get random indices
    indices = np.random.choice(len(x_train), num_samples, replace=False)

    images = []
    labels = []
    label_indices = []

    for idx in indices:
        img_base64 = image_to_base64(x_train[idx])
        images.append(img_base64)
        labels.append(CLASS_NAMES[y_train[idx]])
        label_indices.append(int(y_train[idx]))

    return SampleImagesResponse(
        images=images,
        labels=labels,
        label_indices=label_indices
    )

@router.post("/train", response_model=TrainResponse, tags=["Training"])
async def train_custom_model(request: TrainRequest):
    """Train a custom CNN architecture on CIFAR-10."""
    x_train, y_train, x_test, y_test = load_cifar10_data()

    # Build model based on type
    if request.model_type == ModelType.mlp:
        model = build_mlp_model(
            dropout_rate=request.dropout_rate,
            learning_rate=request.learning_rate
        )
    elif request.model_type == ModelType.simple_cnn:
        model = build_simple_cnn_model(
            num_conv_blocks=request.num_conv_blocks,
            filters_per_block=request.filters_per_block,
            kernel_size=request.kernel_size,
            use_batch_norm=request.use_batch_norm,
            dropout_rate=request.dropout_rate,
            learning_rate=request.learning_rate
        )
    else:  # advanced_cnn
        model = build_advanced_cnn_model(
            num_conv_blocks=request.num_conv_blocks,
            filters_per_block=request.filters_per_block,
            kernel_size=request.kernel_size,
            use_batch_norm=request.use_batch_norm,
            dropout_rate=request.dropout_rate,
            learning_rate=request.learning_rate
        )

    # Train model
    results = train_model(
        model, x_train, y_train, x_test, y_test,
        epochs=request.epochs,
        batch_size=request.batch_size
    )

    return TrainResponse(
        model_type=request.model_type.value,
        **results
    )

@router.get("/compare", response_model=CompareResponse, tags=["Comparison"])
async def compare_models(
    epochs: int = 5,
    batch_size: int = 64,
    learning_rate: float = 0.001
):
    """Compare MLP vs Simple CNN vs Advanced CNN architectures."""
    x_train, y_train, x_test, y_test = load_cifar10_data()

    results = []

    # Train MLP
    print("Training MLP...")
    mlp_model = build_mlp_model(learning_rate=learning_rate)
    mlp_results = train_model(
        mlp_model, x_train, y_train, x_test, y_test,
        epochs=epochs, batch_size=batch_size
    )
    results.append(TrainResponse(model_type="mlp", **mlp_results))

    # Train Simple CNN
    print("Training Simple CNN...")
    simple_cnn = build_simple_cnn_model(learning_rate=learning_rate)
    simple_cnn_results = train_model(
        simple_cnn, x_train, y_train, x_test, y_test,
        epochs=epochs, batch_size=batch_size
    )
    results.append(TrainResponse(model_type="simple_cnn", **simple_cnn_results))

    # Train Advanced CNN
    print("Training Advanced CNN...")
    advanced_cnn = build_advanced_cnn_model(learning_rate=learning_rate)
    advanced_cnn_results = train_model(
        advanced_cnn, x_train, y_train, x_test, y_test,
        epochs=epochs, batch_size=batch_size
    )
    results.append(TrainResponse(model_type="advanced_cnn", **advanced_cnn_results))

    return CompareResponse(results=results)

@router.post("/visualize_filters", response_model=VisualizeFiltersResponse, tags=["Visualization"])
async def visualize_filters(request: VisualizeFiltersRequest):
    """Visualize convolutional layer filters."""
    if request.model_type == ModelType.mlp:
        raise HTTPException(
            status_code=400,
            detail="MLP models don't have convolutional filters to visualize"
        )

    # Build model
    if request.model_type == ModelType.simple_cnn:
        model = build_simple_cnn_model(
            num_conv_blocks=request.num_conv_blocks,
            filters_per_block=request.filters_per_block,
            kernel_size=request.kernel_size,
            use_batch_norm=request.use_batch_norm
        )
    else:
        model = build_advanced_cnn_model(
            num_conv_blocks=request.num_conv_blocks,
            filters_per_block=request.filters_per_block,
            kernel_size=request.kernel_size,
            use_batch_norm=request.use_batch_norm
        )

    # Find convolutional layers
    conv_layers = [layer for layer in model.layers
                   if isinstance(layer, layers.Conv2D)]

    if request.layer_index >= len(conv_layers):
        raise HTTPException(
            status_code=400,
            detail=f"Layer index {request.layer_index} out of range. "
                   f"Model has {len(conv_layers)} conv layers (0-{len(conv_layers)-1})"
        )

    target_layer = conv_layers[request.layer_index]
    filters, biases = target_layer.get_weights()

    # Normalize filters for visualization
    f_min, f_max = filters.min(), filters.max()
    filters = (filters - f_min) / (f_max - f_min + 1e-8)

    # Create filter visualizations
    num_filters = filters.shape[-1]
    filter_images = []

    # Show up to 32 filters
    num_to_show = min(num_filters, 32)

    # Create a grid of filters
    n_cols = 8
    n_rows = (num_to_show + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, n_rows * 1.5))
    axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes.flatten()

    for i in range(num_to_show):
        # For RGB filters, show as color image; for single channel, show grayscale
        if filters.shape[2] == 3:
            ax_img = filters[:, :, :, i]
        else:
            ax_img = filters[:, :, 0, i]

        axes[i].imshow(ax_img, cmap='viridis' if filters.shape[2] != 3 else None)
        axes[i].axis('off')
        axes[i].set_title(f'F{i}', fontsize=8)

    # Hide unused subplots
    for i in range(num_to_show, len(axes)):
        axes[i].axis('off')

    plt.suptitle(f'Filters from {target_layer.name}', fontsize=12)
    plt.tight_layout()

    # Convert to base64
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    grid_image = base64.b64encode(buf.read()).decode('utf-8')

    return VisualizeFiltersResponse(
        filter_images=[grid_image],
        layer_name=target_layer.name,
        num_filters=num_filters
    )

@router.get("/model_summary/{model_type}", tags=["Info"])
async def get_model_summary(
    model_type: ModelType,
    num_conv_blocks: int = 2,
    filters_per_block: int = 32,
    kernel_size: int = 3,
    use_batch_norm: bool = True
):
    """Get a summary of the model architecture."""
    if model_type == ModelType.mlp:
        model = build_mlp_model()
    elif model_type == ModelType.simple_cnn:
        model = build_simple_cnn_model(
            num_conv_blocks=num_conv_blocks,
            filters_per_block=filters_per_block,
            kernel_size=kernel_size,
            use_batch_norm=use_batch_norm
        )
    else:
        model = build_advanced_cnn_model(
            num_conv_blocks=num_conv_blocks,
            filters_per_block=filters_per_block,
            kernel_size=kernel_size,
            use_batch_norm=use_batch_norm
        )

    # Capture model summary
    string_buffer = io.StringIO()
    model.summary(print_fn=lambda x: string_buffer.write(x + '\n'))
    summary_string = string_buffer.getvalue()

    # Get layer info
    layer_info = []
    for layer in model.layers:
        layer_info.append({
            'name': layer.name,
            'type': layer.__class__.__name__,
            'output_shape': str(layer.output_shape) if hasattr(layer, 'output_shape') else 'N/A',
            'num_params': layer.count_params()
        })

    return {
        'model_type': model_type.value,
        'summary': summary_string,
        'total_params': model.count_params(),
        'trainable_params': sum([tf.keras.backend.count_params(w)
                                  for w in model.trainable_weights]),
        'layers': layer_info
    }

