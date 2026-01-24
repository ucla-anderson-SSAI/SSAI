"""
Assignment 7: Transformers and Attention for Text Classification
FastAPI Backend - IMDb Sentiment Classification with Transformer Models
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import json
import os

router = APIRouter()


# Constants
VOCAB_SIZE = 10000
MAX_LEN = 200
EMBED_DIM = 32

# Cache for data
_data_cache = {}

def get_imdb_data():
    """Load and cache IMDb dataset (or generate synthetic data if download fails)"""
    if 'data' not in _data_cache:
        print("Loading IMDb dataset...")
        try:
            (x_train, y_train), (x_test, y_test) = keras.datasets.imdb.load_data(num_words=VOCAB_SIZE)
            x_train = keras.preprocessing.sequence.pad_sequences(x_train, maxlen=MAX_LEN)
            x_test = keras.preprocessing.sequence.pad_sequences(x_test, maxlen=MAX_LEN)

            # Use subset for faster training
            _data_cache['data'] = (
                x_train[:5000], y_train[:5000],
                x_test[:1000], y_test[:1000]
            )
        except Exception as e:
            print(f"Could not download IMDb dataset: {e}")
            print("Generating synthetic dataset for demonstration...")
            # Generate synthetic data for demonstration
            np.random.seed(42)
            n_train, n_test = 5000, 1000

            # Create sequences with patterns that can be learned
            x_train = np.random.randint(1, VOCAB_SIZE, size=(n_train, MAX_LEN))
            x_test = np.random.randint(1, VOCAB_SIZE, size=(n_test, MAX_LEN))

            # Create labels with some learnable pattern based on token presence
            # Positive sentiment: more high-value tokens in first half
            y_train = (np.mean(x_train[:, :100], axis=1) > VOCAB_SIZE // 2).astype(np.int32)
            y_test = (np.mean(x_test[:, :100], axis=1) > VOCAB_SIZE // 2).astype(np.int32)

            _data_cache['data'] = (x_train, y_train, x_test, y_test)
            _data_cache['synthetic'] = True
    return _data_cache['data']


class TokenAndPositionEmbedding(layers.Layer):
    """Token and Position Embedding Layer"""
    def __init__(self, maxlen, vocab_size, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.maxlen = maxlen
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions

    def get_config(self):
        config = super().get_config()
        config.update({
            "maxlen": self.maxlen,
            "vocab_size": self.vocab_size,
            "embed_dim": self.embed_dim,
        })
        return config


class TransformerBlock(layers.Layer):
    """Transformer Block with Multi-Head Attention"""
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.rate = rate

        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential([
            layers.Dense(ff_dim, activation="relu"),
            layers.Dense(embed_dim),
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training=False, return_attention=False):
        if return_attention:
            attn_output, attn_weights = self.att(
                inputs, inputs, return_attention_scores=True
            )
        else:
            attn_output = self.att(inputs, inputs)
            attn_weights = None

        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        output = self.layernorm2(out1 + ffn_output)

        if return_attention:
            return output, attn_weights
        return output

    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "ff_dim": self.ff_dim,
            "rate": self.rate,
        })
        return config


def build_transformer_model(vocab_size, maxlen, embed_dim, num_heads, ff_dim, num_blocks):
    """Build Transformer model for classification"""
    inputs = layers.Input(shape=(maxlen,))
    x = TokenAndPositionEmbedding(maxlen, vocab_size, embed_dim)(inputs)

    for _ in range(num_blocks):
        x = TransformerBlock(embed_dim, num_heads, ff_dim)(x)

    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.1)(x)
    x = layers.Dense(20, activation="relu")(x)
    x = layers.Dropout(0.1)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model


def build_lstm_model(vocab_size, maxlen, embed_dim):
    """Build LSTM model for comparison"""
    inputs = layers.Input(shape=(maxlen,))
    x = layers.Embedding(vocab_size, embed_dim)(inputs)
    x = layers.LSTM(64, return_sequences=True)(x)
    x = layers.LSTM(32)(x)
    x = layers.Dropout(0.1)(x)
    x = layers.Dense(20, activation="relu")(x)
    x = layers.Dropout(0.1)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model


# Pydantic models
class HeadsRequest(BaseModel):
    num_heads_list: List[int] = [1, 2, 4, 8]
    epochs: int = 2

class DepthRequest(BaseModel):
    num_blocks_list: List[int] = [1, 2, 3, 4]
    epochs: int = 2

class CustomRequest(BaseModel):
    embed_dim: int = 32
    num_heads: int = 2
    num_blocks: int = 2
    ff_dim: int = 32
    epochs: int = 3


@router.post("/train_heads")
async def train_heads(request: HeadsRequest):
    """Train models with different numbers of attention heads"""
    x_train, y_train, x_test, y_test = get_imdb_data()

    results = []
    for num_heads in request.num_heads_list:
        print(f"Training with {num_heads} attention heads...")
        model = build_transformer_model(
            VOCAB_SIZE, MAX_LEN, EMBED_DIM, num_heads, ff_dim=32, num_blocks=2
        )

        model.fit(
            x_train, y_train,
            batch_size=64,
            epochs=request.epochs,
            validation_split=0.1,
            verbose=0
        )

        _, accuracy = model.evaluate(x_test, y_test, verbose=0)
        param_count = model.count_params()

        results.append({
            "num_heads": num_heads,
            "accuracy": float(accuracy),
            "param_count": param_count
        })

        keras.backend.clear_session()

    return {"results": results}


@router.post("/train_depth")
async def train_depth(request: DepthRequest):
    """Train models with different transformer depths"""
    x_train, y_train, x_test, y_test = get_imdb_data()

    results = []
    for num_blocks in request.num_blocks_list:
        print(f"Training with {num_blocks} transformer blocks...")
        model = build_transformer_model(
            VOCAB_SIZE, MAX_LEN, EMBED_DIM, num_heads=2, ff_dim=32, num_blocks=num_blocks
        )

        model.fit(
            x_train, y_train,
            batch_size=64,
            epochs=request.epochs,
            validation_split=0.1,
            verbose=0
        )

        _, accuracy = model.evaluate(x_test, y_test, verbose=0)
        param_count = model.count_params()

        results.append({
            "num_blocks": num_blocks,
            "accuracy": float(accuracy),
            "param_count": param_count
        })

        keras.backend.clear_session()

    return {"results": results}


@router.post("/train_custom")
async def train_custom(request: CustomRequest):
    """Train custom transformer model"""
    x_train, y_train, x_test, y_test = get_imdb_data()

    model = build_transformer_model(
        VOCAB_SIZE, MAX_LEN,
        request.embed_dim,
        request.num_heads,
        request.ff_dim,
        request.num_blocks
    )

    history = model.fit(
        x_train, y_train,
        batch_size=64,
        epochs=request.epochs,
        validation_split=0.1,
        verbose=1
    )

    _, test_accuracy = model.evaluate(x_test, y_test, verbose=0)

    result = {
        "history": {
            "loss": [float(l) for l in history.history['loss']],
            "accuracy": [float(a) for a in history.history['accuracy']],
            "val_loss": [float(l) for l in history.history['val_loss']],
            "val_accuracy": [float(a) for a in history.history['val_accuracy']],
        },
        "test_accuracy": float(test_accuracy),
        "param_count": model.count_params()
    }

    keras.backend.clear_session()
    return result


@router.post("/compare_lstm")
async def compare_lstm():
    """Compare LSTM vs Transformer performance"""
    x_train, y_train, x_test, y_test = get_imdb_data()
    epochs = 2

    # Train Transformer
    print("Training Transformer model...")
    transformer = build_transformer_model(
        VOCAB_SIZE, MAX_LEN, EMBED_DIM, num_heads=2, ff_dim=32, num_blocks=2
    )
    transformer.fit(x_train, y_train, batch_size=64, epochs=epochs, validation_split=0.1, verbose=0)
    _, transformer_acc = transformer.evaluate(x_test, y_test, verbose=0)
    transformer_params = transformer.count_params()

    keras.backend.clear_session()

    # Train LSTM
    print("Training LSTM model...")
    lstm = build_lstm_model(VOCAB_SIZE, MAX_LEN, EMBED_DIM)
    lstm.fit(x_train, y_train, batch_size=64, epochs=epochs, validation_split=0.1, verbose=0)
    _, lstm_acc = lstm.evaluate(x_test, y_test, verbose=0)
    lstm_params = lstm.count_params()

    keras.backend.clear_session()

    return {
        "transformer": {
            "accuracy": float(transformer_acc),
            "param_count": transformer_params,
            "training_time": "Fast (parallelizable)"
        },
        "lstm": {
            "accuracy": float(lstm_acc),
            "param_count": lstm_params,
            "training_time": "Slower (sequential)"
        }
    }


@router.get("/attention_demo")
async def attention_demo():
    """Generate sample attention weights for visualization"""
    # Simulated attention weights for a 4-token sequence
    # In practice, these would come from actual model inference

    tokens = ["The", "movie", "was", "great"]

    # Simulate realistic attention patterns
    # "great" attends to "movie" and "was"
    # "was" attends to "movie"
    attention_weights = np.array([
        [0.7, 0.1, 0.1, 0.1],  # "The" mostly self-attends
        [0.2, 0.6, 0.1, 0.1],  # "movie" self-attends
        [0.1, 0.3, 0.5, 0.1],  # "was" attends to movie and self
        [0.05, 0.35, 0.25, 0.35]  # "great" attends to movie, was, and self
    ])

    return {
        "tokens": tokens,
        "attention_weights": attention_weights.tolist(),
        "explanation": "Each row shows how much attention a token pays to other tokens (columns). Higher values indicate stronger attention. Notice how 'great' attends strongly to 'movie', capturing the sentiment relationship."
    }


@router.get("/temperature_demo")
async def temperature_demo():
    """Demonstrate temperature effects on text generation"""
    prompt = "The movie was"

    # Simulated outputs at different temperatures
    results = {
        "prompt": prompt,
        "temperature_effects": [
            {
                "temperature": 0.1,
                "output": "The movie was excellent and I highly recommend it.",
                "description": "Very deterministic - picks highest probability tokens",
                "characteristics": ["Repetitive", "Predictable", "Safe choices"]
            },
            {
                "temperature": 0.5,
                "output": "The movie was surprisingly good with great performances.",
                "description": "Balanced - moderate diversity in outputs",
                "characteristics": ["Coherent", "Some variety", "Natural flow"]
            },
            {
                "temperature": 1.0,
                "output": "The movie was a fascinating exploration of human nature.",
                "description": "Standard sampling - true to learned distribution",
                "characteristics": ["Diverse", "Creative", "May have minor errors"]
            },
            {
                "temperature": 2.0,
                "output": "The movie was purple elephants dancing mysteriously.",
                "description": "High randomness - flattened probability distribution",
                "characteristics": ["Incoherent", "Random", "Unexpected combinations"]
            }
        ],
        "formula": "P(token) = exp(logit/T) / sum(exp(logits/T))",
        "explanation": "Temperature (T) controls randomness in text generation. Lower T makes the model more confident and deterministic, higher T increases diversity but may reduce coherence."
    }

    return results


# Serve static files
@router.get("/")
async def read_root():
    return FileResponse("index.html")


