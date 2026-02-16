"""
Week 6: Word Embeddings & Transformers - FastAPI Backend

Part A: Word Embeddings with GloVe
Part B: Text Classification with Transformer Architecture
"""

import os
import time
import zipfile
import requests
import numpy as np
from typing import List, Optional, Dict, Any
from io import BytesIO

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity

import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.datasets import reuters
from keras.preprocessing.sequence import pad_sequences

app = FastAPI(
    title="Week 6: Word Embeddings & Transformers",
    description="Interactive ML backend for word embeddings and transformer-based text classification",
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

# Global storage for embeddings and models
glove_embeddings: Dict[str, np.ndarray] = {}
embedding_dim = 50
GLOVE_PATH = "/tmp/glove.6B.50d.txt"

# ============================================================================
# Pydantic Models
# ============================================================================

class SimilarityRequest(BaseModel):
    word1: str = Field(..., description="First word")
    word2: str = Field(..., description="Second word")

class SimilarityResponse(BaseModel):
    word1: str
    word2: str
    similarity: float
    word1_found: bool
    word2_found: bool

class AnalogyRequest(BaseModel):
    word_a: str = Field(..., description="First word (e.g., 'king')")
    word_b: str = Field(..., description="Second word (e.g., 'man')")
    word_c: str = Field(..., description="Third word (e.g., 'woman')")
    top_n: int = Field(default=5, ge=1, le=20, description="Number of results to return")

class AnalogyResponse(BaseModel):
    analogy: str
    results: List[Dict[str, Any]]
    words_found: Dict[str, bool]

class EmbeddingVizRequest(BaseModel):
    words: Optional[List[str]] = Field(
        default=None,
        description="Words to visualize (uses default set if not provided)"
    )
    n_components: int = Field(default=2, ge=2, le=3, description="PCA dimensions (2 or 3)")

class EmbeddingVizResponse(BaseModel):
    words: List[str]
    coordinates: List[List[float]]
    n_components: int
    explained_variance_ratio: List[float]

class ClassifierConfig(BaseModel):
    embed_dim: int = Field(default=64, description="Embedding dimension")
    num_heads: int = Field(default=2, description="Number of attention heads")
    num_transformer_blocks: int = Field(default=2, ge=1, le=4, description="Number of transformer blocks")
    ff_dim: int = Field(default=64, description="Feed-forward dimension")
    dropout_rate: float = Field(default=0.1, ge=0.0, le=0.5, description="Dropout rate")
    epochs: int = Field(default=10, ge=5, le=30, description="Training epochs")
    max_len: int = Field(default=200, description="Maximum sequence length")
    batch_size: int = Field(default=32, description="Batch size")

class ClassifierResponse(BaseModel):
    config: Dict[str, Any]
    accuracy_history: List[float]
    val_accuracy_history: List[float]
    loss_history: List[float]
    val_loss_history: List[float]
    test_accuracy: float
    test_loss: float
    training_time: float
    num_params: int

class CompareTransformersResponse(BaseModel):
    one_block: Dict[str, Any]
    three_blocks: Dict[str, Any]
    comparison: Dict[str, Any]

# ============================================================================
# GloVe Embeddings Functions
# ============================================================================

def download_glove():
    """Download GloVe embeddings if not available."""
    if os.path.exists(GLOVE_PATH):
        return True

    print("Downloading GloVe embeddings...")
    url = "http://nlp.stanford.edu/data/glove.6B.zip"

    try:
        response = requests.get(url, stream=True, timeout=300)
        response.raise_for_status()

        with zipfile.ZipFile(BytesIO(response.content)) as z:
            # Extract only the 50d file
            with z.open("glove.6B.50d.txt") as f:
                with open(GLOVE_PATH, "wb") as out:
                    out.write(f.read())

        print("GloVe embeddings downloaded successfully!")
        return True
    except Exception as e:
        print(f"Failed to download GloVe: {e}")
        return False

def load_glove_embeddings(max_words: int = 50000):
    """Load GloVe embeddings from file."""
    global glove_embeddings

    if glove_embeddings:
        return True

    # Try to download if not exists
    if not os.path.exists(GLOVE_PATH):
        if not download_glove():
            # Use a pre-computed subset as fallback
            print("Using pre-computed subset of embeddings...")
            create_sample_embeddings()
            return True

    try:
        print("Loading GloVe embeddings...")
        count = 0
        with open(GLOVE_PATH, "r", encoding="utf-8") as f:
            for line in f:
                if count >= max_words:
                    break
                parts = line.strip().split()
                word = parts[0]
                vector = np.array(parts[1:], dtype=np.float32)
                if len(vector) == embedding_dim:
                    glove_embeddings[word] = vector
                    count += 1

        print(f"Loaded {len(glove_embeddings)} word embeddings")
        return True
    except Exception as e:
        print(f"Error loading GloVe: {e}")
        create_sample_embeddings()
        return True

def create_sample_embeddings():
    """Create a sample set of embeddings for demonstration."""
    global glove_embeddings

    # Pre-computed sample embeddings (normalized for demonstration)
    np.random.seed(42)

    # Common words with semantically meaningful random vectors
    sample_words = [
        "king", "queen", "man", "woman", "prince", "princess",
        "boy", "girl", "father", "mother", "son", "daughter",
        "brother", "sister", "uncle", "aunt",
        "dog", "cat", "bird", "fish", "horse", "cow",
        "car", "truck", "bus", "train", "plane", "boat",
        "red", "blue", "green", "yellow", "black", "white",
        "happy", "sad", "angry", "afraid", "love", "hate",
        "big", "small", "tall", "short", "fast", "slow",
        "good", "bad", "new", "old", "young", "beautiful",
        "house", "home", "building", "city", "country", "world",
        "water", "fire", "earth", "air", "sun", "moon",
        "food", "drink", "bread", "meat", "fruit", "vegetable",
        "work", "play", "read", "write", "speak", "listen",
        "think", "feel", "know", "see", "hear", "touch",
        "time", "day", "night", "year", "week", "month",
        "one", "two", "three", "four", "five", "ten",
        "computer", "phone", "internet", "technology", "science", "math",
        "school", "teacher", "student", "book", "class", "learn",
        "doctor", "nurse", "hospital", "medicine", "health", "sick",
        "money", "bank", "rich", "poor", "buy", "sell",
        "france", "paris", "germany", "berlin", "italy", "rome",
        "japan", "tokyo", "china", "beijing", "russia", "moscow"
    ]

    # Create semantically grouped embeddings
    base_vectors = {}

    # Royalty cluster
    base_vectors["royalty"] = np.random.randn(embedding_dim).astype(np.float32)
    base_vectors["male"] = np.random.randn(embedding_dim).astype(np.float32)
    base_vectors["female"] = np.random.randn(embedding_dim).astype(np.float32)

    # Create embeddings with semantic relationships
    for word in sample_words:
        base = np.random.randn(embedding_dim).astype(np.float32) * 0.5

        # Add semantic components
        if word in ["king", "prince", "father", "son", "brother", "uncle", "boy", "man"]:
            base += base_vectors["male"] * 0.5
        if word in ["queen", "princess", "mother", "daughter", "sister", "aunt", "girl", "woman"]:
            base += base_vectors["female"] * 0.5
        if word in ["king", "queen", "prince", "princess"]:
            base += base_vectors["royalty"] * 0.5

        # Normalize
        glove_embeddings[word] = base / (np.linalg.norm(base) + 1e-8)

    print(f"Created {len(glove_embeddings)} sample embeddings")

def get_embedding(word: str) -> Optional[np.ndarray]:
    """Get embedding for a word."""
    return glove_embeddings.get(word.lower())

def compute_cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    return float(cosine_similarity(vec1.reshape(1, -1), vec2.reshape(1, -1))[0, 0])

def find_nearest_words(vector: np.ndarray, exclude: List[str] = None, top_n: int = 5) -> List[tuple]:
    """Find nearest words to a vector."""
    exclude = exclude or []
    exclude_lower = [w.lower() for w in exclude]

    similarities = []
    for word, emb in glove_embeddings.items():
        if word.lower() not in exclude_lower:
            sim = compute_cosine_similarity(vector, emb)
            similarities.append((word, sim))

    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_n]

# ============================================================================
# Transformer Model Components
# ============================================================================

class TransformerBlock(layers.Layer):
    """Transformer block with multi-head attention."""

    def __init__(self, embed_dim: int, num_heads: int, ff_dim: int, dropout_rate: float = 0.1):
        super(TransformerBlock, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout_rate = dropout_rate

        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential([
            layers.Dense(ff_dim, activation="relu"),
            layers.Dense(embed_dim),
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(dropout_rate)
        self.dropout2 = layers.Dropout(dropout_rate)

    def call(self, inputs, training=False):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "ff_dim": self.ff_dim,
            "dropout_rate": self.dropout_rate,
        })
        return config

class TokenAndPositionEmbedding(layers.Layer):
    """Token and position embedding layer."""

    def __init__(self, maxlen: int, vocab_size: int, embed_dim: int):
        super(TokenAndPositionEmbedding, self).__init__()
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

def build_transformer_classifier(
    vocab_size: int,
    maxlen: int,
    num_classes: int,
    embed_dim: int = 64,
    num_heads: int = 2,
    num_transformer_blocks: int = 2,
    ff_dim: int = 64,
    dropout_rate: float = 0.1
) -> keras.Model:
    """Build a transformer-based text classifier."""

    inputs = layers.Input(shape=(maxlen,))
    embedding_layer = TokenAndPositionEmbedding(maxlen, vocab_size, embed_dim)
    x = embedding_layer(inputs)

    for _ in range(num_transformer_blocks):
        x = TransformerBlock(embed_dim, num_heads, ff_dim, dropout_rate)(x)

    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs=inputs, outputs=outputs)
    return model

def load_reuters_data(max_len: int = 200, max_words: int = 10000):
    """Load and preprocess Reuters dataset."""
    (x_train, y_train), (x_test, y_test) = reuters.load_data(num_words=max_words)

    # Pad sequences
    x_train = pad_sequences(x_train, maxlen=max_len)
    x_test = pad_sequences(x_test, maxlen=max_len)

    num_classes = max(y_train.max(), y_test.max()) + 1

    return (x_train, y_train), (x_test, y_test), num_classes, max_words

# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/", tags=["Health"])
async def health_check():
    """Health check endpoint."""
    embeddings_loaded = len(glove_embeddings) > 0
    return {
        "status": "healthy",
        "service": "Week 6: Word Embeddings & Transformers",
        "embeddings_loaded": embeddings_loaded,
        "num_embeddings": len(glove_embeddings),
        "tensorflow_version": tf.__version__
    }

@app.on_event("startup")
async def startup_event():
    """Load embeddings on startup."""
    load_glove_embeddings()

@app.post("/similarity", response_model=SimilarityResponse, tags=["Word Embeddings"])
async def compute_similarity(request: SimilarityRequest):
    """Compute cosine similarity between two words."""
    if not glove_embeddings:
        load_glove_embeddings()

    vec1 = get_embedding(request.word1)
    vec2 = get_embedding(request.word2)

    word1_found = vec1 is not None
    word2_found = vec2 is not None

    if not word1_found or not word2_found:
        similarity = 0.0
        if not word1_found and not word2_found:
            raise HTTPException(status_code=404, detail=f"Words '{request.word1}' and '{request.word2}' not found in vocabulary")
        elif not word1_found:
            raise HTTPException(status_code=404, detail=f"Word '{request.word1}' not found in vocabulary")
        else:
            raise HTTPException(status_code=404, detail=f"Word '{request.word2}' not found in vocabulary")

    similarity = compute_cosine_similarity(vec1, vec2)

    return SimilarityResponse(
        word1=request.word1,
        word2=request.word2,
        similarity=similarity,
        word1_found=word1_found,
        word2_found=word2_found
    )

@app.post("/analogy", response_model=AnalogyResponse, tags=["Word Embeddings"])
async def solve_analogy(request: AnalogyRequest):
    """
    Solve word analogy: word_a - word_b + word_c = ?
    Example: king - man + woman = queen
    """
    if not glove_embeddings:
        load_glove_embeddings()

    vec_a = get_embedding(request.word_a)
    vec_b = get_embedding(request.word_b)
    vec_c = get_embedding(request.word_c)

    words_found = {
        request.word_a: vec_a is not None,
        request.word_b: vec_b is not None,
        request.word_c: vec_c is not None
    }

    missing = [w for w, found in words_found.items() if not found]
    if missing:
        raise HTTPException(
            status_code=404,
            detail=f"Words not found in vocabulary: {', '.join(missing)}"
        )

    # Compute analogy vector: a - b + c
    result_vector = vec_a - vec_b + vec_c

    # Find nearest words (excluding input words)
    nearest = find_nearest_words(
        result_vector,
        exclude=[request.word_a, request.word_b, request.word_c],
        top_n=request.top_n
    )

    results = [{"word": word, "similarity": float(sim)} for word, sim in nearest]

    return AnalogyResponse(
        analogy=f"{request.word_a} - {request.word_b} + {request.word_c} = ?",
        results=results,
        words_found=words_found
    )

@app.get("/embedding_viz", response_model=EmbeddingVizResponse, tags=["Word Embeddings"])
async def get_embedding_visualization(
    n_components: int = 2,
    words: Optional[str] = None
):
    """
    Get 2D/3D PCA projection of word embeddings for visualization.
    Pass words as comma-separated string, or use default set.
    """
    if not glove_embeddings:
        load_glove_embeddings()

    # Default sample words for visualization
    default_words = [
        "king", "queen", "man", "woman", "prince", "princess",
        "dog", "cat", "bird", "fish",
        "happy", "sad", "good", "bad",
        "big", "small", "fast", "slow",
        "car", "truck", "bus", "train",
        "red", "blue", "green", "yellow"
    ]

    if words:
        word_list = [w.strip() for w in words.split(",")]
    else:
        word_list = default_words

    # Filter to words we have embeddings for
    valid_words = []
    embeddings = []
    for word in word_list:
        emb = get_embedding(word)
        if emb is not None:
            valid_words.append(word)
            embeddings.append(emb)

    if len(valid_words) < 3:
        raise HTTPException(
            status_code=400,
            detail=f"Need at least 3 valid words for PCA. Found {len(valid_words)}: {valid_words}"
        )

    # Perform PCA
    embeddings_array = np.array(embeddings)
    n_components = min(n_components, len(valid_words) - 1, 3)

    pca = PCA(n_components=n_components)
    reduced = pca.fit_transform(embeddings_array)

    return EmbeddingVizResponse(
        words=valid_words,
        coordinates=reduced.tolist(),
        n_components=n_components,
        explained_variance_ratio=pca.explained_variance_ratio_.tolist()
    )

@app.post("/train_classifier", response_model=ClassifierResponse, tags=["Transformers"])
async def train_classifier(config: ClassifierConfig):
    """Train a transformer-based text classifier on Reuters dataset."""
    start_time = time.time()

    # Load data
    (x_train, y_train), (x_test, y_test), num_classes, vocab_size = load_reuters_data(
        max_len=config.max_len
    )

    # Build model
    model = build_transformer_classifier(
        vocab_size=vocab_size,
        maxlen=config.max_len,
        num_classes=num_classes,
        embed_dim=config.embed_dim,
        num_heads=config.num_heads,
        num_transformer_blocks=config.num_transformer_blocks,
        ff_dim=config.ff_dim,
        dropout_rate=config.dropout_rate
    )

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    # Train
    history = model.fit(
        x_train, y_train,
        batch_size=config.batch_size,
        epochs=config.epochs,
        validation_split=0.1,
        verbose=1
    )

    # Evaluate
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)

    training_time = time.time() - start_time

    return ClassifierResponse(
        config={
            "embed_dim": config.embed_dim,
            "num_heads": config.num_heads,
            "num_transformer_blocks": config.num_transformer_blocks,
            "ff_dim": config.ff_dim,
            "dropout_rate": config.dropout_rate,
            "epochs": config.epochs,
            "max_len": config.max_len,
            "batch_size": config.batch_size,
            "num_classes": num_classes,
            "vocab_size": vocab_size
        },
        accuracy_history=history.history["accuracy"],
        val_accuracy_history=history.history["val_accuracy"],
        loss_history=history.history["loss"],
        val_loss_history=history.history["val_loss"],
        test_accuracy=float(test_accuracy),
        test_loss=float(test_loss),
        training_time=training_time,
        num_params=model.count_params()
    )

@app.get("/compare_transformers", response_model=CompareTransformersResponse, tags=["Transformers"])
async def compare_transformers(
    embed_dim: int = 64,
    num_heads: int = 2,
    ff_dim: int = 64,
    dropout_rate: float = 0.1,
    epochs: int = 5,
    max_len: int = 200,
    batch_size: int = 32
):
    """Compare 1-block vs 3-block transformer architectures."""

    # Load data once
    (x_train, y_train), (x_test, y_test), num_classes, vocab_size = load_reuters_data(
        max_len=max_len
    )

    results = {}

    for num_blocks in [1, 3]:
        start_time = time.time()

        model = build_transformer_classifier(
            vocab_size=vocab_size,
            maxlen=max_len,
            num_classes=num_classes,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_transformer_blocks=num_blocks,
            ff_dim=ff_dim,
            dropout_rate=dropout_rate
        )

        model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )

        history = model.fit(
            x_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_split=0.1,
            verbose=1
        )

        test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
        training_time = time.time() - start_time

        key = "one_block" if num_blocks == 1 else "three_blocks"
        results[key] = {
            "num_transformer_blocks": num_blocks,
            "accuracy_history": history.history["accuracy"],
            "val_accuracy_history": history.history["val_accuracy"],
            "loss_history": history.history["loss"],
            "val_loss_history": history.history["val_loss"],
            "test_accuracy": float(test_accuracy),
            "test_loss": float(test_loss),
            "training_time": training_time,
            "num_params": model.count_params()
        }

        # Clear session to free memory
        keras.backend.clear_session()

    # Comparison summary
    comparison = {
        "accuracy_improvement": results["three_blocks"]["test_accuracy"] - results["one_block"]["test_accuracy"],
        "parameter_increase": results["three_blocks"]["num_params"] - results["one_block"]["num_params"],
        "time_increase": results["three_blocks"]["training_time"] - results["one_block"]["training_time"],
        "winner": "three_blocks" if results["three_blocks"]["test_accuracy"] > results["one_block"]["test_accuracy"] else "one_block"
    }

    return CompareTransformersResponse(
        one_block=results["one_block"],
        three_blocks=results["three_blocks"],
        comparison=comparison
    )

@app.get("/available_words", tags=["Word Embeddings"])
async def get_available_words(limit: int = 100):
    """Get a sample of available words in the vocabulary."""
    if not glove_embeddings:
        load_glove_embeddings()

    words = list(glove_embeddings.keys())[:limit]
    return {
        "total_vocabulary": len(glove_embeddings),
        "sample_words": words
    }

@app.get("/model_info", tags=["Transformers"])
async def get_model_info():
    """Get information about model architecture options."""
    return {
        "embed_dim_options": [32, 64, 128, 256],
        "num_heads_options": [1, 2, 4, 8],
        "num_transformer_blocks_range": {"min": 1, "max": 4},
        "ff_dim_options": [32, 64, 128],
        "dropout_rate_range": {"min": 0.0, "max": 0.5},
        "epochs_range": {"min": 5, "max": 30},
        "dataset": "Reuters (46 categories)",
        "architecture": "Transformer with Multi-Head Attention"
    }

# Serve frontend
app.mount("/static", StaticFiles(directory="."), name="static")

@app.get("/app")
async def serve_frontend():
    """Serve the frontend application."""
    return FileResponse("index.html")


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
