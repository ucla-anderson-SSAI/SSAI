"""
Week 6: real embeddings + pretrained Tiny LLM inference.

The app does not train models at request time. It loads exported Keras models
from the Week 6 notebook and runs inference for next-token probabilities and
short generation. Embedding math uses real GloVe vectors when available; if a
deployment cannot download GloVe, the app falls back to a tiny deterministic
demo space so the class site still boots.
"""

from __future__ import annotations

import json
import os
import re
import threading
from typing import Dict, List

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
from flask import Flask, jsonify, request, send_file


app = Flask(__name__)
APP_DIR = os.path.dirname(os.path.abspath(__file__))
EXPORT_DIR = os.path.join(APP_DIR, "week6_tiny_llm_exports")
GLOVE_SUBSET = os.path.join(APP_DIR, "glove_subset.npz")
GLOVE_TEXT = os.path.join(APP_DIR, "glove.6B.50d.txt")

_embeddings: Dict[str, np.ndarray] = {}
_embedding_source = "not loaded"
_llm_config = None
_llm_history = None
_llm_models = {}
_llm_vocabs = {}
_llm_word_to_id = {}
_tf = None
_keras = None
_load_lock = threading.Lock()

MODEL_LABELS = {
    "10k_1epoch": "10,000 reviews, 1 epoch",
    "500k_1epoch": "500,000 reviews, 1 epoch",
    "500k_20epochs": "500,000 reviews, 20 epochs",
}


def _unit(vec: np.ndarray) -> np.ndarray:
    return vec / (np.linalg.norm(vec) + 1e-8)


def _toy_embeddings() -> None:
    """Small fallback used only if real GloVe cannot be loaded."""
    global _embedding_source
    rng = np.random.default_rng(42)
    dim = 48
    basis = np.eye(dim, dtype=np.float32)
    axes = {
        "royalty": basis[0],
        "male": basis[1],
        "female": basis[2],
        "animal": basis[3],
        "vehicle": basis[4],
        "food": basis[5],
        "positive": basis[6],
        "negative": basis[7],
        "place": basis[8],
        "capital": basis[9],
    }
    groups = {
        "male": ["man", "king", "prince", "father", "brother", "boy"],
        "female": ["woman", "queen", "princess", "mother", "sister", "girl"],
        "royalty": ["king", "queen", "prince", "princess"],
        "animal": ["dog", "cat", "bird", "fish", "horse", "cow"],
        "vehicle": ["car", "truck", "bus", "train", "plane", "boat"],
        "food": ["taco", "pizza", "sushi", "burger", "salad", "pasta"],
        "positive": ["good", "great", "excellent", "happy", "love", "delicious"],
        "negative": ["bad", "terrible", "awful", "sad", "hate", "bland"],
        "place": ["france", "germany", "italy", "japan", "china", "russia"],
        "capital": ["paris", "berlin", "rome", "tokyo", "beijing", "moscow"],
    }
    words = sorted({word for values in groups.values() for word in values})
    for word in words:
        vec = rng.normal(scale=0.08, size=dim)
        for group, members in groups.items():
            if word in members:
                vec += axes[group]
        if word == "king":
            vec += axes["royalty"] + axes["male"]
        if word == "queen":
            vec += axes["royalty"] + axes["female"]
        _embeddings[word] = _unit(vec.astype(np.float32))
    _embedding_source = "fallback demo vectors"


def load_embeddings(max_words: int = 50000) -> None:
    """Load real GloVe vectors into memory, with a tiny fallback."""
    global _embedding_source
    if _embeddings:
        return

    try:
        if os.path.exists(GLOVE_SUBSET):
            data = np.load(GLOVE_SUBSET, allow_pickle=True)
            words = data["words"].tolist()
            vectors = data["vectors"].astype(np.float32)
            vectors = vectors / (np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-8)
            for word, vec in zip(words, vectors):
                _embeddings[str(word)] = vec
            _embedding_source = "GloVe embeddings"
            return

        if not os.path.exists(GLOVE_TEXT):
            raise FileNotFoundError("glove_subset.npz not found")

        with open(GLOVE_TEXT, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= max_words:
                    break
                parts = line.strip().split()
                if len(parts) != 51:
                    continue
                _embeddings[parts[0]] = _unit(np.array(parts[1:], dtype=np.float32))
        _embedding_source = "GloVe embeddings"
    except Exception as exc:
        print(f"Could not load GloVe; using fallback embeddings: {exc}", flush=True)
        _embeddings.clear()
        _toy_embeddings()


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / ((np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8))


def nearest(vector: np.ndarray, exclude: List[str] | None = None, limit: int = 5):
    exclude_set = {w.lower() for w in (exclude or [])}
    scored = [
        {"word": word, "similarity": cosine(vector, emb)}
        for word, emb in _embeddings.items()
        if word not in exclude_set
    ]
    return sorted(scored, key=lambda row: row["similarity"], reverse=True)[:limit]


def load_tf():
    global _tf, _keras
    if _tf is not None and _keras is not None:
        return _tf, _keras
    import tensorflow as tf
    from tensorflow import keras

    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    _tf = tf
    _keras = keras
    return _tf, _keras


def load_llm_artifacts() -> None:
    global _llm_config, _llm_history
    if _llm_models:
        return
    with _load_lock:
        if _llm_models:
            return

        _tf_local, keras = load_tf()
        config_path = os.path.join(EXPORT_DIR, "config.json")
        history_path = os.path.join(EXPORT_DIR, "history.json")
        if not os.path.exists(config_path):
            raise RuntimeError(f"Missing Tiny LLM export folder: {EXPORT_DIR}")

        with open(config_path, "r", encoding="utf-8") as f:
            _llm_config = json.load(f)
        with open(history_path, "r", encoding="utf-8") as f:
            _llm_history = json.load(f)

        class CompatEmbedding(keras.layers.Embedding):
            @classmethod
            def from_config(cls, config):
                config.pop("quantization_config", None)
                return super().from_config(config)

        class CompatDense(keras.layers.Dense):
            @classmethod
            def from_config(cls, config):
                config.pop("quantization_config", None)
                return super().from_config(config)

        custom_objects = {
            "Embedding": CompatEmbedding,
            "Dense": CompatDense,
        }

        for key, model_file in _llm_config["models"].items():
            model_path = os.path.join(EXPORT_DIR, model_file)
            vocab_path = os.path.join(EXPORT_DIR, _llm_config["vocabs"][key])
            _llm_models[key] = keras.models.load_model(
                model_path,
                compile=False,
                custom_objects=custom_objects,
            )
            with open(vocab_path, "r", encoding="utf-8") as f:
                vocab = json.load(f)
            _llm_vocabs[key] = vocab
            _llm_word_to_id[key] = {word: idx for idx, word in enumerate(vocab)}


def tokenize_prompt(prompt: str, key: str) -> List[int]:
    """Match the notebook's lowercase + whitespace tokenization closely."""
    word_to_id = _llm_word_to_id[key]
    tokens = prompt.lower().strip().split()
    return [word_to_id.get(token, 1) for token in tokens if token]


def next_token_distribution(prompt: str, key: str, top_k: int = 10):
    load_llm_artifacts()
    tf, _keras_local = load_tf()
    model = _llm_models[key]
    vocab = _llm_vocabs[key]
    seq_len = int(_llm_config["seq_len"])
    ids = tokenize_prompt(prompt, key)
    if not ids:
        return []

    padded = np.zeros(seq_len, dtype="int32")
    context = ids[-seq_len:]
    padded[:len(context)] = context
    pred_pos = len(context) - 1

    logits = model.predict(padded[np.newaxis, :], verbose=0)[0][pred_pos]
    probs = tf.nn.softmax(logits).numpy().astype("float64")
    probs[0] = 0
    probs[1] = 0
    probs = probs / probs.sum()
    top_ids = np.argsort(probs)[-top_k:][::-1]
    return [
        {"token": vocab[int(i)], "probability": float(probs[int(i)]), "rank": rank + 1}
        for rank, i in enumerate(top_ids)
    ]


def generate_text(prompt: str, key: str, length: int = 20, temperature: float = 0.8):
    load_llm_artifacts()
    tf, _keras_local = load_tf()
    model = _llm_models[key]
    vocab = _llm_vocabs[key]
    seq_len = int(_llm_config["seq_len"])
    ids = tokenize_prompt(prompt, key)
    if not ids:
        return ""

    temperature = max(0.2, min(float(temperature), 1.5))
    rng = np.random.default_rng()
    for _ in range(max(1, min(int(length), 40))):
        padded = np.zeros(seq_len, dtype="int32")
        context = ids[-seq_len:]
        padded[:len(context)] = context
        pred_pos = len(context) - 1
        logits = model.predict(padded[np.newaxis, :], verbose=0)[0][pred_pos]
        logits = logits / temperature
        probs = tf.nn.softmax(logits).numpy().astype("float64")
        probs[0] = 0
        probs[1] = 0
        probs = probs / probs.sum()
        ids.append(int(rng.choice(len(probs), p=probs)))

    return " ".join(vocab[i] for i in ids if 0 <= i < len(vocab))


def model_summary(key: str):
    if _llm_history is None:
        load_llm_artifacts()
    hist = (_llm_history or {}).get(key, {})
    return {
        "key": key,
        "label": MODEL_LABELS.get(key, key),
        "train_loss": hist.get("loss", [None])[-1],
        "val_loss": hist.get("val_loss", [None])[-1],
    }


@app.get("/api/health")
def health():
    export_ready = os.path.exists(os.path.join(EXPORT_DIR, "config.json"))
    return jsonify({
        "status": "healthy",
        "service": "Week 6: Transformers and Next Token Prediction",
        "exports_ready": export_ready,
        "llm_loaded": bool(_llm_models),
        "embedding_source": _embedding_source,
        "num_embeddings": len(_embeddings),
    })


@app.get("/api/words")
def words():
    load_embeddings()
    return jsonify({"words": sorted(_embeddings.keys())[:500], "source": _embedding_source})


@app.post("/api/nearest")
def nearest_words():
    load_embeddings()
    payload = request.get_json(force=True)
    word = str(payload.get("word", "")).lower().strip()
    limit = max(1, min(int(payload.get("limit", 5)), 10))
    if word not in _embeddings:
        return jsonify({"error": f"'{word}' is not in the embedding vocabulary."}), 404
    rows = nearest(_embeddings[word], exclude=[word], limit=limit)
    return jsonify({
        "word": word,
        "source": _embedding_source,
        "neighbors": [{**row, "rank": idx + 1} for idx, row in enumerate(rows)],
    })


@app.post("/api/analogy")
def analogy():
    load_embeddings()
    payload = request.get_json(force=True)
    a = str(payload.get("a", "king")).lower().strip()
    b = str(payload.get("b", "woman")).lower().strip()
    c = str(payload.get("c", "man")).lower().strip()
    limit = max(1, min(int(payload.get("limit", 5)), 10))
    missing = [w for w in [a, b, c] if w not in _embeddings]
    if missing:
        return jsonify({"error": f"Missing words: {', '.join(missing)}"}), 404
    result = _embeddings[a] - _embeddings[b] + _embeddings[c]
    rows = nearest(result, exclude=[a, b, c], limit=limit)
    return jsonify({
        "relationship": f"{b} : {c} :: {a} : ?",
        "formula": f"{a} - {b} + {c}",
        "source": _embedding_source,
        "results": [{**row, "rank": idx + 1} for idx, row in enumerate(rows)],
    })


@app.get("/api/llm/models")
def llm_models():
    load_llm_artifacts()
    return jsonify({
        "models": [model_summary(key) for key in ["10k_1epoch", "500k_1epoch", "500k_20epochs"]],
        "config": _llm_config,
    })


@app.post("/api/llm/next-token")
def llm_next_token():
    payload = request.get_json(force=True)
    prompt = re.sub(r"\s+", " ", str(payload.get("prompt", "the fish tacos were")).strip())
    key = str(payload.get("model", "500k_20epochs"))
    top_k = max(1, min(int(payload.get("top_k", 10)), 10))
    if key not in MODEL_LABELS:
        return jsonify({"error": f"Unknown model: {key}"}), 400
    return jsonify({
        "prompt": prompt,
        "model": model_summary(key),
        "top_tokens": next_token_distribution(prompt, key, top_k=top_k),
    })


@app.post("/api/llm/generate")
def llm_generate():
    payload = request.get_json(force=True)
    prompt = re.sub(r"\s+", " ", str(payload.get("prompt", "the fish tacos were")).strip())
    key = str(payload.get("model", "500k_20epochs"))
    length = int(payload.get("length", 20))
    temperature = float(payload.get("temperature", 0.8))
    if key not in MODEL_LABELS:
        return jsonify({"error": f"Unknown model: {key}"}), 400
    return jsonify({
        "prompt": prompt,
        "model": model_summary(key),
        "text": generate_text(prompt, key, length=length, temperature=temperature),
    })


@app.post("/api/generate")
def generate_compare():
    """Compatibility endpoint: compare all three real pretrained models."""
    payload = request.get_json(force=True)
    prompt = re.sub(r"\s+", " ", str(payload.get("prompt", "the fish tacos were")).strip())
    results = []
    for key in ["10k_1epoch", "500k_1epoch", "500k_20epochs"]:
        summary = model_summary(key)
        results.append({
            "key": key,
            "label": summary["label"],
            "train_loss": summary["train_loss"],
            "val_loss": summary["val_loss"],
            "top_tokens": next_token_distribution(prompt, key, top_k=5),
            "continuation": generate_text(prompt, key, length=12, temperature=0.8),
        })
    return jsonify({"prompt": prompt, "results": results})


@app.get("/")
def root():
    return send_file(os.path.join(APP_DIR, "index.html"))


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8000"))
    app.run(host="0.0.0.0", port=port)
