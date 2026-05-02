"""
Week 6 Tiny LLM Lab.

This app is intentionally lightweight for Railway hobby deployments. It avoids
runtime model training and external dataset downloads, while preserving the
core Week 6 ideas students need to reason about: embeddings, analogies,
next-token prediction, and how more data/training changes generation.
"""

from __future__ import annotations

import os
import re
from typing import Dict, List

import numpy as np
from flask import Flask, jsonify, request, send_file


app = Flask(__name__)
APP_DIR = os.path.dirname(os.path.abspath(__file__))

EMBEDDING_DIM = 48
_embeddings: Dict[str, np.ndarray] = {}


def _unit(vec: np.ndarray) -> np.ndarray:
    return vec / (np.linalg.norm(vec) + 1e-8)


def load_embeddings() -> None:
    """Create a deterministic toy embedding space with visible relationships."""
    if _embeddings:
        return

    rng = np.random.default_rng(42)
    basis = np.eye(EMBEDDING_DIM, dtype=np.float32)
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
    words += [
        "restaurant", "service", "waiter", "menu", "price", "table",
        "review", "customer", "business", "model", "token", "attention",
        "data", "training", "prompt", "language", "movie", "music",
    ]

    for word in words:
        vec = rng.normal(scale=0.08, size=EMBEDDING_DIM)
        for group, members in groups.items():
            if word in members:
                vec += axes[group]
        if word == "king":
            vec += axes["royalty"] + axes["male"]
        if word == "queen":
            vec += axes["royalty"] + axes["female"]
        if word == "paris":
            vec += axes["place"] * 0.35 + axes["capital"]
        if word == "france":
            vec += axes["place"]
        _embeddings[word] = _unit(vec.astype(np.float32))


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


EXPERIMENTS = {
    "small": {
        "label": "10,000 reviews, 1 epoch",
        "reviews": 10000,
        "epochs": 1,
        "train_loss": 5.92,
        "val_loss": 6.08,
        "description": "Small-data baseline. It learns common review words but is often generic.",
    },
    "more_data": {
        "label": "500,000 reviews, 1 epoch",
        "reviews": 500000,
        "epochs": 1,
        "train_loss": 4.74,
        "val_loss": 4.86,
        "description": "More examples improve the next-token distribution even with one pass.",
    },
    "more_training": {
        "label": "500,000 reviews, 5 epochs",
        "reviews": 500000,
        "epochs": 5,
        "train_loss": 3.52,
        "val_loss": 3.89,
        "description": "More training makes continuations more review-like, though still imperfect.",
    },
}

NEXT_TOKEN_BANK = {
    "tacos": [
        ("small", [("good", 0.14), ("great", 0.11), ("fresh", 0.08), ("and", 0.07), ("but", 0.06)]),
        ("more_data", [("delicious", 0.18), ("fresh", 0.13), ("amazing", 0.10), ("and", 0.08), ("with", 0.06)]),
        ("more_training", [("delicious", 0.24), ("fresh", 0.16), ("perfectly", 0.11), ("with", 0.08), ("and", 0.06)]),
    ],
    "service": [
        ("small", [("good", 0.16), ("bad", 0.09), ("was", 0.08), ("and", 0.07), ("the", 0.06)]),
        ("more_data", [("friendly", 0.18), ("slow", 0.13), ("excellent", 0.10), ("attentive", 0.08), ("was", 0.06)]),
        ("more_training", [("friendly", 0.22), ("attentive", 0.17), ("excellent", 0.12), ("slow", 0.08), ("and", 0.05)]),
    ],
    "default": [
        ("small", [("the", 0.18), ("and", 0.12), ("good", 0.09), ("was", 0.08), ("food", 0.05)]),
        ("more_data", [("food", 0.16), ("service", 0.11), ("great", 0.10), ("was", 0.08), ("place", 0.06)]),
        ("more_training", [("food", 0.18), ("service", 0.13), ("experience", 0.10), ("delicious", 0.08), ("place", 0.06)]),
    ],
}

CONTINUATIONS = {
    "tacos": {
        "small": "good and the place was good but not very good",
        "more_data": "delicious and the service was friendly with fresh salsa",
        "more_training": "delicious with bright salsa and friendly service throughout dinner",
    },
    "service": {
        "small": "good and the place was good but not very good",
        "more_data": "friendly and the staff checked on our table",
        "more_training": "friendly, attentive, and quick without making dinner feel rushed",
    },
    "default": {
        "small": "the food was good and the place was good",
        "more_data": "the food and service made the visit worthwhile",
        "more_training": "the food, service, and atmosphere made the visit feel memorable",
    },
}


def prompt_bucket(prompt: str) -> str:
    text = prompt.lower()
    if "taco" in text or "fish" in text:
        return "tacos"
    if "service" in text or "waiter" in text:
        return "service"
    return "default"


@app.get("/api/health")
def health():
    load_embeddings()
    return jsonify({
        "status": "healthy",
        "service": "Week 6 Tiny LLM Lab",
        "num_embeddings": len(_embeddings),
    })


@app.get("/api/words")
def words():
    load_embeddings()
    return jsonify({"words": sorted(_embeddings.keys())})


@app.post("/api/nearest")
def nearest_words():
    load_embeddings()
    payload = request.get_json(force=True)
    word = str(payload.get("word", "")).lower().strip()
    if word not in _embeddings:
        return jsonify({"error": f"'{word}' is not in the demo vocabulary."}), 404
    return jsonify({"word": word, "neighbors": nearest(_embeddings[word], exclude=[word], limit=5)})


@app.post("/api/analogy")
def analogy():
    load_embeddings()
    payload = request.get_json(force=True)
    a = str(payload.get("a", "king")).lower().strip()
    b = str(payload.get("b", "man")).lower().strip()
    c = str(payload.get("c", "woman")).lower().strip()
    missing = [w for w in [a, b, c] if w not in _embeddings]
    if missing:
        return jsonify({"error": f"Missing words: {', '.join(missing)}"}), 404
    result = _embeddings[a] - _embeddings[b] + _embeddings[c]
    return jsonify({
        "formula": f"{a} - {b} + {c}",
        "results": nearest(result, exclude=[a, b, c], limit=5),
    })


@app.get("/api/experiments")
def experiments():
    return jsonify({"experiments": EXPERIMENTS})


@app.post("/api/generate")
def generate():
    payload = request.get_json(force=True)
    prompt = re.sub(r"\s+", " ", str(payload.get("prompt", "the fish tacos were")).strip())
    bucket = prompt_bucket(prompt)

    results = []
    for key in ["small", "more_data", "more_training"]:
        distribution = dict(NEXT_TOKEN_BANK[bucket])[key]
        results.append({
            "key": key,
            "label": EXPERIMENTS[key]["label"],
            "train_loss": EXPERIMENTS[key]["train_loss"],
            "val_loss": EXPERIMENTS[key]["val_loss"],
            "top_tokens": [{"token": token, "probability": prob} for token, prob in distribution],
            "continuation": f"{prompt} {CONTINUATIONS[bucket][key]}",
        })

    return jsonify({"prompt": prompt, "bucket": bucket, "results": results})


@app.get("/")
def root():
    return send_file(os.path.join(APP_DIR, "index.html"))


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8000"))
    app.run(host="0.0.0.0", port=port)
