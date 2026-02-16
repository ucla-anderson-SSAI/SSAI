import os
"""
Assignment 8: Building with Generative AI APIs (LLM Parameters)
FastAPI Backend - Simulated LLM Responses
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from typing import List, Optional
import random
import math
import time
import json

app = FastAPI(title="Assignment 8: Large Language Models")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response Models
class GenerateRequest(BaseModel):
    prompt: str
    temperature: float = Field(default=1.0, ge=0.1, le=2.0)
    top_k: int = Field(default=40, ge=1, le=50)
    top_p: float = Field(default=0.9, ge=0.1, le=1.0)
    num_outputs: int = Field(default=5, ge=1, le=10)

class ReviewRequest(BaseModel):
    review: str

class BatchReviewRequest(BaseModel):
    reviews: List[str]

# Simulated vocabulary for demonstrations
SAMPLE_VOCABULARY = {
    "weather": {
        "tokens": ["sunny", "cloudy", "rainy", "stormy", "clear", "foggy", "humid", "dry", "windy", "calm"],
        "base_probs": [0.25, 0.20, 0.15, 0.10, 0.10, 0.07, 0.05, 0.04, 0.02, 0.02]
    },
    "greeting": {
        "tokens": ["Hello", "Hi", "Hey", "Greetings", "Welcome", "Good day", "Howdy", "Salutations", "Yo", "Hola"],
        "base_probs": [0.30, 0.25, 0.15, 0.10, 0.08, 0.05, 0.03, 0.02, 0.01, 0.01]
    },
    "sentiment": {
        "tokens": ["great", "good", "okay", "fine", "excellent", "wonderful", "amazing", "decent", "poor", "terrible"],
        "base_probs": [0.22, 0.20, 0.15, 0.12, 0.10, 0.08, 0.06, 0.04, 0.02, 0.01]
    },
    "continuation": {
        "tokens": ["the", "a", "this", "that", "an", "some", "my", "your", "their", "our"],
        "base_probs": [0.28, 0.22, 0.12, 0.10, 0.08, 0.07, 0.05, 0.04, 0.02, 0.02]
    }
}

# Sample outputs for different prompts at different temperatures
SAMPLE_OUTPUTS = {
    "weather": {
        "low_temp": [
            "The weather today is sunny and pleasant with clear skies.",
            "The weather today is sunny and warm with clear conditions.",
            "The weather today is sunny with clear skies expected.",
        ],
        "mid_temp": [
            "The weather today is mostly cloudy with a chance of rain.",
            "The weather today brings warm temperatures and scattered clouds.",
            "The weather today seems humid with overcast conditions developing.",
        ],
        "high_temp": [
            "The weather today? Absolutely bonkers - sunshine mixed with existential fog!",
            "The weather today dances between moody storms and whimsical breezes.",
            "The weather today is conducting a symphony of chaotic atmospherics!",
        ]
    },
    "story": {
        "low_temp": [
            "Once upon a time, there was a young princess who lived in a castle.",
            "Once upon a time, there was a brave knight who sought adventure.",
            "Once upon a time, there was a wise wizard in a tall tower.",
        ],
        "mid_temp": [
            "Once upon a time, a curious fox discovered a hidden garden of dreams.",
            "Once upon a time, the moon decided to visit the ocean floor.",
            "Once upon a time, a clockwork bird sang songs of forgotten ages.",
        ],
        "high_temp": [
            "Once upon a time, quantum jellyfish negotiated treaties with sentient nebulae!",
            "Once upon a time, yesterday's tomorrow danced with next week's memories.",
            "Once upon a time, the concept of 'once' questioned its own existence.",
        ]
    },
    "default": {
        "low_temp": [
            "This is a standard response with predictable, common word choices.",
            "This is a typical response following expected patterns.",
            "This is a conventional response using common vocabulary.",
        ],
        "mid_temp": [
            "This response shows moderate variation in word selection and structure.",
            "This output demonstrates balanced creativity with coherent ideas.",
            "This generation balances novelty with meaningful content.",
        ],
        "high_temp": [
            "This response ventures into unexpected territories of linguistic exploration!",
            "This output embraces the beautiful chaos of unlimited possibility!",
            "This generation defies conventional patterns with wild abandon!",
        ]
    }
}

def apply_temperature(probs: List[float], temperature: float) -> List[float]:
    """Apply temperature scaling to probability distribution."""
    if temperature == 0:
        temperature = 0.01

    # Apply temperature: p_i^(1/T) / sum(p_j^(1/T))
    scaled = [p ** (1.0 / temperature) for p in probs]
    total = sum(scaled)
    return [p / total for p in scaled]

def apply_top_k(probs: List[float], k: int) -> List[float]:
    """Apply top-k filtering to probability distribution."""
    indexed = list(enumerate(probs))
    sorted_probs = sorted(indexed, key=lambda x: x[1], reverse=True)

    result = [0.0] * len(probs)
    kept_probs = []
    for i, (idx, prob) in enumerate(sorted_probs):
        if i < k:
            kept_probs.append((idx, prob))

    total = sum(p for _, p in kept_probs)
    for idx, prob in kept_probs:
        result[idx] = prob / total if total > 0 else 0

    return result

def apply_top_p(probs: List[float], p: float) -> List[float]:
    """Apply nucleus (top-p) sampling to probability distribution."""
    indexed = list(enumerate(probs))
    sorted_probs = sorted(indexed, key=lambda x: x[1], reverse=True)

    result = [0.0] * len(probs)
    cumulative = 0.0
    kept_probs = []

    for idx, prob in sorted_probs:
        if cumulative < p:
            kept_probs.append((idx, prob))
            cumulative += prob
        else:
            break

    total = sum(prob for _, prob in kept_probs)
    for idx, prob in kept_probs:
        result[idx] = prob / total if total > 0 else 0

    return result

def get_probability_visualization(vocab_key: str, temperature: float, top_k: int, top_p: float):
    """Generate probability distribution visualization data."""
    vocab = SAMPLE_VOCABULARY.get(vocab_key, SAMPLE_VOCABULARY["sentiment"])
    tokens = vocab["tokens"]
    base_probs = vocab["base_probs"].copy()

    # Apply transformations in sequence
    temp_probs = apply_temperature(base_probs, temperature)
    topk_probs = apply_top_k(temp_probs, top_k)
    final_probs = apply_top_p(topk_probs, top_p)

    return {
        "tokens": tokens,
        "base_probs": base_probs,
        "after_temperature": temp_probs,
        "after_top_k": topk_probs,
        "after_top_p": final_probs
    }

def select_output(prompt: str, temperature: float) -> str:
    """Select appropriate output based on prompt and temperature."""
    prompt_lower = prompt.lower()

    if "weather" in prompt_lower:
        outputs = SAMPLE_OUTPUTS["weather"]
    elif "story" in prompt_lower or "once upon" in prompt_lower:
        outputs = SAMPLE_OUTPUTS["story"]
    else:
        outputs = SAMPLE_OUTPUTS["default"]

    if temperature < 0.5:
        category = "low_temp"
    elif temperature < 1.2:
        category = "mid_temp"
    else:
        category = "high_temp"

    return random.choice(outputs[category])

def analyze_review_text(review: str) -> dict:
    """Simulate structured review analysis."""
    review_lower = review.lower()

    # Simulate sentiment detection
    positive_words = ["great", "excellent", "amazing", "love", "fantastic", "wonderful", "best", "perfect"]
    negative_words = ["bad", "terrible", "awful", "hate", "worst", "poor", "disappointing", "horrible"]

    positive_count = sum(1 for word in positive_words if word in review_lower)
    negative_count = sum(1 for word in negative_words if word in review_lower)

    if positive_count > negative_count:
        sentiment = "positive"
        rating = min(5, 3 + positive_count)
    elif negative_count > positive_count:
        sentiment = "negative"
        rating = max(1, 3 - negative_count)
    else:
        sentiment = "neutral"
        rating = 3

    # Extract topics
    topic_keywords = {
        "quality": ["quality", "well-made", "durable", "sturdy", "flimsy"],
        "price": ["price", "expensive", "cheap", "value", "worth", "cost"],
        "delivery": ["delivery", "shipping", "arrived", "fast", "slow"],
        "service": ["service", "support", "help", "customer", "staff"],
        "usability": ["easy", "difficult", "intuitive", "confusing", "simple"]
    }

    topics = []
    for topic, keywords in topic_keywords.items():
        if any(kw in review_lower for kw in keywords):
            topics.append(topic)

    if not topics:
        topics = ["general"]

    # Determine if action needed
    action_needed = sentiment == "negative" or rating <= 2

    return {
        "sentiment": sentiment,
        "confidence": round(random.uniform(0.75, 0.98), 2),
        "rating": rating,
        "topics": topics,
        "action_needed": action_needed,
        "action_reason": "Negative feedback requires follow-up" if action_needed else None,
        "key_phrases": extract_key_phrases(review),
        "word_count": len(review.split())
    }

def extract_key_phrases(text: str) -> List[str]:
    """Extract simulated key phrases from text."""
    words = text.split()
    phrases = []

    # Simple extraction of 2-3 word phrases
    for i in range(len(words) - 1):
        if len(words[i]) > 3 and len(words[i + 1]) > 3:
            phrases.append(f"{words[i]} {words[i + 1]}")

    return phrases[:3] if phrases else ["general feedback"]

@app.post("/generate")
async def generate_text(request: GenerateRequest):
    """Generate text with specified LLM parameters."""
    outputs = []

    for i in range(request.num_outputs):
        # Add slight variation to temperature for demonstration
        temp_variation = request.temperature + random.uniform(-0.1, 0.1)
        temp_variation = max(0.1, min(2.0, temp_variation))

        output = select_output(request.prompt, temp_variation)
        outputs.append({
            "text": output,
            "effective_temperature": round(temp_variation, 2),
            "tokens_generated": len(output.split()),
            "generation_id": i + 1
        })

    # Get vocabulary type for visualization
    prompt_lower = request.prompt.lower()
    if "weather" in prompt_lower:
        vocab_key = "weather"
    elif "hello" in prompt_lower or "greeting" in prompt_lower:
        vocab_key = "greeting"
    else:
        vocab_key = "sentiment"

    prob_viz = get_probability_visualization(
        vocab_key,
        request.temperature,
        request.top_k,
        request.top_p
    )

    return {
        "prompt": request.prompt,
        "parameters": {
            "temperature": request.temperature,
            "top_k": request.top_k,
            "top_p": request.top_p
        },
        "outputs": outputs,
        "probability_visualization": prob_viz,
        "explanation": get_parameter_explanation(request.temperature, request.top_k, request.top_p)
    }

def get_parameter_explanation(temperature: float, top_k: int, top_p: float) -> dict:
    """Provide educational explanation of current parameters."""
    if temperature < 0.5:
        temp_desc = "Very low temperature produces highly deterministic, repetitive outputs. The model strongly favors the most probable tokens."
    elif temperature < 1.0:
        temp_desc = "Low-to-moderate temperature balances coherence with some variation. Good for factual content."
    elif temperature < 1.5:
        temp_desc = "Moderate-to-high temperature increases creativity and diversity. Outputs become more varied and surprising."
    else:
        temp_desc = "High temperature produces very diverse, potentially chaotic outputs. Useful for brainstorming but may lose coherence."

    if top_k <= 10:
        topk_desc = f"Top-K={top_k} severely limits choices to only the {top_k} most probable tokens. Very focused output."
    elif top_k <= 30:
        topk_desc = f"Top-K={top_k} provides moderate token diversity while filtering unlikely options."
    else:
        topk_desc = f"Top-K={top_k} allows broad token selection, maintaining most of the probability mass."

    if top_p <= 0.5:
        topp_desc = f"Top-P={top_p} (nucleus sampling) keeps only tokens comprising {int(top_p*100)}% cumulative probability. Very focused."
    elif top_p <= 0.9:
        topp_desc = f"Top-P={top_p} balances diversity with quality by keeping the top {int(top_p*100)}% probability mass."
    else:
        topp_desc = f"Top-P={top_p} includes nearly all probable tokens, maximizing potential diversity."

    return {
        "temperature": temp_desc,
        "top_k": topk_desc,
        "top_p": topp_desc
    }

@app.post("/analyze_review")
async def analyze_review(request: ReviewRequest):
    """Analyze a single review and return structured JSON output."""
    if not request.review.strip():
        raise HTTPException(status_code=400, detail="Review text cannot be empty")

    start_time = time.time()

    # Simulate processing delay
    time.sleep(random.uniform(0.1, 0.3))

    analysis = analyze_review_text(request.review)

    processing_time = time.time() - start_time

    return {
        "success": True,
        "input": request.review,
        "analysis": analysis,
        "raw_json": json.dumps(analysis, indent=2),
        "parsing_status": "success",
        "processing_time_ms": round(processing_time * 1000, 2),
        "model_info": {
            "simulated_model": "gpt-4-turbo",
            "output_format": "JSON",
            "schema_validated": True
        }
    }

@app.post("/batch_analyze")
async def batch_analyze(request: BatchReviewRequest):
    """Analyze multiple reviews in batch with timing information."""
    if not request.reviews:
        raise HTTPException(status_code=400, detail="Reviews list cannot be empty")

    if len(request.reviews) > 20:
        raise HTTPException(status_code=400, detail="Maximum 20 reviews per batch")

    start_time = time.time()
    results = []

    for i, review in enumerate(request.reviews):
        review_start = time.time()

        # Simulate variable processing time
        time.sleep(random.uniform(0.05, 0.15))

        if review.strip():
            analysis = analyze_review_text(review)
            results.append({
                "index": i,
                "review": review[:100] + "..." if len(review) > 100 else review,
                "analysis": analysis,
                "processing_time_ms": round((time.time() - review_start) * 1000, 2),
                "status": "success"
            })
        else:
            results.append({
                "index": i,
                "review": "",
                "analysis": None,
                "processing_time_ms": 0,
                "status": "skipped",
                "error": "Empty review"
            })

    total_time = time.time() - start_time
    successful = sum(1 for r in results if r["status"] == "success")

    return {
        "batch_id": f"batch_{int(time.time())}",
        "total_reviews": len(request.reviews),
        "successful": successful,
        "failed": len(request.reviews) - successful,
        "results": results,
        "timing": {
            "total_time_ms": round(total_time * 1000, 2),
            "average_per_review_ms": round((total_time * 1000) / len(request.reviews), 2),
            "throughput_reviews_per_second": round(len(request.reviews) / total_time, 2)
        },
        "cost_estimate": {
            "estimated_tokens": sum(len(r.get("review", "").split()) * 2 for r in results),
            "estimated_cost_usd": round(len(request.reviews) * 0.002, 4),
            "note": "Simulated cost based on typical GPT-4 pricing"
        }
    }

@app.get("/probability_demo")
async def probability_demo():
    """Return sample probability distributions for different parameter combinations."""
    demos = []

    # Demo 1: Temperature comparison
    for temp in [0.2, 0.7, 1.0, 1.5, 2.0]:
        demos.append({
            "name": f"Temperature = {temp}",
            "category": "temperature",
            "data": get_probability_visualization("sentiment", temp, 50, 1.0)
        })

    # Demo 2: Top-K comparison
    for k in [3, 5, 10, 25, 50]:
        demos.append({
            "name": f"Top-K = {k}",
            "category": "top_k",
            "data": get_probability_visualization("sentiment", 1.0, k, 1.0)
        })

    # Demo 3: Top-P comparison
    for p in [0.3, 0.5, 0.7, 0.9, 1.0]:
        demos.append({
            "name": f"Top-P = {p}",
            "category": "top_p",
            "data": get_probability_visualization("sentiment", 1.0, 50, p)
        })

    # Demo 4: Combined effects
    demos.append({
        "name": "Low Temp + Low Top-K",
        "category": "combined",
        "data": get_probability_visualization("sentiment", 0.3, 5, 1.0)
    })
    demos.append({
        "name": "High Temp + High Top-P",
        "category": "combined",
        "data": get_probability_visualization("sentiment", 1.5, 50, 0.95)
    })
    demos.append({
        "name": "Balanced Settings",
        "category": "combined",
        "data": get_probability_visualization("sentiment", 0.7, 20, 0.9)
    })

    return {
        "demos": demos,
        "vocabulary_info": {
            "tokens": SAMPLE_VOCABULARY["sentiment"]["tokens"],
            "description": "Sentiment-related tokens used for demonstration"
        },
        "explanation": {
            "temperature": "Reshapes the probability distribution. Low values sharpen (favor top tokens), high values flatten (more uniform).",
            "top_k": "Keeps only the K most probable tokens, zeroing out the rest before renormalization.",
            "top_p": "Keeps the smallest set of tokens whose cumulative probability exceeds P (nucleus sampling)."
        }
    }

@app.get("/")
async def root():
    return FileResponse("index.html")

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8107))
    uvicorn.run(app, host="0.0.0.0", port=port)
