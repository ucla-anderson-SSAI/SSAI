"""
Week 8: LLM APIs and Agents - FastAPI Backend
Demonstrates API parameter effects and agent behavior through simulation.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import numpy as np
import random
from enum import Enum

router = APIRouter()

# CORS middleware
# CORS handled by main app

# =============================================================================
# Helper Data
# =============================================================================

# Sample vocabulary for demonstrations
SAMPLE_VOCABULARY = [
    "the", "a", "is", "are", "was", "were", "will", "can", "could",
    "happy", "sad", "excited", "calm", "nervous",
    "quickly", "slowly", "carefully", "loudly", "quietly",
    "run", "walk", "jump", "think", "create", "build", "explore",
    "sun", "moon", "star", "ocean", "mountain", "forest", "city"
]

# Sample prompts for different use cases
SAMPLE_PROMPTS = {
    "creative_writing": [
        "Write a short story about a robot learning to paint",
        "Describe a magical forest at sunset",
        "Create a poem about the ocean"
    ],
    "code_generation": [
        "Write a Python function to sort a list",
        "Create a JavaScript class for a shopping cart",
        "Implement a binary search algorithm"
    ],
    "question_answering": [
        "What is machine learning?",
        "Explain how neural networks work",
        "What are the benefits of renewable energy?"
    ],
    "summarization": [
        "Summarize the main points of climate change",
        "Give a brief overview of the Industrial Revolution",
        "Explain the key concepts of quantum computing"
    ]
}

# Pre-defined responses for simulation (keyed by prompt type and temperature range)
SIMULATED_RESPONSES = {
    "creative": {
        "low_temp": "The robot carefully studied the canvas, its sensors analyzing each brushstroke with precision.",
        "mid_temp": "With trembling mechanical fingers, the robot touched paint to canvas, discovering colors it had never truly seen before.",
        "high_temp": "Canvas exploding with chromatic fury! The robot's arm danced—whirled—splattered dreams in electric blue and sunset orange!"
    },
    "technical": {
        "low_temp": "The function iterates through the list, comparing adjacent elements and swapping them if necessary.",
        "mid_temp": "This sorting algorithm uses a divide-and-conquer approach, recursively splitting the array into smaller segments.",
        "high_temp": "Imagine the data as dancers at a ball, each number finding its perfect partner through elegant comparisons!"
    },
    "analytical": {
        "low_temp": "Machine learning is a subset of artificial intelligence that enables systems to learn from data.",
        "mid_temp": "Machine learning represents a paradigm shift in computing, where systems improve through experience rather than explicit programming.",
        "high_temp": "Picture a curious child learning to recognize faces—that's machine learning! Systems that grow, adapt, and surprise us!"
    },
    "default": {
        "low_temp": "This is a straightforward and predictable response.",
        "mid_temp": "Here's a balanced response with some creative elements mixed with clarity.",
        "high_temp": "Unexpected pathways emerge! The response ventures into creative territory with surprising connections!"
    }
}

# Agent's simulated database
AGENT_DATABASE = {
    "users": [
        {"id": 1, "name": "Alice Johnson", "email": "alice@example.com", "department": "Engineering"},
        {"id": 2, "name": "Bob Smith", "email": "bob@example.com", "department": "Marketing"},
        {"id": 3, "name": "Carol Williams", "email": "carol@example.com", "department": "Sales"},
        {"id": 4, "name": "David Brown", "email": "david@example.com", "department": "Engineering"},
    ],
    "products": [
        {"id": 101, "name": "Widget Pro", "price": 29.99, "stock": 150},
        {"id": 102, "name": "Gadget Plus", "price": 49.99, "stock": 75},
        {"id": 103, "name": "Super Tool", "price": 19.99, "stock": 200},
    ],
    "orders": [
        {"id": 1001, "user_id": 1, "product_id": 101, "quantity": 2, "status": "shipped"},
        {"id": 1002, "user_id": 2, "product_id": 102, "quantity": 1, "status": "pending"},
        {"id": 1003, "user_id": 3, "product_id": 103, "quantity": 5, "status": "delivered"},
    ]
}

# Example probability distributions
EXAMPLE_DISTRIBUTIONS = {
    "uniform": [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
    "peaked": [0.4, 0.25, 0.15, 0.1, 0.05, 0.02, 0.015, 0.01, 0.005, 0.0],
    "bimodal": [0.3, 0.05, 0.05, 0.1, 0.3, 0.05, 0.05, 0.05, 0.03, 0.02],
    "long_tail": [0.5, 0.2, 0.1, 0.08, 0.05, 0.03, 0.02, 0.01, 0.005, 0.005]
}

# =============================================================================
# Pydantic Models
# =============================================================================

class TemperatureRequest(BaseModel):
    base_probs: List[float] = Field(..., description="Base probability distribution")
    temperature: float = Field(..., ge=0.1, le=2.0, description="Temperature value (0.1-2.0)")
    token_labels: Optional[List[str]] = Field(None, description="Optional labels for tokens")

class TemperatureResponse(BaseModel):
    original_probs: List[float]
    adjusted_probs: List[float]
    temperature: float
    token_labels: List[str]
    explanation: str
    entropy_original: float
    entropy_adjusted: float

class SamplingRequest(BaseModel):
    probs: List[float] = Field(..., description="Probability distribution")
    top_k: Optional[int] = Field(None, ge=1, description="Top-k sampling parameter")
    top_p: Optional[float] = Field(None, ge=0.0, le=1.0, description="Top-p (nucleus) sampling parameter")
    token_labels: Optional[List[str]] = Field(None, description="Optional labels for tokens")

class SamplingResponse(BaseModel):
    original_probs: List[float]
    filtered_probs: List[float]
    included_tokens: List[Dict[str, Any]]
    excluded_tokens: List[Dict[str, Any]]
    top_k_applied: Optional[int]
    top_p_applied: Optional[float]
    cumulative_prob_included: float
    explanation: str

class GenerateRequest(BaseModel):
    prompt: str = Field(..., description="Input prompt for generation")
    temperature: float = Field(0.7, ge=0.0, le=2.0, description="Temperature for randomness")
    top_k: Optional[int] = Field(None, ge=1, description="Top-k sampling")
    top_p: Optional[float] = Field(None, ge=0.0, le=1.0, description="Top-p sampling")
    max_tokens: int = Field(100, ge=1, le=500, description="Maximum tokens to generate")

class GenerateResponse(BaseModel):
    prompt: str
    generated_text: str
    parameters_used: Dict[str, Any]
    token_count: int
    prompt_category: str
    explanation: str

class AgentRequest(BaseModel):
    goal: str = Field(..., description="The goal for the agent to accomplish")
    max_steps: int = Field(5, ge=1, le=10, description="Maximum steps for agent execution")

class AgentStep(BaseModel):
    step_number: int
    thought: str
    action: str
    action_input: Dict[str, Any]
    observation: str

class AgentResponse(BaseModel):
    goal: str
    steps: List[AgentStep]
    final_result: str
    tools_used: List[str]
    success: bool
    total_steps: int

# =============================================================================
# Helper Functions
# =============================================================================

def apply_temperature(probs: List[float], temperature: float) -> List[float]:
    """Apply temperature scaling to probability distribution."""
    probs_array = np.array(probs, dtype=np.float64)
    # Avoid log(0) by adding small epsilon
    probs_array = np.clip(probs_array, 1e-10, 1.0)

    # Convert to logits, apply temperature, convert back
    logits = np.log(probs_array)
    scaled_logits = logits / temperature

    # Softmax to get new probabilities
    exp_logits = np.exp(scaled_logits - np.max(scaled_logits))
    adjusted_probs = exp_logits / np.sum(exp_logits)

    return adjusted_probs.tolist()

def calculate_entropy(probs: List[float]) -> float:
    """Calculate Shannon entropy of a distribution."""
    probs_array = np.array(probs, dtype=np.float64)
    probs_array = np.clip(probs_array, 1e-10, 1.0)
    return -np.sum(probs_array * np.log2(probs_array))

def apply_top_k(probs: List[float], k: int) -> List[float]:
    """Apply top-k filtering to probabilities."""
    probs_array = np.array(probs)
    indices = np.argsort(probs_array)[::-1]
    filtered = np.zeros_like(probs_array)
    top_k_indices = indices[:k]
    filtered[top_k_indices] = probs_array[top_k_indices]
    # Renormalize
    if filtered.sum() > 0:
        filtered = filtered / filtered.sum()
    return filtered.tolist()

def apply_top_p(probs: List[float], p: float) -> List[float]:
    """Apply top-p (nucleus) filtering to probabilities."""
    probs_array = np.array(probs)
    sorted_indices = np.argsort(probs_array)[::-1]
    sorted_probs = probs_array[sorted_indices]
    cumsum = np.cumsum(sorted_probs)

    # Find cutoff
    cutoff_idx = np.searchsorted(cumsum, p) + 1
    cutoff_idx = min(cutoff_idx, len(probs))

    filtered = np.zeros_like(probs_array)
    included_indices = sorted_indices[:cutoff_idx]
    filtered[included_indices] = probs_array[included_indices]

    # Renormalize
    if filtered.sum() > 0:
        filtered = filtered / filtered.sum()
    return filtered.tolist()

def categorize_prompt(prompt: str) -> str:
    """Categorize a prompt into a response type."""
    prompt_lower = prompt.lower()

    creative_keywords = ["story", "poem", "creative", "imagine", "describe", "write about", "magical", "fantasy"]
    technical_keywords = ["function", "code", "algorithm", "implement", "class", "program", "python", "javascript"]
    analytical_keywords = ["what is", "explain", "how does", "why", "define", "summarize", "overview"]

    for keyword in creative_keywords:
        if keyword in prompt_lower:
            return "creative"

    for keyword in technical_keywords:
        if keyword in prompt_lower:
            return "technical"

    for keyword in analytical_keywords:
        if keyword in prompt_lower:
            return "analytical"

    return "default"

def get_temperature_range(temperature: float) -> str:
    """Get temperature range label."""
    if temperature < 0.5:
        return "low_temp"
    elif temperature < 1.2:
        return "mid_temp"
    else:
        return "high_temp"

def simulate_tool_search_database(query: str) -> str:
    """Simulate searching the agent's database."""
    query_lower = query.lower()
    results = []

    # Search users
    if "user" in query_lower or "engineer" in query_lower or "marketing" in query_lower or "sales" in query_lower:
        for user in AGENT_DATABASE["users"]:
            if any(term in query_lower for term in [user["name"].lower(), user["department"].lower()]):
                results.append(f"User: {user['name']} ({user['department']}) - {user['email']}")
        if "engineering" in query_lower:
            engineers = [u for u in AGENT_DATABASE["users"] if u["department"] == "Engineering"]
            results = [f"User: {u['name']} - {u['email']}" for u in engineers]

    # Search products
    if "product" in query_lower or "widget" in query_lower or "gadget" in query_lower or "stock" in query_lower:
        for product in AGENT_DATABASE["products"]:
            if product["name"].lower() in query_lower or "product" in query_lower:
                results.append(f"Product: {product['name']} - ${product['price']} (Stock: {product['stock']})")

    # Search orders
    if "order" in query_lower or "pending" in query_lower or "shipped" in query_lower:
        for order in AGENT_DATABASE["orders"]:
            if order["status"] in query_lower or "order" in query_lower:
                user = next((u for u in AGENT_DATABASE["users"] if u["id"] == order["user_id"]), None)
                product = next((p for p in AGENT_DATABASE["products"] if p["id"] == order["product_id"]), None)
                if user and product:
                    results.append(f"Order #{order['id']}: {user['name']} ordered {order['quantity']}x {product['name']} - Status: {order['status']}")

    if not results:
        return "No matching records found in database."

    return "\n".join(results[:5])  # Limit to 5 results

def simulate_tool_calculate(expression: str) -> str:
    """Simulate a calculation tool."""
    try:
        # Simple safe evaluation for basic math
        allowed_chars = set("0123456789+-*/().% ")
        if all(c in allowed_chars for c in expression):
            result = eval(expression)
            return f"Calculation result: {expression} = {result}"
        else:
            # Simulate some common calculations
            if "total" in expression.lower() or "sum" in expression.lower():
                return "Calculation result: Total = 299.85"
            elif "average" in expression.lower():
                return "Calculation result: Average = 33.32"
            else:
                return f"Calculated: {expression} (simulated result: 42)"
    except Exception:
        return f"Calculated: {expression} (simulated result: 42)"

def simulate_tool_send_email(to: str, subject: str, body: str) -> str:
    """Simulate sending an email."""
    return f"Email sent successfully to {to} with subject '{subject}'"

# =============================================================================
# API Endpoints
# =============================================================================

@router.get("/")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "week": 8,
        "topic": "LLM APIs and Agents",
        "note": "This backend provides SIMULATED responses for educational purposes",
        "endpoints": {
            "Part A - LLM Parameters": [
                "POST /visualize_temperature",
                "POST /visualize_sampling",
                "POST /generate"
            ],
            "Part B - AI Agents": [
                "POST /agent/run"
            ],
            "Helper Data": [
                "GET /sample_prompts",
                "GET /example_distributions",
                "GET /agent_database"
            ]
        }
    }

# -----------------------------------------------------------------------------
# Part A: LLM API Parameters
# -----------------------------------------------------------------------------

@router.post("/visualize_temperature", response_model=TemperatureResponse)
async def visualize_temperature(request: TemperatureRequest):
    """
    Visualize how temperature affects probability distribution.

    Lower temperature (< 1.0): Makes distribution more peaked (more deterministic)
    Higher temperature (> 1.0): Makes distribution more uniform (more random)
    """
    if len(request.base_probs) == 0:
        raise HTTPException(status_code=400, detail="base_probs cannot be empty")

    # Normalize input probabilities
    probs_sum = sum(request.base_probs)
    if probs_sum <= 0:
        raise HTTPException(status_code=400, detail="Probabilities must sum to a positive value")

    normalized_probs = [p / probs_sum for p in request.base_probs]

    # Apply temperature
    adjusted_probs = apply_temperature(normalized_probs, request.temperature)

    # Generate token labels if not provided
    token_labels = request.token_labels or [f"token_{i}" for i in range(len(normalized_probs))]
    if len(token_labels) < len(normalized_probs):
        token_labels.extend([f"token_{i}" for i in range(len(token_labels), len(normalized_probs))])

    # Calculate entropy
    entropy_orig = calculate_entropy(normalized_probs)
    entropy_adj = calculate_entropy(adjusted_probs)

    # Generate explanation
    if request.temperature < 1.0:
        explanation = (
            f"With temperature={request.temperature} (< 1.0), the distribution becomes more peaked. "
            f"The model is more likely to select high-probability tokens, making outputs more deterministic and focused. "
            f"Entropy decreased from {entropy_orig:.3f} to {entropy_adj:.3f} bits."
        )
    elif request.temperature > 1.0:
        explanation = (
            f"With temperature={request.temperature} (> 1.0), the distribution becomes more uniform. "
            f"Lower-probability tokens become more likely to be selected, increasing creativity and randomness. "
            f"Entropy increased from {entropy_orig:.3f} to {entropy_adj:.3f} bits."
        )
    else:
        explanation = (
            f"With temperature=1.0, the distribution remains unchanged. "
            f"This is the 'neutral' temperature setting. Entropy: {entropy_orig:.3f} bits."
        )

    return TemperatureResponse(
        original_probs=normalized_probs,
        adjusted_probs=adjusted_probs,
        temperature=request.temperature,
        token_labels=token_labels[:len(normalized_probs)],
        explanation=explanation,
        entropy_original=entropy_orig,
        entropy_adjusted=entropy_adj
    )

@router.post("/visualize_sampling", response_model=SamplingResponse)
async def visualize_sampling(request: SamplingRequest):
    """
    Visualize how top_k and top_p sampling affect token selection.

    Top-k: Only consider the k highest probability tokens
    Top-p (nucleus): Only consider tokens whose cumulative probability reaches p
    """
    if len(request.probs) == 0:
        raise HTTPException(status_code=400, detail="probs cannot be empty")

    # Normalize input probabilities
    probs_sum = sum(request.probs)
    if probs_sum <= 0:
        raise HTTPException(status_code=400, detail="Probabilities must sum to a positive value")

    normalized_probs = [p / probs_sum for p in request.probs]

    # Generate token labels
    token_labels = request.token_labels or [f"token_{i}" for i in range(len(normalized_probs))]
    if len(token_labels) < len(normalized_probs):
        token_labels.extend([f"token_{i}" for i in range(len(token_labels), len(normalized_probs))])

    # Apply sampling filters
    filtered_probs = normalized_probs.copy()

    if request.top_k is not None:
        filtered_probs = apply_top_k(filtered_probs, request.top_k)

    if request.top_p is not None:
        filtered_probs = apply_top_p(filtered_probs, request.top_p)

    # Categorize tokens
    included_tokens = []
    excluded_tokens = []

    for i, (orig, filt, label) in enumerate(zip(normalized_probs, filtered_probs, token_labels)):
        token_info = {
            "index": i,
            "label": label,
            "original_prob": round(orig, 6),
            "filtered_prob": round(filt, 6)
        }
        if filt > 0:
            included_tokens.append(token_info)
        else:
            excluded_tokens.append(token_info)

    # Sort by probability
    included_tokens.sort(key=lambda x: x["original_prob"], reverse=True)
    excluded_tokens.sort(key=lambda x: x["original_prob"], reverse=True)

    cumulative_included = sum(t["original_prob"] for t in included_tokens)

    # Generate explanation
    explanations = []
    if request.top_k is not None:
        explanations.append(f"Top-k={request.top_k} keeps only the {request.top_k} highest probability tokens.")
    if request.top_p is not None:
        explanations.append(f"Top-p={request.top_p} keeps tokens until cumulative probability reaches {request.top_p}.")

    explanations.append(f"{len(included_tokens)} tokens included, {len(excluded_tokens)} excluded.")
    explanations.append(f"Included tokens cover {cumulative_included*100:.1f}% of original probability mass.")

    return SamplingResponse(
        original_probs=normalized_probs,
        filtered_probs=filtered_probs,
        included_tokens=included_tokens,
        excluded_tokens=excluded_tokens,
        top_k_applied=request.top_k,
        top_p_applied=request.top_p,
        cumulative_prob_included=cumulative_included,
        explanation=" ".join(explanations)
    )

@router.post("/generate", response_model=GenerateResponse)
async def generate_text(request: GenerateRequest):
    """
    SIMULATED text generation endpoint.

    Returns pre-written responses that vary based on:
    - Prompt category (creative, technical, analytical)
    - Temperature setting (affects response style)

    This demonstrates how parameters would affect real LLM output.
    """
    # Categorize the prompt
    category = categorize_prompt(request.prompt)
    temp_range = get_temperature_range(request.temperature)

    # Get base response
    base_response = SIMULATED_RESPONSES.get(category, SIMULATED_RESPONSES["default"])[temp_range]

    # Add some variation based on temperature
    if request.temperature > 1.5:
        # High temperature: add some "randomness"
        random_additions = [
            " Unexpected connections emerge!",
            " The boundaries of convention dissolve...",
            " *creative flourish*",
        ]
        base_response += random.choice(random_additions)
    elif request.temperature < 0.3:
        # Very low temperature: make it more "robotic"
        base_response = base_response.split(".")[0] + "."  # Truncate to first sentence

    # Simulate token count (rough estimate)
    token_count = min(len(base_response.split()) + random.randint(0, 10), request.max_tokens)

    # Build explanation
    explanation_parts = [
        f"Prompt categorized as '{category}' content.",
        f"Temperature {request.temperature} produces {temp_range.replace('_', ' ')} style response."
    ]

    if request.top_k:
        explanation_parts.append(f"Top-k={request.top_k} limits vocabulary diversity.")
    if request.top_p:
        explanation_parts.append(f"Top-p={request.top_p} uses nucleus sampling.")

    explanation_parts.append("Note: This is a SIMULATED response for educational purposes.")

    return GenerateResponse(
        prompt=request.prompt,
        generated_text=base_response,
        parameters_used={
            "temperature": request.temperature,
            "top_k": request.top_k,
            "top_p": request.top_p,
            "max_tokens": request.max_tokens
        },
        token_count=token_count,
        prompt_category=category,
        explanation=" ".join(explanation_parts)
    )

# -----------------------------------------------------------------------------
# Part B: AI Agents
# -----------------------------------------------------------------------------

@router.post("/agent/run", response_model=AgentResponse)
async def run_agent(request: AgentRequest):
    """
    Run a SIMULATED AI agent with tools.

    The agent follows a think-act-observe loop:
    1. Think: Analyze the current situation
    2. Act: Choose and execute a tool
    3. Observe: Process the result
    4. Repeat until goal is achieved or max_steps reached

    Available tools:
    - search_database: Query the simulated database
    - calculate: Perform calculations
    - send_email: Send simulated emails
    """
    goal_lower = request.goal.lower()
    steps = []
    tools_used = []

    # Simulate agent reasoning based on goal keywords
    if "engineer" in goal_lower or "user" in goal_lower or "find" in goal_lower:
        # Database search scenario
        steps.append(AgentStep(
            step_number=1,
            thought="I need to search the database to find the relevant information. Let me query for users or specific criteria.",
            action="search_database",
            action_input={"query": "engineering users"},
            observation=simulate_tool_search_database("engineering users")
        ))
        tools_used.append("search_database")

        if len(steps) < request.max_steps and "email" in goal_lower:
            steps.append(AgentStep(
                step_number=2,
                thought="Now I have the user information. The goal mentions sending an email, so I should compose and send it.",
                action="send_email",
                action_input={"to": "alice@example.com", "subject": "Team Update", "body": "Hello team..."},
                observation=simulate_tool_send_email("alice@example.com", "Team Update", "Hello team...")
            ))
            tools_used.append("send_email")

        final_result = "Successfully found engineering team members and completed the requested action."

    elif "order" in goal_lower or "product" in goal_lower or "inventory" in goal_lower:
        # Order/product scenario
        steps.append(AgentStep(
            step_number=1,
            thought="I need to check the database for order or product information.",
            action="search_database",
            action_input={"query": "orders pending"},
            observation=simulate_tool_search_database("orders pending")
        ))
        tools_used.append("search_database")

        if len(steps) < request.max_steps:
            steps.append(AgentStep(
                step_number=2,
                thought="Let me also check the product inventory status.",
                action="search_database",
                action_input={"query": "product stock"},
                observation=simulate_tool_search_database("product stock")
            ))

        if len(steps) < request.max_steps and ("total" in goal_lower or "calculate" in goal_lower):
            steps.append(AgentStep(
                step_number=3,
                thought="I should calculate the total value or quantity requested.",
                action="calculate",
                action_input={"expression": "29.99 * 2 + 49.99 * 1 + 19.99 * 5"},
                observation=simulate_tool_calculate("29.99 * 2 + 49.99 * 1 + 19.99 * 5")
            ))
            tools_used.append("calculate")

        final_result = "Retrieved order and product information. Analysis complete."

    elif "calculate" in goal_lower or "math" in goal_lower or "compute" in goal_lower:
        # Calculation scenario
        steps.append(AgentStep(
            step_number=1,
            thought="This goal requires mathematical computation. I'll use the calculate tool.",
            action="calculate",
            action_input={"expression": "100 * 1.15 + 50 * 0.85"},
            observation=simulate_tool_calculate("100 * 1.15 + 50 * 0.85")
        ))
        tools_used.append("calculate")

        final_result = "Calculation completed successfully."

    elif "email" in goal_lower or "send" in goal_lower or "notify" in goal_lower:
        # Email scenario
        steps.append(AgentStep(
            step_number=1,
            thought="I need to find the recipient's contact information first.",
            action="search_database",
            action_input={"query": "users"},
            observation=simulate_tool_search_database("users")
        ))
        tools_used.append("search_database")

        if len(steps) < request.max_steps:
            steps.append(AgentStep(
                step_number=2,
                thought="Now I have the email addresses. I'll compose and send the email.",
                action="send_email",
                action_input={"to": "bob@example.com", "subject": "Notification", "body": "This is an automated notification..."},
                observation=simulate_tool_send_email("bob@example.com", "Notification", "This is an automated notification...")
            ))
            tools_used.append("send_email")

        final_result = "Email notification sent successfully."

    else:
        # Generic multi-step scenario
        steps.append(AgentStep(
            step_number=1,
            thought=f"Analyzing the goal: '{request.goal}'. Let me search for relevant information.",
            action="search_database",
            action_input={"query": request.goal[:50]},
            observation="Searched database. Found general information related to the query."
        ))
        tools_used.append("search_database")

        if len(steps) < request.max_steps:
            steps.append(AgentStep(
                step_number=2,
                thought="Let me perform any necessary calculations or analysis.",
                action="calculate",
                action_input={"expression": "analysis of " + request.goal[:20]},
                observation=simulate_tool_calculate("analysis")
            ))
            tools_used.append("calculate")

        final_result = f"Completed analysis for goal: {request.goal[:50]}..."

    # Ensure we don't exceed max_steps
    steps = steps[:request.max_steps]

    return AgentResponse(
        goal=request.goal,
        steps=steps,
        final_result=final_result,
        tools_used=list(set(tools_used)),
        success=True,
        total_steps=len(steps)
    )

# -----------------------------------------------------------------------------
# Helper Data Endpoints
# -----------------------------------------------------------------------------

@router.get("/sample_prompts")
async def get_sample_prompts():
    """Get sample prompts for different use cases."""
    return {
        "prompts": SAMPLE_PROMPTS,
        "categories": list(SAMPLE_PROMPTS.keys()),
        "description": "Sample prompts organized by use case for testing the /generate endpoint"
    }

@router.get("/example_distributions")
async def get_example_distributions():
    """Get example probability distributions for visualization."""
    return {
        "distributions": EXAMPLE_DISTRIBUTIONS,
        "vocabulary_sample": SAMPLE_VOCABULARY[:10],
        "description": "Pre-defined probability distributions for testing temperature and sampling endpoints"
    }

@router.get("/agent_database")
async def get_agent_database():
    """Get the simulated database that the agent can query."""
    return {
        "database": AGENT_DATABASE,
        "tables": list(AGENT_DATABASE.keys()),
        "description": "Simulated database for the AI agent to query",
        "available_tools": [
            {"name": "search_database", "description": "Query users, products, or orders"},
            {"name": "calculate", "description": "Perform mathematical calculations"},
            {"name": "send_email", "description": "Send simulated emails"}
        ]
    }

# =============================================================================
# Run with: uvicorn main:app --reload --port 8008
# =============================================================================
