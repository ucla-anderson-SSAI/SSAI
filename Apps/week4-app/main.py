import os
"""
Week 4: Reinforcement Learning - Dynamic Pricing Application
Multi-Armed Bandits + Q-Learning for Price Optimization
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Literal
import numpy as np
from enum import Enum

app = FastAPI(
    title="Week 4: Reinforcement Learning - Dynamic Pricing",
    description="Multi-Armed Bandits and Q-Learning for price optimization"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============== Constants ==============
PRICES = [5, 10, 15, 20, 25]
TRUE_CONVERSION_RATES = [0.50, 0.35, 0.22, 0.12, 0.05]
NUM_ARMS = len(PRICES)

# Q-Learning states
INVENTORY_LEVELS = ["high", "low"]
TIME_PERIODS = ["early", "late"]
STATES = [(inv, time) for inv in INVENTORY_LEVELS for time in TIME_PERIODS]
STATE_TO_IDX = {state: idx for idx, state in enumerate(STATES)}
NUM_STATES = len(STATES)
NUM_ACTIONS = NUM_ARMS


# ============== Request/Response Models ==============
class BanditStrategy(str, Enum):
    random = "random"
    epsilon_greedy = "epsilon_greedy"
    ucb = "ucb"


class BanditRunRequest(BaseModel):
    strategy: BanditStrategy
    n_customers: int = Field(default=1000, ge=100, le=5000)
    epsilon: float = Field(default=0.1, ge=0.0, le=0.5)
    confidence: float = Field(default=2.0, ge=0.5, le=5.0)


class BanditCompareRequest(BaseModel):
    n_customers: int = Field(default=1000, ge=100, le=5000)
    epsilon: float = Field(default=0.1, ge=0.0, le=0.5)
    confidence: float = Field(default=2.0, ge=0.5, le=5.0)


class BanditResult(BaseModel):
    strategy: str
    cumulative_revenue: List[float]
    final_revenue: float
    action_counts: List[int]
    estimated_conversion_rates: List[float]
    optimal_price: int
    prices: List[int] = PRICES
    true_conversion_rates: List[float] = TRUE_CONVERSION_RATES


class BanditCompareResult(BaseModel):
    results: Dict[str, BanditResult]
    prices: List[int] = PRICES
    true_conversion_rates: List[float] = TRUE_CONVERSION_RATES


class QLearningTrainRequest(BaseModel):
    learning_rate: float = Field(default=0.1, ge=0.01, le=0.5)
    discount_factor: float = Field(default=0.9, ge=0.0, le=0.99)
    episodes: int = Field(default=1000, ge=100, le=3000)
    epsilon: float = Field(default=0.1, ge=0.0, le=0.5)


class QLearningCompareRequest(BaseModel):
    param_to_compare: Literal["learning_rate", "discount"]
    episodes: int = Field(default=1000, ge=100, le=3000)
    base_learning_rate: float = Field(default=0.1, ge=0.01, le=0.5)
    base_discount: float = Field(default=0.9, ge=0.0, le=0.99)
    epsilon: float = Field(default=0.1, ge=0.0, le=0.5)


class QLearningResult(BaseModel):
    q_table: Dict[str, List[float]]
    policy: Dict[str, str]
    learning_curve: List[float]
    final_avg_reward: float
    episodes: int
    learning_rate: float
    discount_factor: float
    states: List[str]
    prices: List[int] = PRICES


class QLearningCompareResult(BaseModel):
    param_name: str
    param_values: List[float]
    results: List[QLearningResult]


# ============== Bandit Algorithms ==============
class MultiArmedBandit:
    def __init__(self, n_arms: int = NUM_ARMS):
        self.n_arms = n_arms
        self.reset()

    def reset(self):
        self.counts = np.zeros(self.n_arms)
        self.values = np.zeros(self.n_arms)
        self.total_reward = 0.0
        self.cumulative_rewards = []

    def update(self, arm: int, reward: float):
        self.counts[arm] += 1
        n = self.counts[arm]
        self.values[arm] = self.values[arm] + (reward - self.values[arm]) / n
        self.total_reward += reward
        self.cumulative_rewards.append(self.total_reward)


def random_strategy(bandit: MultiArmedBandit) -> int:
    return np.random.randint(bandit.n_arms)


def epsilon_greedy_strategy(bandit: MultiArmedBandit, epsilon: float) -> int:
    if np.random.random() < epsilon:
        return np.random.randint(bandit.n_arms)
    return int(np.argmax(bandit.values))


def ucb_strategy(bandit: MultiArmedBandit, t: int, confidence: float) -> int:
    # Handle arms that haven't been pulled yet
    for arm in range(bandit.n_arms):
        if bandit.counts[arm] == 0:
            return arm

    ucb_values = bandit.values + confidence * np.sqrt(np.log(t + 1) / bandit.counts)
    return int(np.argmax(ucb_values))


def simulate_purchase(arm: int) -> tuple:
    """Simulate a customer purchase decision. Returns (converted, revenue)."""
    price = PRICES[arm]
    conversion_rate = TRUE_CONVERSION_RATES[arm]
    converted = np.random.random() < conversion_rate
    revenue = price if converted else 0
    return converted, revenue


def run_bandit_simulation(
    strategy: str,
    n_customers: int,
    epsilon: float = 0.1,
    confidence: float = 2.0
) -> BanditResult:
    bandit = MultiArmedBandit()

    for t in range(n_customers):
        if strategy == "random":
            arm = random_strategy(bandit)
        elif strategy == "epsilon_greedy":
            arm = epsilon_greedy_strategy(bandit, epsilon)
        elif strategy == "ucb":
            arm = ucb_strategy(bandit, t, confidence)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        converted, revenue = simulate_purchase(arm)
        bandit.update(arm, revenue)

    # Compute estimated conversion rates from values and prices
    estimated_rates = [
        bandit.values[i] / PRICES[i] if PRICES[i] > 0 else 0
        for i in range(NUM_ARMS)
    ]

    optimal_idx = int(np.argmax(bandit.values))

    return BanditResult(
        strategy=strategy,
        cumulative_revenue=bandit.cumulative_rewards,
        final_revenue=bandit.total_reward,
        action_counts=[int(c) for c in bandit.counts],
        estimated_conversion_rates=estimated_rates,
        optimal_price=PRICES[optimal_idx]
    )


# ============== Q-Learning Environment ==============
class InventoryPricingEnv:
    """
    Environment for inventory-based dynamic pricing.
    States: (inventory_level, time_period)
    Actions: price indices (0-4 corresponding to $5-$25)
    """

    def __init__(self):
        self.state_idx = 0
        self.reset()

    def reset(self):
        # Start with random state
        self.state_idx = np.random.randint(NUM_STATES)
        return self.state_idx

    def get_conversion_rate(self, state_idx: int, action: int) -> float:
        """
        Conversion rate depends on state and price.
        - High inventory + early time: base rates
        - High inventory + late time: slightly higher (urgency to sell)
        - Low inventory + early time: can afford higher prices
        - Low inventory + late time: mixed strategy
        """
        base_rate = TRUE_CONVERSION_RATES[action]
        state = STATES[state_idx]
        inv, time = state

        if inv == "high" and time == "early":
            modifier = 1.0
        elif inv == "high" and time == "late":
            modifier = 1.2  # Urgency increases conversions
        elif inv == "low" and time == "early":
            modifier = 0.9  # Can afford to be selective
        else:  # low + late
            modifier = 1.1

        return min(1.0, base_rate * modifier)

    def step(self, action: int) -> tuple:
        """
        Take an action, return (next_state, reward, done).
        """
        price = PRICES[action]
        conversion_rate = self.get_conversion_rate(self.state_idx, action)
        converted = np.random.random() < conversion_rate
        reward = price if converted else 0

        # Transition to next state (simplified: random transitions)
        next_state_idx = np.random.randint(NUM_STATES)
        done = np.random.random() < 0.1  # 10% chance episode ends

        self.state_idx = next_state_idx
        return next_state_idx, reward, done


class QLearningAgent:
    def __init__(
        self,
        n_states: int = NUM_STATES,
        n_actions: int = NUM_ACTIONS,
        learning_rate: float = 0.1,
        discount_factor: float = 0.9,
        epsilon: float = 0.1
    ):
        self.n_states = n_states
        self.n_actions = n_actions
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.q_table = np.zeros((n_states, n_actions))

    def choose_action(self, state: int) -> int:
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        return int(np.argmax(self.q_table[state]))

    def update(self, state: int, action: int, reward: float, next_state: int):
        best_next = np.max(self.q_table[next_state])
        td_target = reward + self.gamma * best_next
        td_error = td_target - self.q_table[state, action]
        self.q_table[state, action] += self.lr * td_error

    def get_policy(self) -> Dict[str, str]:
        policy = {}
        for state_idx, state in enumerate(STATES):
            best_action = int(np.argmax(self.q_table[state_idx]))
            state_name = f"{state[0]}_{state[1]}"
            policy[state_name] = f"${PRICES[best_action]}"
        return policy

    def get_q_table_dict(self) -> Dict[str, List[float]]:
        q_dict = {}
        for state_idx, state in enumerate(STATES):
            state_name = f"{state[0]}_{state[1]}"
            q_dict[state_name] = [float(v) for v in self.q_table[state_idx]]
        return q_dict


def train_qlearning(
    learning_rate: float = 0.1,
    discount_factor: float = 0.9,
    episodes: int = 1000,
    epsilon: float = 0.1
) -> QLearningResult:
    env = InventoryPricingEnv()
    agent = QLearningAgent(
        learning_rate=learning_rate,
        discount_factor=discount_factor,
        epsilon=epsilon
    )

    learning_curve = []
    window_rewards = []

    for episode in range(episodes):
        state = env.reset()
        episode_reward = 0
        done = False
        steps = 0
        max_steps = 100

        while not done and steps < max_steps:
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)
            agent.update(state, action, reward, next_state)
            episode_reward += reward
            state = next_state
            steps += 1

        window_rewards.append(episode_reward)

        # Rolling average over last 100 episodes
        if len(window_rewards) > 100:
            window_rewards.pop(0)

        learning_curve.append(np.mean(window_rewards))

    return QLearningResult(
        q_table=agent.get_q_table_dict(),
        policy=agent.get_policy(),
        learning_curve=learning_curve,
        final_avg_reward=float(np.mean(window_rewards)),
        episodes=episodes,
        learning_rate=learning_rate,
        discount_factor=discount_factor,
        states=[f"{s[0]}_{s[1]}" for s in STATES]
    )


# ============== API Endpoints ==============
@app.get("/")
def health_check():
    return {
        "status": "healthy",
        "application": "Week 4: Reinforcement Learning - Dynamic Pricing",
        "components": {
            "bandit": "Multi-Armed Bandits for Price Optimization",
            "qlearning": "Q-Learning for Inventory Pricing"
        },
        "prices": PRICES,
        "true_conversion_rates": TRUE_CONVERSION_RATES
    }


@app.post("/bandit/run", response_model=BanditResult)
def run_bandit(request: BanditRunRequest):
    """Run a single bandit simulation with the specified strategy."""
    try:
        result = run_bandit_simulation(
            strategy=request.strategy.value,
            n_customers=request.n_customers,
            epsilon=request.epsilon,
            confidence=request.confidence
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/bandit/compare", response_model=BanditCompareResult)
def compare_bandits(request: BanditCompareRequest):
    """Compare all three bandit strategies."""
    try:
        results = {}
        for strategy in BanditStrategy:
            result = run_bandit_simulation(
                strategy=strategy.value,
                n_customers=request.n_customers,
                epsilon=request.epsilon,
                confidence=request.confidence
            )
            results[strategy.value] = result

        return BanditCompareResult(results=results)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/qlearning/train", response_model=QLearningResult)
def train_qlearning_endpoint(request: QLearningTrainRequest):
    """Train a Q-learning agent for inventory-based pricing."""
    try:
        result = train_qlearning(
            learning_rate=request.learning_rate,
            discount_factor=request.discount_factor,
            episodes=request.episodes,
            epsilon=request.epsilon
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/qlearning/compare_params", response_model=QLearningCompareResult)
def compare_qlearning_params(request: QLearningCompareRequest):
    """Compare Q-learning performance across different parameter values."""
    try:
        if request.param_to_compare == "learning_rate":
            param_values = [0.01, 0.05, 0.1, 0.2, 0.3, 0.5]
            results = [
                train_qlearning(
                    learning_rate=lr,
                    discount_factor=request.base_discount,
                    episodes=request.episodes,
                    epsilon=request.epsilon
                )
                for lr in param_values
            ]
        else:  # discount
            param_values = [0.0, 0.5, 0.7, 0.9, 0.95, 0.99]
            results = [
                train_qlearning(
                    learning_rate=request.base_learning_rate,
                    discount_factor=df,
                    episodes=request.episodes,
                    epsilon=request.epsilon
                )
                for df in param_values
            ]

        return QLearningCompareResult(
            param_name=request.param_to_compare,
            param_values=param_values,
            results=results
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


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
