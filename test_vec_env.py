"""
Test for the TradingEnv class with DummyVecEnv wrapper
"""

import pandas as pd
import numpy as np
from trading_agent import TradingEnv
from stable_baselines3.common.vec_env import DummyVecEnv

# Create a simple DataFrame for testing
df = pd.DataFrame({
    'macd': [0.1, 0.2, 0.3, 0.4, 0.5],
    'macdhist': [0.01, 0.02, 0.03, 0.04, 0.05],
    'rsi': [30, 40, 50, 60, 70],
    'bbhigh': [105, 106, 107, 108, 109],
    'bblow': [95, 96, 97, 98, 99],
    'open': [100, 101, 102, 103, 104],
    'high': [105, 106, 107, 108, 109],
    'low': [95, 96, 97, 98, 99],
    'close': [100, 102, 103, 101, 105],
    'volume': [1000, 1100, 1200, 1300, 1400]
})

# Create vectorized environment
vec_env = DummyVecEnv([lambda: TradingEnv(df, initial_balance=10000, commission_pct=0.001, slippage_pct=0.0005)])

# Reset environment
obs = vec_env.reset()
print("Initial observation shape:", obs.shape)

# Take a few actions
actions = [0, 1, 0, 2, 0]  # hold, buy, hold, sell, hold
for i, action in enumerate(actions):
    obs, reward, done, info = vec_env.step([action])
    print(f"Step {i+1}: Action={action}, Reward={reward[0]:.6f}, Done={done[0]}")

# Print history
print("\nHistory:")
for entry in vec_env.envs[0].history:
    print(entry)

print("\nTest completed successfully!")