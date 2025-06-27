"""
Simple test for the TradingEnv class
"""

import pandas as pd
import numpy as np
from trading_agent import TradingEnv

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

# Create environment
env = TradingEnv(df, initial_balance=10000, commission_pct=0.001, slippage_pct=0.0005)

# Reset environment
obs, _ = env.reset()
print("Initial observation shape:", obs.shape)

# Take a few actions
actions = [0, 1, 0, 2, 0]  # hold, buy, hold, sell, hold
for i, action in enumerate(actions):
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"Step {i+1}: Action={action}, Reward={reward:.6f}, Position={env.position}, Net Worth={env.net_worth:.2f}")

# Print history
print("\nHistory:")
for entry in env.history:
    print(entry)

print("\nTest completed successfully!")