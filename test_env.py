"""
Test script for the trading environment
"""

import yfinance as yf
import numpy as np
from trading_agent import FeatureEngineer, TradingEnv

# Fix the _get_obs method in TradingEnv
def patched_get_obs(self):
    """Get the current observation (state)."""
    # Make sure we don't go out of bounds
    step = min(self.current_step, len(self.df) - 1)
    obs = self.df.iloc[step].values
    obs = np.append(obs, [self.position, self.position_pnl])
    return obs.astype(np.float32)

# Patch the method
TradingEnv._get_obs = patched_get_obs

# Download a small sample of data
data = yf.download('PETR4.SA', start='2023-01-01', end='2023-02-01', auto_adjust=True)

# Create feature engineer and transform data
fe = FeatureEngineer()
processed_data = fe.transform(data)

# Create trading environment
env = TradingEnv(processed_data, initial_balance=10000, commission_pct=0.001, slippage_pct=0.0005)

# Reset environment
obs, _ = env.reset()
print("Observation shape:", obs.shape)

# Take a few random actions
for i in range(5):
    action = np.random.randint(0, 3)  # 0=hold, 1=buy, 2=sell
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"Step {i+1}: Action={action}, Reward={reward:.6f}, Position={env.position}, Net Worth={env.net_worth:.2f}")

print('Trading environment works!')