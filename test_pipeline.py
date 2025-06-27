"""
Test script for the complete trading pipeline
"""

import yfinance as yf
import numpy as np
from trading_agent import FeatureEngineer, TradingEnv, TradingAgent, evaluate_agent
from stable_baselines3.common.vec_env import DummyVecEnv

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
print("Downloading data...")
data = yf.download('PETR4.SA', start='2023-01-01', end='2023-06-01', auto_adjust=True)

# Create feature engineer and transform data
print("Processing features...")
fe = FeatureEngineer()
processed_data = fe.transform(data)

# Split data into train and test
train_size = int(len(processed_data) * 0.8)
train_data = processed_data[:train_size]
test_data = processed_data[train_size:]

# Create environments
print("Creating environments...")
train_env = DummyVecEnv([
    lambda: TradingEnv(
        train_data, 
        commission_pct=0.001, 
        slippage_pct=0.0005,
        reward_type='pnl'
    )
])

# Create and train agent (with minimal training for testing)
print("Training agent (minimal training for testing)...")
model_params = {
    'policy': 'MlpPolicy',
    'learning_rate': 0.0003,
    'n_steps': 64,
    'batch_size': 32,
    'gamma': 0.99,
    'policy_kwargs': {'net_arch': [64, 64]},
    'tensorboard_log': "./ppo_trading_tensorboard/"
}

agent = TradingAgent(model_params)
agent.train(train_env, total_timesteps=1000, progress_bar=True)

# Evaluate on test data
print("\n--- EVALUATING AGENT ON TEST DATA ---")
history_df = evaluate_agent(
    agent, 
    test_data, 
    commission_pct=0.001, 
    slippage_pct=0.0005,
    reward_type='pnl'
)

print("Pipeline test completed successfully!")