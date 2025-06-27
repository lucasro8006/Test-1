"""
Test script for the complete trading pipeline
"""

import yfinance as yf
import numpy as np
from trading_agent import FeatureEngineer, TradingEnv, TradingAgent
from stable_baselines3.common.vec_env import DummyVecEnv
import pandas as pd
import matplotlib.pyplot as plt

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

# Create a direct environment (not vectorized) for evaluation
test_env_direct = TradingEnv(
    test_data, 
    commission_pct=0.001, 
    slippage_pct=0.0005,
    reward_type='pnl'
)

# Reset environment
obs, _ = test_env_direct.reset()
done = False

# Run agent on test data
while not done:
    # Reshape observation for the model
    obs_reshaped = np.array([obs])
    action, _ = agent.model.predict(obs_reshaped, deterministic=True)
    obs, reward, terminated, truncated, _ = test_env_direct.step(action[0])
    done = terminated or truncated

# Get history
history = test_env_direct.history
print(f"History length: {len(history)}")
print(f"First history entry: {history[0] if history else 'No history'}")

history_df = pd.DataFrame(history)
if len(history_df) > 0:
    history_df.index = test_data.index[:len(history_df)]
    
    # Calculate metrics
    returns = history_df['net_worth'].pct_change().fillna(0)
    sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0
    total_return = (history_df['net_worth'].iloc[-1] / history_df['net_worth'].iloc[0] - 1) * 100
    
    # Calculate drawdown
    peak = history_df['net_worth'].cummax()
    drawdown = (history_df['net_worth'] - peak) / peak
    max_drawdown = drawdown.min() * 100
    
    print("\n--- MÉTRICAS DE DESEMPENHO NO PERÍODO DE TESTE ---")
    print(f"Retorno Total da Estratégia: {total_return:.2f}%")
    print(f"Sharpe Ratio (Anualizado): {sharpe_ratio:.2f}")
    print(f"Drawdown Máximo: {max_drawdown:.2f}%")
    
    # Plot performance
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(15, 7))
    ax.plot(history_df.index, history_df['net_worth'], color='cyan', label='Desempenho do Agente de RL')
    
    # Plot buy and hold
    buy_and_hold_data = test_data['close']
    buy_and_hold = (buy_and_hold_data / buy_and_hold_data.iloc[0]) * test_env_direct.initial_balance
    ax.plot(buy_and_hold.index, buy_and_hold.values, color='gray', linestyle='--', label='Buy & Hold')
    
    # Plot buy/sell markers
    buys = history_df[history_df['position'] > history_df['position'].shift(1).fillna(0)]
    sells = history_df[history_df['position'] < history_df['position'].shift(1).fillna(0)]
    
    if not buys.empty:
        ax.scatter(buys.index, buys['net_worth'], color='green', marker='^', s=100, label='Compra')
    if not sells.empty:
        ax.scatter(sells.index, sells['net_worth'], color='red', marker='v', s=100, label='Venda')
    
    ax.set_title(f'Desempenho do Agente de RL vs. Buy & Hold')
    ax.set_ylabel('Patrimônio Líquido')
    ax.set_xlabel('Data')
    ax.legend()
    plt.grid(True, alpha=0.2)
    plt.savefig('performance.png')
    print("Performance plot saved to 'performance.png'")
else:
    print("No history data available for evaluation")

print("Pipeline test completed successfully!")