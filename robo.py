# 1. INSTALAÇÃO DAS BIBLIOTECAS (se necessário)
# !pip install pandas numpy yfinance gymnasium "stable-baselines3[extra]" matplotlib

import pandas as pd
import numpy as np
import yfinance as yf
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import matplotlib.pyplot as plt

# ==============================================================================
# 2. ENGENHARIA DE FEATURES
# ==============================================================================
def create_features(df):
    df.columns = [col.lower() for col in df.columns]
    close = df['close']
    
    df['ema12'] = close.ewm(span=12, adjust=False).mean()
    df['ema26'] = close.ewm(span=26, adjust=False).mean()
    df['macd'] = df['ema12'] - df['ema26']
    df['macdsignal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macdhist'] = df['macd'] - df['macdsignal']
    
    delta = close.diff(1)
    gain = delta.clip(lower=0).ewm(span=14, adjust=False).mean()
    loss = abs(delta.clip(upper=0)).ewm(span=14, adjust=False).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    df['bbhigh'] = close.rolling(20).mean() + (close.rolling(20).std() * 2)
    df['bblow'] = close.rolling(20).mean() - (close.rolling(20).std() * 2)
    
    feature_cols = ['macd', 'macdhist', 'rsi', 'bbhigh', 'bblow']
    for col in feature_cols:
        df[col] = (df[col] - df[col].mean()) / df[col].std()
        
    state_columns = feature_cols + ['open', 'high', 'low', 'close', 'volume']
    
    return df[state_columns].dropna()

# ==============================================================================
# 3. AMBIENTE DE TRADING CUSTOMIZADO
# ==============================================================================
class TradingEnv(gym.Env):
    def __init__(self, df, initial_balance=100000):
        super(TradingEnv, self).__init__()
        
        self.df = df
        self.initial_balance = initial_balance
        self.current_step = 0
        
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(len(df.columns) + 2,), 
            dtype=np.float32
        )

    def _get_obs(self):
        obs = self.df.iloc[self.current_step].values
        obs = np.append(obs, [self.position, self.position_pnl])
        return obs.astype(np.float32)

    def reset(self, seed=None):
        super().reset(seed=seed)
        self.balance = self.initial_balance
        self.net_worth = self.initial_balance
        self.position = 0
        self.entry_price = 0
        self.position_pnl = 0
        self.current_step = 0
        self.history = []
        return self._get_obs(), {}

    def step(self, action):
        current_price = self.df['close'].iloc[self.current_step]
        reward = 0
        
        if action == 1 and self.position == 0:
            self.position = 1
            self.entry_price = current_price
        elif action == 2 and self.position == 1:
            pnl = (current_price - self.entry_price) / self.entry_price
            reward = pnl
            self.balance *= (1 + pnl)
            self.position = 0
            self.entry_price = 0

        if self.position == 1:
            self.position_pnl = (current_price - self.entry_price) / self.entry_price
        else:
            self.position_pnl = 0
            
        self.net_worth = self.balance * (1 + self.position_pnl)
        
        if self.position != 0:
            reward -= 0.0001
            
        self.current_step += 1
        terminated = self.current_step >= len(self.df) - 1
        truncated = False
        
        self.history.append({'step': self.current_step, 'net_worth': self.net_worth})
        
        return self._get_obs(), reward, terminated, truncated, {}

# ==============================================================================
# 4. EXECUÇÃO PRINCIPAL
# ==============================================================================
ticker = "PETR4.SA"
data = yf.download(ticker, start="2018-01-01", end="2024-01-01")

if isinstance(data.columns, pd.MultiIndex):
    data.columns = [col[0].lower() for col in data.columns]
else:
    data.columns = [col.lower() for col in data.columns]

data_featured = create_features(data)
train_size = int(len(data_featured) * 0.8)
train_data = data_featured[:train_size]
test_data = data_featured[train_size:]

train_env = DummyVecEnv([lambda: TradingEnv(train_data)])
test_env = DummyVecEnv([lambda: TradingEnv(test_data)])

print("--- INICIANDO TREINAMENTO DO AGENTE DE RL ---")
model = PPO('MlpPolicy', train_env, verbose=0, tensorboard_log="./ppo_trading_tensorboard/")
model.learn(total_timesteps=100000, progress_bar=True)
model.save("ppo_trading_agent_final")
print("--- TREINAMENTO CONCLUÍDO ---")

print("\n--- AVALIANDO O AGENTE EM DADOS NÃO VISTOS ---")
obs = test_env.reset()
is_done = False
while not is_done:
    action, _states = model.predict(obs, deterministic=True)
    # *** CORREÇÃO APLICADA AQUI ***
    # Desempacotar 4 valores, não 5
    obs, reward, done, info = test_env.step(action)
    # A variável 'done' agora é um array booleano, precisamos checar se algum ambiente terminou
    is_done = done[0]

history_df = pd.DataFrame(test_env.envs[0].history)
history_df.index = test_data.index[:len(history_df)]

returns = history_df['net_worth'].pct_change().fillna(0)
sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0
total_return = (history_df['net_worth'].iloc[-1] / history_df['net_worth'].iloc[0] - 1) * 100

print("\n--- MÉTRICAS DE DESEMPENHO NO PERÍODO DE TESTE ---")
print(f"Retorno Total da Estratégia: {total_return:.2f}%")
print(f"Sharpe Ratio (Anualizado): {sharpe_ratio:.2f}")

plt.style.use('dark_background')
fig, ax = plt.subplots(figsize=(15, 7))
ax.plot(history_df.index, history_df['net_worth'], color='cyan', label='Desempenho do Agente de RL')
buy_and_hold_data = test_data['close']
buy_and_hold = (buy_and_hold_data / buy_and_hold_data.iloc[0]) * test_env.envs[0].initial_balance
ax.plot(buy_and_hold.index, buy_and_hold.values, color='gray', linestyle='--', label='Buy & Hold')
ax.set_title(f'Desempenho do Agente de RL vs. Buy & Hold para {ticker}')
ax.set_ylabel('Patrimônio Líquido')
ax.set_xlabel('Data')
ax.legend()
plt.grid(True, alpha=0.2)
plt.show()
