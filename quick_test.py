"""
Versão rápida para testar o pipeline completo
"""

import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from trading_agent import FeatureEngineer, TradingEnv, TradingAgent, evaluate_agent
from stable_baselines3.common.vec_env import DummyVecEnv

# Configurações
TICKER = 'PETR4.SA'
START_DATE = '2023-01-01'
END_DATE = '2023-06-01'
COMMISSION = 0.001
SLIPPAGE = 0.0005
REWARD_TYPE = 'pnl'

# Download de dados
print(f"Baixando dados para {TICKER}...")
data = yf.download(TICKER, start=START_DATE, end=END_DATE, auto_adjust=True)

# Preparação de dados
print("Preparando dados...")
fe = FeatureEngineer()
processed_data = fe.transform(data)

# Divisão em treino e teste
train_size = int(len(processed_data) * 0.8)
train_data = processed_data[:train_size]
test_data = processed_data[train_size:]

print(f"Dados divididos: Treino={len(train_data)}, Teste={len(test_data)}")

# Criação do ambiente de treino
train_env = DummyVecEnv([
    lambda: TradingEnv(
        train_data, 
        commission_pct=COMMISSION, 
        slippage_pct=SLIPPAGE,
        reward_type=REWARD_TYPE
    )
])

# Parâmetros do modelo
model_params = {
    'policy': 'MlpPolicy',
    'learning_rate': 0.0003,
    'n_steps': 64,
    'batch_size': 32,
    'gamma': 0.99,
    'policy_kwargs': {'net_arch': [64, 64]},
    'tensorboard_log': "./ppo_trading_tensorboard/"
}

# Criação e treinamento do agente
print("\n--- INICIANDO TREINAMENTO DO AGENTE DE RL ---")
agent = TradingAgent(model_params)
agent.train(train_env, total_timesteps=1000, progress_bar=True)
print("--- TREINAMENTO CONCLUÍDO ---")

# Avaliação do agente
print("\n--- AVALIANDO O AGENTE EM DADOS NÃO VISTOS ---")
history_df = evaluate_agent(
    agent, 
    test_data, 
    commission_pct=COMMISSION, 
    slippage_pct=SLIPPAGE,
    reward_type=REWARD_TYPE
)

print("\nTeste rápido concluído com sucesso!")