"""
Versão rápida para testar o pipeline completo com a nova estrutura
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Adiciona o diretório raiz ao path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Importa componentes do sistema
from src.features.feature_engineer import FeatureEngineer
from src.environments.trading_env import TradingEnv
from src.models.trading_agent import TradingAgent
from src.utils.data_utils import download_data, split_data
from src.evaluation.evaluator import evaluate_agent

# Configurações
TICKER = 'PETR4.SA'
START_DATE = '2022-01-01'  # Mais dados para treinamento
END_DATE = '2023-06-01'
COMMISSION = 0.001
SLIPPAGE = 0.0005
REWARD_TYPE = 'sharpe'  # Sharpe ratio tende a incentivar mais operações
TIMESTEPS = 10000  # Mais passos para garantir operações

# Download de dados
data = download_data(TICKER, START_DATE, END_DATE)

# Preparação de features
print("Preparando features...")
fe = FeatureEngineer()
processed_data = fe.transform(data)

# Divisão em treino e teste
train_data, val_data, test_data = split_data(processed_data, train_size=0.7, val_size=0.0, test_size=0.3)

# Parâmetros do ambiente
env_params = {
    'commission_pct': COMMISSION,
    'slippage_pct': SLIPPAGE,
    'reward_type': REWARD_TYPE
}

# Parâmetros do modelo
model_params = {
    'policy': 'MlpPolicy',
    'learning_rate': 0.0003,
    'n_steps': 128,  # Passos maiores para capturar mais contexto
    'batch_size': 64,  # Batch maior para melhor generalização
    'gamma': 0.99,
    'ent_coef': 0.01,  # Incentiva exploração
    'policy_kwargs': {'net_arch': [128, 64]},  # Rede neural maior
    'tensorboard_log': "./ppo_trading_tensorboard/"
}

# Criação do ambiente de treino
from stable_baselines3.common.vec_env import DummyVecEnv
train_env = DummyVecEnv([
    lambda: TradingEnv(train_data, **env_params)
])

# Criação e treinamento do agente
print("\n--- INICIANDO TREINAMENTO DO AGENTE DE RL ---")
agent = TradingAgent(model_params)
agent.train(train_env, total_timesteps=TIMESTEPS, progress_bar=True)
print("--- TREINAMENTO CONCLUÍDO ---")

# Avaliação do agente
print("\n--- AVALIANDO O AGENTE EM DADOS NÃO VISTOS ---")
test_env = TradingEnv(test_data, **env_params)
history_df, metrics = evaluate_agent(agent, test_env, test_data)

# Verifica se o agente fez operações
if 'position' in history_df.columns:
    position_changes = history_df['position'].diff().fillna(0)
    n_trades = (position_changes != 0).sum()
    print(f"\nNúmero de operações realizadas: {n_trades}")
    
    if n_trades == 0:
        print("\nAVISO: O agente não realizou nenhuma operação!")
        print("Possíveis causas:")
        print("1. Treinamento insuficiente (aumente TIMESTEPS)")
        print("2. Recompensa inadequada (experimente outros tipos de recompensa)")
        print("3. Dados insuficientes ou sem padrões claros")
        print("4. Hiperparâmetros inadequados")

print("\nTeste rápido concluído com sucesso!")