"""
Script final para testar o agente com incentivos para operações
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
START_DATE = '2020-01-01'  # Mais dados para treinamento
END_DATE = '2023-06-01'
COMMISSION = 0.0001  # Comissão muito baixa para incentivar operações
SLIPPAGE = 0.0001  # Slippage muito baixo
REWARD_TYPE = 'pnl'  # PnL direto para feedback mais claro
TIMESTEPS = 30000  # Passos suficientes para aprendizado

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
    'learning_rate': 0.0003,  # Taxa de aprendizado padrão
    'n_steps': 128,  # Passos maiores para capturar mais contexto
    'batch_size': 64,  # Batch maior para melhor generalização
    'gamma': 0.99,
    'ent_coef': 0.1,  # Muito mais exploração
    'policy_kwargs': {'net_arch': [64, 32]},  # Rede neural menor e mais simples
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
    else:
        print("\nAnálise das operações:")
        buys = history_df[history_df['position'] > history_df['position'].shift(1).fillna(0)]
        sells = history_df[history_df['position'] < history_df['position'].shift(1).fillna(0)]
        
        print(f"Número de compras: {len(buys)}")
        print(f"Número de vendas: {len(sells)}")
        
        if len(buys) > 0 and len(sells) > 0:
            # Calcula o lucro médio por operação
            buy_prices = buys['price'].values
            sell_prices = sells['price'].values
            
            # Considera apenas operações completas (compra seguida de venda)
            min_ops = min(len(buy_prices), len(sell_prices))
            if min_ops > 0:
                profits = []
                for i in range(min_ops):
                    profit_pct = (sell_prices[i] / buy_prices[i] - 1) * 100
                    profits.append(profit_pct)
                
                avg_profit = np.mean(profits)
                win_rate = (np.array(profits) > 0).mean() * 100
                
                print(f"Lucro médio por operação: {avg_profit:.2f}%")
                print(f"Taxa de acerto: {win_rate:.2f}%")

# Salva o modelo
model_path = f"models/{TICKER.replace('.', '_')}_model"
agent.save(model_path)
print(f"\nModelo salvo em {model_path}")

print("\nTeste final concluído com sucesso!")