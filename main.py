"""
Script principal para o sistema de trading com RL
"""

import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3.common.vec_env import DummyVecEnv

# Importa componentes do sistema
from src.features.feature_engineer import FeatureEngineer
from src.environments.trading_env import TradingEnv
from src.models.trading_agent import TradingAgent
from src.utils.data_utils import download_data, split_data
from src.utils.optimization import optimize_agent_hyperparams
from src.evaluation.evaluator import evaluate_agent, analyze_feature_importance


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Sistema de Trading com RL')
    
    # Argumentos de dados
    parser.add_argument('--ticker', type=str, default='PETR4.SA',
                        help='Símbolo do ativo')
    parser.add_argument('--start_date', type=str, default='2018-01-01',
                        help='Data de início (formato: YYYY-MM-DD)')
    parser.add_argument('--end_date', type=str, default='2023-12-31',
                        help='Data de fim (formato: YYYY-MM-DD)')
    
    # Argumentos de treinamento
    parser.add_argument('--optimize', action='store_true',
                        help='Otimizar hiperparâmetros')
    parser.add_argument('--n_trials', type=int, default=20,
                        help='Número de trials para otimização')
    parser.add_argument('--timesteps', type=int, default=100000,
                        help='Número de passos de treinamento')
    
    # Argumentos de avaliação
    parser.add_argument('--model_path', type=str, default=None,
                        help='Caminho para carregar modelo treinado')
    parser.add_argument('--save_model', action='store_true',
                        help='Salvar modelo treinado')
    parser.add_argument('--analyze_features', action='store_true',
                        help='Analisar importância das features')
    
    return parser.parse_args()


def main():
    """Função principal."""
    # Parse argumentos
    args = parse_args()
    
    # Cria diretórios necessários
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    # Download de dados
    data = download_data(args.ticker, args.start_date, args.end_date)
    
    # Preparação de features
    print("\n--- PREPARANDO FEATURES ---")
    fe = FeatureEngineer()
    processed_data = fe.transform(data)
    
    # Divisão dos dados
    print("\n--- DIVIDINDO DADOS ---")
    train_data, val_data, test_data = split_data(processed_data)
    
    # Carrega modelo existente ou treina um novo
    if args.model_path:
        print(f"\n--- CARREGANDO MODELO DE {args.model_path} ---")
        agent = TradingAgent({})
        agent.load(args.model_path)
    else:
        # Otimização de hiperparâmetros ou treinamento com parâmetros padrão
        if args.optimize:
            print("\n--- OTIMIZANDO HIPERPARÂMETROS ---")
            best_params = optimize_agent_hyperparams(
                TradingAgent,
                TradingEnv,
                train_data,
                val_data,
                n_trials=args.n_trials,
                total_timesteps=args.timesteps // 10,  # Menos passos para otimização
                progress_bar=True
            )
            
            # Extrai parâmetros
            env_params = best_params['env_params']
            agent_params = best_params['agent_params']
        else:
            # Parâmetros padrão
            env_params = {
                'commission_pct': 0.001,
                'slippage_pct': 0.0005,
                'reward_type': 'sharpe'
            }
            
            agent_params = {
                'policy': 'MlpPolicy',
                'learning_rate': 0.0003,
                'n_steps': 64,
                'batch_size': 32,
                'gamma': 0.99,
                'policy_kwargs': {'net_arch': [64, 64]},
                'tensorboard_log': "./ppo_trading_tensorboard/"
            }
        
        # Cria ambiente de treinamento
        train_env = DummyVecEnv([
            lambda: TradingEnv(
                train_data, 
                **env_params
            )
        ])
        
        # Treina o agente
        print("\n--- TREINANDO AGENTE ---")
        agent = TradingAgent(agent_params)
        agent.train(train_env, total_timesteps=args.timesteps, progress_bar=True)
        
        # Salva o modelo
        if args.save_model:
            model_path = f"models/{args.ticker.replace('.', '_')}_model"
            agent.save(model_path)
            print(f"Modelo salvo em {model_path}")
    
    # Avalia o agente nos dados de teste
    print("\n--- AVALIANDO AGENTE EM DADOS DE TESTE ---")
    test_env = TradingEnv(
        test_data,
        commission_pct=0.001,
        slippage_pct=0.0005,
        reward_type='pnl'
    )
    
    history_df, metrics = evaluate_agent(agent, test_env, test_data)
    
    # Analisa importância das features
    if args.analyze_features and len(fe.feature_columns) > 0:
        print("\n--- ANALISANDO IMPORTÂNCIA DAS FEATURES ---")
        analyze_feature_importance(agent, test_env, fe.feature_columns)
    
    print("\nProcesso concluído com sucesso!")


if __name__ == "__main__":
    main()