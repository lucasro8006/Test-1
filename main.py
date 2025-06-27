"""
Script principal para o sistema de trading com Reinforcement Learning.

Este script implementa um pipeline completo para treinar e avaliar um agente
de trading usando Aprendizagem por Reforço (RL) com stable-baselines3 e um
ambiente Gymnasium customizado.
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from stable_baselines3.common.vec_env import DummyVecEnv

# Componentes do sistema
from src.features.feature_engineer import FeatureEngineer
from src.environments.trading_env import TradingEnv
from src.models.trading_agent import TradingAgent
from src.utils.data_utils import download_data, split_data
from src.utils.cross_validation import PurgedKFold, WalkForwardValidation, cross_validate_agent
from src.evaluation.evaluator import evaluate_agent
from src.evaluation.feature_importance import FeatureImportanceAnalyzer

# Tenta importar Optuna para otimização de hiperparâmetros
try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("Aviso: Optuna não está disponível. Use 'pip install optuna' para instalar.")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Sistema de Trading com RL')
    
    # Argumentos de dados
    parser.add_argument('--ticker', type=str, default='PETR4.SA',
                        help='Ticker do ativo')
    parser.add_argument('--start_date', type=str, default='2018-01-01',
                        help='Data de início (YYYY-MM-DD)')
    parser.add_argument('--end_date', type=str, default='2023-12-31',
                        help='Data de fim (YYYY-MM-DD)')
    
    # Argumentos de treinamento
    parser.add_argument('--timesteps', type=int, default=50000,
                        help='Número de passos de treinamento')
    parser.add_argument('--algorithm', type=str, default='PPO',
                        choices=['PPO', 'A2C', 'SAC', 'TD3'],
                        help='Algoritmo de RL')
    
    # Argumentos do ambiente
    parser.add_argument('--initial_balance', type=float, default=100000,
                        help='Saldo inicial')
    parser.add_argument('--commission', type=float, default=0.001,
                        help='Comissão por operação')
    parser.add_argument('--slippage', type=float, default=0.0005,
                        help='Slippage por operação')
    parser.add_argument('--reward_type', type=str, default='sharpe',
                        choices=['pnl', 'sharpe', 'calmar', 'sortino', 'information_ratio'],
                        help='Tipo de recompensa')
    
    # Argumentos de validação
    parser.add_argument('--cv', type=str, default='purged',
                        choices=['purged', 'walkforward', 'none'],
                        help='Tipo de validação cruzada')
    parser.add_argument('--n_splits', type=int, default=5,
                        help='Número de folds para validação cruzada')
    parser.add_argument('--pct_embargo', type=float, default=0.01,
                        help='Percentual de embargo para PurgedKFold')
    
    # Argumentos de otimização
    parser.add_argument('--optimize', action='store_true',
                        help='Realizar otimização de hiperparâmetros')
    parser.add_argument('--n_trials', type=int, default=50,
                        help='Número de trials para otimização')
    
    # Argumentos de saída
    parser.add_argument('--output_dir', type=str, default='results',
                        help='Diretório para salvar resultados')
    parser.add_argument('--model_name', type=str, default=None,
                        help='Nome do modelo (default: ticker_algorithm_timestamp)')
    parser.add_argument('--model_path', type=str, default=None,
                        help='Caminho para carregar modelo treinado')
    parser.add_argument('--save_model', action='store_true',
                        help='Salvar modelo treinado')
    
    # Argumentos de análise
    parser.add_argument('--analyze_features', action='store_true',
                        help='Realizar análise de importância das features')
    parser.add_argument('--n_samples', type=int, default=100,
                        help='Número de amostras para análise SHAP')
    
    return parser.parse_args()


def optimize_hyperparameters(train_data, val_data, env_params, n_trials=50):
    """
    Otimiza hiperparâmetros usando Optuna.
    
    Args:
        train_data (pd.DataFrame): Dados de treino
        val_data (pd.DataFrame): Dados de validação
        env_params (dict): Parâmetros do ambiente
        n_trials (int): Número de trials
        
    Returns:
        dict: Melhores hiperparâmetros
    """
    if not OPTUNA_AVAILABLE:
        print("Aviso: Optuna não está disponível. Usando hiperparâmetros padrão.")
        return get_default_model_params()
        
    def objective(trial):
        # Parâmetros do modelo
        model_params = {
            'policy': 'MlpPolicy',
            'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True),
            'n_steps': trial.suggest_int('n_steps', 16, 2048, log=True),
            'batch_size': trial.suggest_int('batch_size', 8, 256, log=True),
            'gamma': trial.suggest_float('gamma', 0.9, 0.9999),
            'ent_coef': trial.suggest_float('ent_coef', 0.0, 0.1),
            'policy_kwargs': {
                'net_arch': trial.suggest_categorical('net_arch', [
                    [64, 64],
                    [128, 64],
                    [256, 128, 64],
                    [128, 128, 64, 32]
                ])
            }
        }
        
        # Cria ambiente de treino
        train_env = DummyVecEnv([
            lambda: TradingEnv(train_data, **env_params)
        ])
        
        # Cria ambiente de validação
        val_env = TradingEnv(val_data, **env_params)
        
        # Cria e treina o agente
        agent = TradingAgent(model_params)
        agent.train(train_env, total_timesteps=10000)  # Treinamento rápido para otimização
        
        # Avalia no conjunto de validação
        val_env_vec = DummyVecEnv([lambda: val_env])
        metrics = agent.evaluate(val_env_vec)
        
        # Retorna métrica a ser maximizada
        if env_params['reward_type'] == 'sharpe':
            return metrics.get('sharpe_ratio', 0)
        elif env_params['reward_type'] == 'calmar':
            return metrics.get('calmar_ratio', 0)
        elif env_params['reward_type'] == 'sortino':
            return metrics.get('sortino_ratio', 0)
        else:
            return metrics.get('total_return', 0)
            
    # Cria estudo Optuna
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)
    
    # Obtém melhores parâmetros
    best_params = study.best_params
    
    # Constrói dicionário de parâmetros
    model_params = {
        'policy': 'MlpPolicy',
        'learning_rate': best_params['learning_rate'],
        'n_steps': best_params['n_steps'],
        'batch_size': best_params['batch_size'],
        'gamma': best_params['gamma'],
        'ent_coef': best_params['ent_coef'],
        'policy_kwargs': {
            'net_arch': best_params['net_arch']
        }
    }
    
    return model_params


def get_default_model_params():
    """Retorna parâmetros padrão do modelo."""
    return {
        'policy': 'MlpPolicy',
        'learning_rate': 0.0003,
        'n_steps': 128,
        'batch_size': 64,
        'gamma': 0.99,
        'ent_coef': 0.01,
        'policy_kwargs': {
            'net_arch': [128, 64]
        }
    }


def main():
    """Função principal."""
    # Parse argumentos
    args = parse_args()
    
    # Cria diretório de saída
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    # Define nome do modelo
    if args.model_name is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        args.model_name = f"{args.ticker.replace('.', '_')}_{args.algorithm}_{timestamp}"
        
    # Download de dados
    print(f"Baixando dados para {args.ticker}...")
    data = download_data(args.ticker, args.start_date, args.end_date)
    print(f"Dados baixados: {len(data)} registros de {data.index[0].date()} a {data.index[-1].date()}")
    
    # Preparação de features
    print("Preparando features...")
    fe = FeatureEngineer()
    processed_data = fe.transform(data)
    
    # Divisão em treino, validação e teste
    train_data, val_data, test_data = split_data(
        processed_data, 
        train_size=0.7, 
        val_size=0.15, 
        test_size=0.15
    )
    print(f"Dados divididos: Treino={len(train_data)}, Validação={len(val_data)}, Teste={len(test_data)}")
    
    # Parâmetros do ambiente
    env_params = {
        'initial_balance': args.initial_balance,
        'commission_pct': args.commission,
        'slippage_pct': args.slippage,
        'reward_type': args.reward_type,
        'window_size': 20,
        'max_drawdown_penalty': 0.5,
        'transaction_cost_model': 'percentage',
        'market_impact_model': 'linear',
        'risk_free_rate': 0.03,
        'enable_fractional': True,
        'max_position_size': 1.0
    }
    
    # Carrega modelo existente ou treina um novo
    if args.model_path:
        print(f"\n--- CARREGANDO MODELO DE {args.model_path} ---")
        agent = TradingAgent({})
        agent.load(args.model_path)
    else:
        # Otimização de hiperparâmetros
        if args.optimize and len(val_data) > 0:
            print("\n--- INICIANDO OTIMIZAÇÃO DE HIPERPARÂMETROS ---")
            model_params = optimize_hyperparameters(
                train_data, 
                val_data, 
                env_params, 
                n_trials=args.n_trials
            )
            print("--- OTIMIZAÇÃO CONCLUÍDA ---")
            print(f"Melhores parâmetros: {model_params}")
        else:
            model_params = get_default_model_params()
            
        # Validação cruzada
        if args.cv != 'none':
            print("\n--- INICIANDO VALIDAÇÃO CRUZADA ---")
            
            # Combina treino e validação para CV
            train_val_data = pd.concat([train_data, val_data])
            
            if args.cv == 'purged':
                cv = PurgedKFold(n_splits=args.n_splits, pct_embargo=args.pct_embargo)
            else:  # walkforward
                cv = WalkForwardValidation(n_splits=args.n_splits, train_size=0.7)
                
            # Realiza validação cruzada
            cv_results = cross_validate_agent(
                TradingAgent,
                TradingEnv,
                train_val_data,
                cv,
                model_params,
                env_params,
                total_timesteps=args.timesteps // 2,  # Menos passos para CV
                metric=args.reward_type
            )
            
            print("--- VALIDAÇÃO CRUZADA CONCLUÍDA ---")
            print(f"Média de {args.reward_type} no treino: {cv_results.get(f'mean_train_{args.reward_type}', 'N/A')}")
            print(f"Média de {args.reward_type} no teste: {cv_results.get(f'mean_test_{args.reward_type}', 'N/A')}")
            
            # Salva resultados da CV
            cv_results_df = pd.DataFrame({
                'metric': list(cv_results.keys()),
                'value': list(cv_results.values())
            })
            cv_results_df.to_csv(f"{args.output_dir}/{args.model_name}_cv_results.csv", index=False)
        
        # Treinamento do modelo final
        print("\n--- INICIANDO TREINAMENTO DO AGENTE DE RL ---")
        
        # Cria ambiente de treino
        train_env = DummyVecEnv([
            lambda: TradingEnv(pd.concat([train_data, val_data]), **env_params)
        ])
        
        # Cria e treina o agente
        agent = TradingAgent(model_params, algorithm=args.algorithm)
        agent.train(
            train_env, 
            total_timesteps=args.timesteps,
            progress_bar=True,
            save_path=f"{args.output_dir}/{args.model_name}"
        )
        
        print("--- TREINAMENTO CONCLUÍDO ---")
        
        # Salva o modelo
        if args.save_model:
            model_path = f"{args.output_dir}/{args.model_name}"
            agent.save(model_path)
            print(f"Modelo salvo em {model_path}")
    
    # Avaliação do modelo
    print("\n--- AVALIANDO O AGENTE EM DADOS NÃO VISTOS ---")
    test_env = TradingEnv(test_data, **env_params)
    history_df, metrics = evaluate_agent(agent, test_env, test_data)
    
    # Salva métricas
    metrics_df = pd.DataFrame({
        'metric': list(metrics.keys()),
        'value': list(metrics.values())
    })
    metrics_df.to_csv(f"{args.output_dir}/{args.model_name}_metrics.csv", index=False)
    
    # Salva histórico
    if not history_df.empty:
        history_df.to_csv(f"{args.output_dir}/{args.model_name}_history.csv")
    
    # Análise de importância das features
    if args.analyze_features:
        print("\n--- ANALISANDO IMPORTÂNCIA DAS FEATURES ---")
        
        # Cria analisador
        analyzer = FeatureImportanceAnalyzer()
        
        # Analisa importância
        feature_importance = analyzer.analyze_shap(
            agent.model,
            test_data,
            feature_names=fe.feature_columns,
            n_samples=args.n_samples
        )
        
        if feature_importance is not None:
            # Salva importância
            feature_importance.to_csv(f"{args.output_dir}/{args.model_name}_feature_importance.csv", index=False)
            
            # Plota importância
            fig = analyzer.plot_feature_importance(top_n=20)
            if fig is not None:
                fig.savefig(f"{args.output_dir}/{args.model_name}_feature_importance.png")
                
            # Plota resumo SHAP
            fig = analyzer.plot_shap_summary()
            if fig is not None:
                fig.savefig(f"{args.output_dir}/{args.model_name}_shap_summary.png")
                
            print(f"Análise de importância salva em {args.output_dir}/{args.model_name}_feature_importance.csv")
    
    print("\nProcesso concluído com sucesso!")


if __name__ == "__main__":
    main()