"""
Utilitários para otimização de hiperparâmetros
"""

import optuna
import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv


def optimize_agent_hyperparams(
    agent_class, 
    env_class, 
    train_data, 
    val_data, 
    n_trials=20, 
    total_timesteps=20000,
    progress_bar=False
):
    """
    Otimiza os hiperparâmetros do agente usando Optuna.
    
    Args:
        agent_class: Classe do agente
        env_class: Classe do ambiente
        train_data (pd.DataFrame): Dados de treino
        val_data (pd.DataFrame): Dados de validação
        n_trials (int): Número de trials para otimização
        total_timesteps (int): Número de passos de treinamento
        progress_bar (bool): Exibir barra de progresso
        
    Returns:
        dict: Melhores hiperparâmetros encontrados
    """
    def objective(trial):
        # Hiperparâmetros do ambiente
        env_params = {
            'commission_pct': trial.suggest_float('commission', 0.0001, 0.005, log=True),
            'slippage_pct': trial.suggest_float('slippage', 0.0001, 0.005, log=True),
            'reward_type': trial.suggest_categorical('reward_type', ['pnl', 'sharpe', 'calmar'])
        }
        
        # Hiperparâmetros do agente
        agent_params = {
            'policy': 'MlpPolicy',
            'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True),
            'n_steps': trial.suggest_int('n_steps', 16, 2048, log=True),
            'batch_size': trial.suggest_int('batch_size', 8, 256, log=True),
            'gamma': trial.suggest_float('gamma', 0.9, 0.9999),
            'policy_kwargs': {
                'net_arch': eval(trial.suggest_categorical(
                    'net_arch', 
                    ['[64, 64]', '[128, 64]', '[256, 128, 64]', '[64, 32, 16]']
                ))
            },
            'tensorboard_log': "./ppo_trading_tensorboard/"
        }
        
        # Cria o ambiente de treino
        train_env = DummyVecEnv([
            lambda: env_class(train_data, **env_params)
        ])
        
        # Cria e treina o agente
        agent = agent_class(agent_params)
        agent.train(train_env, total_timesteps=total_timesteps, progress_bar=progress_bar)
        
        # Cria o ambiente de validação
        val_env = env_class(val_data, **env_params)
        
        # Avalia o agente no ambiente de validação
        obs, _ = val_env.reset()
        done = False
        
        while not done:
            # Reshape da observação para o modelo
            obs_reshaped = np.array([obs])
            action, _ = agent.predict(obs_reshaped, deterministic=True)
            obs, reward, terminated, truncated, _ = val_env.step(action[0])
            done = terminated or truncated
        
        # Obtém o histórico
        history = val_env.history
        
        # Calcula o retorno total
        if history:
            initial_net_worth = history[0]['net_worth']
            final_net_worth = history[-1]['net_worth']
            total_return = (final_net_worth / initial_net_worth - 1) * 100
        else:
            total_return = 0
        
        # Calcula o Sharpe Ratio
        if hasattr(val_env, 'returns_history') and len(val_env.returns_history) > 1:
            returns = np.array(val_env.returns_history)
            sharpe_ratio = returns.mean() / (returns.std() + 1e-9) * np.sqrt(252)
        else:
            sharpe_ratio = 0
        
        # Usamos o Sharpe Ratio como métrica de otimização
        return sharpe_ratio
    
    # Cria o estudo Optuna
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)
    
    print("\n--- MELHORES HIPERPARÂMETROS ---")
    print(f"Valor: {study.best_value:.4f}")
    for key, value in study.best_params.items():
        print(f"{key}: {value}")
    
    # Adiciona parâmetros fixos aos melhores parâmetros
    best_params = {
        'env_params': {
            'commission_pct': study.best_params['commission'],
            'slippage_pct': study.best_params['slippage'],
            'reward_type': study.best_params['reward_type']
        },
        'agent_params': {
            'policy': 'MlpPolicy',
            'learning_rate': study.best_params['learning_rate'],
            'n_steps': study.best_params['n_steps'],
            'batch_size': study.best_params['batch_size'],
            'gamma': study.best_params['gamma'],
            'policy_kwargs': {
                'net_arch': eval(study.best_params['net_arch'])
            },
            'tensorboard_log': "./ppo_trading_tensorboard/"
        }
    }
    
    return best_params