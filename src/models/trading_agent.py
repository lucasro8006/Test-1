"""
Agente de trading baseado em Reinforcement Learning
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import PPO, A2C, SAC, TD3
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# Tenta importar SHAP para análise de importância de features
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False


class TradingAgent:
    """
    Agente de trading baseado em Reinforcement Learning.
    
    Esta classe encapsula um modelo de RL para trading, fornecendo
    métodos para treinamento, previsão, avaliação e persistência.
    Suporta diferentes algoritmos de RL como PPO, A2C, SAC e TD3.
    """
    
    def __init__(self, model_params, algorithm='PPO'):
        """
        Inicializa o agente de trading.
        
        Args:
            model_params (dict): Parâmetros para o modelo
            algorithm (str): Algoritmo de RL a ser usado ('PPO', 'A2C', 'SAC', 'TD3')
        """
        self.model_params = model_params
        self.algorithm = algorithm
        self.model = None
        self.training_history = {
            'rewards': [],
            'net_worths': [],
            'sharpe_ratios': [],
            'drawdowns': []
        }
        self.feature_importance = None
        
    def train(self, env, total_timesteps=100000, progress_bar=True, eval_env=None, 
              eval_freq=10000, callback=None, save_path=None):
        """
        Treina o agente no ambiente fornecido.
        
        Args:
            env: Ambiente de treinamento
            total_timesteps (int): Número total de passos de treinamento
            progress_bar (bool): Exibir barra de progresso
            eval_env: Ambiente de avaliação (opcional)
            eval_freq (int): Frequência de avaliação
            callback: Callback personalizado (opcional)
            save_path (str): Caminho para salvar o melhor modelo (opcional)
            
        Returns:
            self: O próprio agente treinado
        """
        # Cria o modelo se não existir
        if self.model is None:
            self._create_model(env)
        
        # Configura callbacks
        callbacks = []
        
        # Callback para monitoramento
        trading_callback = TradingCallback(verbose=1)
        callbacks.append(trading_callback)
        
        # Callback para avaliação
        if eval_env is not None:
            eval_callback = EvalCallback(
                eval_env,
                best_model_save_path=save_path,
                log_path=save_path,
                eval_freq=eval_freq,
                deterministic=True,
                render=False
            )
            callbacks.append(eval_callback)
            
        # Adiciona callback personalizado
        if callback is not None:
            callbacks.append(callback)
            
        # Treina o modelo
        self.model.learn(
            total_timesteps=total_timesteps, 
            progress_bar=progress_bar,
            callback=callbacks
        )
        
        # Armazena histórico de treinamento
        self.training_history['rewards'] = trading_callback.rewards
        self.training_history['net_worths'] = trading_callback.net_worths
        self.training_history['sharpe_ratios'] = trading_callback.sharpe_ratios
        self.training_history['drawdowns'] = trading_callback.drawdowns
        
        return self
    
    def _create_model(self, env):
        """
        Cria o modelo de RL com base no algoritmo escolhido.
        
        Args:
            env: Ambiente de treinamento
        """
        if self.algorithm == 'PPO':
            self.model = PPO(env=env, **self.model_params)
        elif self.algorithm == 'A2C':
            self.model = A2C(env=env, **self.model_params)
        elif self.algorithm == 'SAC':
            # Verifica se o ambiente tem espaço de ação contínuo
            if hasattr(env, 'action_space') and hasattr(env.action_space, 'shape'):
                self.model = SAC(env=env, **self.model_params)
            else:
                raise ValueError("SAC requer um espaço de ação contínuo")
        elif self.algorithm == 'TD3':
            # Verifica se o ambiente tem espaço de ação contínuo
            if hasattr(env, 'action_space') and hasattr(env.action_space, 'shape'):
                self.model = TD3(env=env, **self.model_params)
            else:
                raise ValueError("TD3 requer um espaço de ação contínuo")
        else:
            raise ValueError(f"Algoritmo não suportado: {self.algorithm}")
    
    def predict(self, observation, deterministic=True):
        """
        Faz uma previsão com base na observação.
        
        Args:
            observation: Observação do ambiente
            deterministic (bool): Se True, retorna a ação com maior probabilidade
            
        Returns:
            tuple: Ação e estados
        """
        if self.model is None:
            raise ValueError("O modelo não foi treinado ainda")
            
        return self.model.predict(observation, deterministic=deterministic)
    
    def evaluate(self, env, n_eval_episodes=10):
        """
        Avalia o desempenho do agente em um ambiente.
        
        Args:
            env: Ambiente de avaliação
            n_eval_episodes (int): Número de episódios para avaliação
            
        Returns:
            dict: Métricas de avaliação
        """
        if self.model is None:
            raise ValueError("O modelo não foi treinado ainda")
            
        # Avalia o modelo
        mean_reward, std_reward = evaluate_policy(
            self.model, 
            env, 
            n_eval_episodes=n_eval_episodes,
            deterministic=True
        )
        
        # Coleta métricas adicionais
        metrics = {
            'mean_reward': mean_reward,
            'std_reward': std_reward
        }
        
        # Coleta métricas financeiras se disponíveis
        if hasattr(env, 'envs') and hasattr(env.envs[0], 'get_metrics'):
            financial_metrics = env.envs[0].get_metrics()
            metrics.update(financial_metrics)
            
        return metrics
    
    def save(self, path):
        """
        Salva o modelo e metadados em disco.
        
        Args:
            path (str): Caminho base para salvar o modelo
            
        Returns:
            str: Caminho onde o modelo foi salvo
        """
        if self.model is None:
            raise ValueError("O modelo não foi treinado ainda")
            
        # Cria o diretório se não existir
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Salva o modelo
        model_path = self.model.save(path)
        
        # Salva metadados
        metadata = {
            'algorithm': self.algorithm,
            'model_params': self.model_params,
            'training_history': {
                'rewards_mean': np.mean(self.training_history['rewards']) if self.training_history['rewards'] else None,
                'rewards_std': np.std(self.training_history['rewards']) if self.training_history['rewards'] else None,
                'net_worths_final': self.training_history['net_worths'][-1] if self.training_history['net_worths'] else None,
                'sharpe_ratio': np.mean(self.training_history['sharpe_ratios']) if self.training_history['sharpe_ratios'] else None,
                'max_drawdown': np.max(self.training_history['drawdowns']) if self.training_history['drawdowns'] else None
            }
        }
        
        with open(f"{path}_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=4)
            
        # Salva histórico de treinamento completo
        with open(f"{path}_history.pkl", 'wb') as f:
            pickle.dump(self.training_history, f)
            
        return model_path
    
    def load(self, path, env=None):
        """
        Carrega o modelo e metadados do disco.
        
        Args:
            path (str): Caminho base para carregar o modelo
            env: Ambiente (opcional, necessário para alguns algoritmos)
            
        Returns:
            self: O próprio agente com o modelo carregado
        """
        # Carrega metadados
        try:
            with open(f"{path}_metadata.json", 'r') as f:
                metadata = json.load(f)
                self.algorithm = metadata['algorithm']
                self.model_params = metadata['model_params']
        except FileNotFoundError:
            print(f"Aviso: Metadados não encontrados em {path}_metadata.json")
            
        # Carrega histórico de treinamento
        try:
            with open(f"{path}_history.pkl", 'rb') as f:
                self.training_history = pickle.load(f)
        except FileNotFoundError:
            print(f"Aviso: Histórico não encontrado em {path}_history.pkl")
            
        # Carrega o modelo
        if self.algorithm == 'PPO':
            self.model = PPO.load(path, env=env)
        elif self.algorithm == 'A2C':
            self.model = A2C.load(path, env=env)
        elif self.algorithm == 'SAC':
            self.model = SAC.load(path, env=env)
        elif self.algorithm == 'TD3':
            self.model = TD3.load(path, env=env)
        else:
            # Tenta carregar como PPO por padrão
            self.model = PPO.load(path, env=env)
            
        return self
        
    def plot_training_history(self, figsize=(15, 10)):
        """
        Plota o histórico de treinamento.
        
        Args:
            figsize (tuple): Tamanho da figura
            
        Returns:
            matplotlib.figure.Figure: Figura com os gráficos
        """
        if not any(self.training_history.values()):
            print("Aviso: Não há histórico de treinamento para plotar")
            return None
            
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # Plota recompensas
        if self.training_history['rewards']:
            axes[0, 0].plot(self.training_history['rewards'])
            axes[0, 0].set_title('Recompensas')
            axes[0, 0].set_xlabel('Passos')
            axes[0, 0].set_ylabel('Recompensa')
            axes[0, 0].grid(True)
            
        # Plota patrimônio líquido
        if self.training_history['net_worths']:
            axes[0, 1].plot(self.training_history['net_worths'])
            axes[0, 1].set_title('Patrimônio Líquido')
            axes[0, 1].set_xlabel('Passos')
            axes[0, 1].set_ylabel('Valor ($)')
            axes[0, 1].grid(True)
            
        # Plota Sharpe Ratio
        if self.training_history['sharpe_ratios']:
            axes[1, 0].plot(self.training_history['sharpe_ratios'])
            axes[1, 0].set_title('Sharpe Ratio')
            axes[1, 0].set_xlabel('Passos')
            axes[1, 0].set_ylabel('Sharpe Ratio')
            axes[1, 0].grid(True)
            
        # Plota drawdowns
        if self.training_history['drawdowns']:
            axes[1, 1].plot(self.training_history['drawdowns'])
            axes[1, 1].set_title('Drawdowns')
            axes[1, 1].set_xlabel('Passos')
            axes[1, 1].set_ylabel('Drawdown (%)')
            axes[1, 1].grid(True)
            
        plt.tight_layout()
        return fig
        
    def analyze_feature_importance(self, env, data, n_samples=100):
        """
        Analisa a importância das features usando SHAP.
        
        Args:
            env: Ambiente de trading
            data (pd.DataFrame): Dados para análise
            n_samples (int): Número de amostras para análise
            
        Returns:
            pd.DataFrame: DataFrame com importância das features
        """
        if not SHAP_AVAILABLE:
            print("Aviso: SHAP não está instalado. Use 'pip install shap' para instalar.")
            return None
            
        if self.model is None:
            raise ValueError("O modelo não foi treinado ainda")
            
        try:
            # Cria um explainer SHAP
            explainer = shap.Explainer(self.model.policy.predict)
            
            # Seleciona amostras aleatórias
            if len(data) > n_samples:
                sample_indices = np.random.choice(len(data), n_samples, replace=False)
                samples = data.iloc[sample_indices]
            else:
                samples = data
                
            # Prepara as observações
            observations = []
            for i in range(len(samples)):
                obs = env.reset()
                observations.append(obs)
                
            # Calcula valores SHAP
            shap_values = explainer(observations)
            
            # Cria DataFrame com importância
            feature_names = env.observation_space.names if hasattr(env.observation_space, 'names') else [f"feature_{i}" for i in range(shap_values.values.shape[1])]
            
            self.feature_importance = pd.DataFrame({
                'feature': feature_names,
                'importance': np.abs(shap_values.values).mean(axis=0)
            })
            
            # Ordena por importância
            self.feature_importance = self.feature_importance.sort_values('importance', ascending=False)
            
            return self.feature_importance
            
        except Exception as e:
            print(f"Erro ao analisar importância das features: {e}")
            return None
            
    def plot_feature_importance(self, figsize=(10, 8)):
        """
        Plota a importância das features.
        
        Args:
            figsize (tuple): Tamanho da figura
            
        Returns:
            matplotlib.figure.Figure: Figura com o gráfico
        """
        if self.feature_importance is None:
            print("Aviso: Análise de importância das features não foi realizada")
            return None
            
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plota importância das features
        self.feature_importance.sort_values('importance').plot(
            kind='barh', 
            x='feature', 
            y='importance', 
            ax=ax
        )
        
        ax.set_title('Importância das Features')
        ax.set_xlabel('Importância')
        ax.set_ylabel('Feature')
        ax.grid(True)
        
        plt.tight_layout()
        return fig


class TradingCallback(BaseCallback):
    """
    Callback para monitorar o treinamento do agente de trading.
    
    Esta classe coleta métricas durante o treinamento para análise posterior.
    """
    
    def __init__(self, verbose=0, log_freq=100, save_path=None):
        """
        Inicializa o callback.
        
        Args:
            verbose (int): Nível de verbosidade
            log_freq (int): Frequência de log (a cada quantos passos)
            save_path (str): Caminho para salvar logs (opcional)
        """
        super(TradingCallback, self).__init__(verbose)
        self.log_freq = log_freq
        self.save_path = save_path
        
        # Métricas coletadas
        self.rewards = []
        self.net_worths = []
        self.sharpe_ratios = []
        self.drawdowns = []
        self.positions = []
        self.trades = []
        
        # Contadores
        self.n_steps = 0
        self.n_trades = 0
        self.last_position = 0
        
    def _on_training_start(self):
        """Chamado no início do treinamento."""
        self.n_steps = 0
        self.n_trades = 0
        self.last_position = 0
        
        # Cria diretório de logs se necessário
        if self.save_path:
            os.makedirs(self.save_path, exist_ok=True)
        
    def _on_step(self):
        """
        Chamado a cada passo do treinamento.
        
        Returns:
            bool: Se True, continua o treinamento
        """
        # Incrementa contador de passos
        self.n_steps += 1
        
        # Obtém informações do ambiente
        env = self.training_env.envs[0]
        
        # Registra recompensas
        if hasattr(env, 'rewards') and len(env.rewards) > 0:
            self.rewards.append(env.rewards[-1])
        elif hasattr(env, 'reward'):
            self.rewards.append(env.reward)
            
        # Registra patrimônio líquido
        if hasattr(env, 'net_worth'):
            self.net_worths.append(env.net_worth)
            
        # Registra posição atual
        if hasattr(env, 'position'):
            current_position = env.position
            self.positions.append(current_position)
            
            # Detecta trades
            if current_position != self.last_position:
                self.n_trades += 1
                self.trades.append({
                    'step': self.n_steps,
                    'type': 'buy' if current_position > self.last_position else 'sell',
                    'price': env.df.iloc[env.current_step]['close'] if hasattr(env, 'df') and hasattr(env, 'current_step') else 0,
                    'position': current_position
                })
                
            self.last_position = current_position
            
        # Calcula métricas financeiras
        if len(self.rewards) >= 20:  # Precisa de pelo menos 20 pontos para métricas significativas
            # Sharpe Ratio
            returns = np.array(self.rewards[-20:])
            sharpe = returns.mean() / (returns.std() + 1e-9)  # Evita divisão por zero
            self.sharpe_ratios.append(sharpe)
            
            # Drawdown
            if len(self.net_worths) >= 20:
                net_worths = np.array(self.net_worths[-20:])
                peak = np.maximum.accumulate(net_worths)
                drawdown = (net_worths - peak) / peak
                max_drawdown = abs(drawdown.min()) if len(drawdown) > 0 else 0
                self.drawdowns.append(max_drawdown)
                
        # Log periódico
        if self.verbose > 0 and self.n_steps % self.log_freq == 0:
            mean_reward = np.mean(self.rewards[-100:]) if len(self.rewards) > 0 else 0
            mean_net_worth = self.net_worths[-1] if len(self.net_worths) > 0 else 0
            mean_sharpe = np.mean(self.sharpe_ratios[-100:]) if len(self.sharpe_ratios) > 0 else 0
            
            print(f"Step: {self.n_steps}, "
                  f"Reward: {mean_reward:.4f}, "
                  f"Net Worth: {mean_net_worth:.2f}, "
                  f"Sharpe: {mean_sharpe:.2f}, "
                  f"Trades: {self.n_trades}")
                  
        # Salva logs periodicamente
        if self.save_path and self.n_steps % (self.log_freq * 10) == 0:
            self._save_logs()
            
        return True
        
    def _on_training_end(self):
        """Chamado no final do treinamento."""
        if self.save_path:
            self._save_logs()
            
    def _save_logs(self):
        """Salva logs em disco."""
        if not self.save_path:
            return
            
        # Cria dicionário com métricas
        logs = {
            'rewards': self.rewards,
            'net_worths': self.net_worths,
            'sharpe_ratios': self.sharpe_ratios,
            'drawdowns': self.drawdowns,
            'positions': self.positions,
            'trades': self.trades,
            'n_steps': self.n_steps,
            'n_trades': self.n_trades
        }
        
        # Salva em pickle
        with open(f"{self.save_path}/training_logs.pkl", 'wb') as f:
            pickle.dump(logs, f)
            
        # Salva resumo em JSON
        summary = {
            'n_steps': self.n_steps,
            'n_trades': self.n_trades,
            'final_net_worth': self.net_worths[-1] if len(self.net_worths) > 0 else 0,
            'mean_reward': np.mean(self.rewards) if len(self.rewards) > 0 else 0,
            'mean_sharpe': np.mean(self.sharpe_ratios) if len(self.sharpe_ratios) > 0 else 0,
            'max_drawdown': np.max(self.drawdowns) if len(self.drawdowns) > 0 else 0
        }
        
        with open(f"{self.save_path}/training_summary.json", 'w') as f:
            json.dump(summary, f, indent=4)