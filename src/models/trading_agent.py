"""
Agente de trading baseado em Reinforcement Learning
"""

import os
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback


class TradingAgent:
    """
    Agente de trading baseado em Reinforcement Learning.
    
    Esta classe encapsula um modelo de RL para trading, fornecendo
    métodos para treinamento, previsão e persistência.
    """
    
    def __init__(self, model_params):
        """
        Inicializa o agente de trading.
        
        Args:
            model_params (dict): Parâmetros para o modelo PPO
        """
        self.model_params = model_params
        self.model = None
        
    def train(self, env, total_timesteps=100000, progress_bar=True):
        """
        Treina o agente no ambiente fornecido.
        
        Args:
            env: Ambiente de treinamento
            total_timesteps (int): Número total de passos de treinamento
            progress_bar (bool): Exibir barra de progresso
            
        Returns:
            self: O próprio agente treinado
        """
        # Cria o modelo se não existir
        if self.model is None:
            self.model = PPO(env=env, **self.model_params)
        
        # Treina o modelo
        self.model.learn(total_timesteps=total_timesteps, progress_bar=progress_bar)
        
        return self
    
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
    
    def save(self, path):
        """
        Salva o modelo em disco.
        
        Args:
            path (str): Caminho para salvar o modelo
            
        Returns:
            str: Caminho onde o modelo foi salvo
        """
        if self.model is None:
            raise ValueError("O modelo não foi treinado ainda")
            
        # Cria o diretório se não existir
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        return self.model.save(path)
    
    def load(self, path):
        """
        Carrega o modelo do disco.
        
        Args:
            path (str): Caminho para carregar o modelo
            
        Returns:
            self: O próprio agente com o modelo carregado
        """
        self.model = PPO.load(path)
        return self


class TradingCallback(BaseCallback):
    """
    Callback para monitorar o treinamento do agente de trading.
    """
    
    def __init__(self, verbose=0):
        """
        Inicializa o callback.
        
        Args:
            verbose (int): Nível de verbosidade
        """
        super(TradingCallback, self).__init__(verbose)
        self.rewards = []
        self.net_worths = []
        
    def _on_step(self):
        """
        Chamado a cada passo do treinamento.
        
        Returns:
            bool: Se True, continua o treinamento
        """
        # Obtém informações do ambiente
        env = self.training_env.envs[0]
        
        # Registra recompensas e patrimônio líquido
        if hasattr(env, 'rewards'):
            self.rewards.append(env.rewards[-1])
        
        if hasattr(env, 'net_worth'):
            self.net_worths.append(env.net_worth)
        
        return True