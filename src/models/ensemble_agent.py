"""
Ensemble Agent - Combina múltiplos algoritmos de RL para melhor performance
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from typing import Dict, List, Any, Tuple
from stable_baselines3 import PPO, A2C, SAC, TD3
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.policies import ActorCriticPolicy
import warnings
warnings.filterwarnings('ignore')

class TransformerPolicy(ActorCriticPolicy):
    """
    Política customizada usando Transformer para capturar dependências temporais
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def _build_mlp_extractor(self) -> None:
        """Constrói o extrator de features usando Transformer"""
        self.mlp_extractor = TransformerFeatureExtractor(
            feature_dim=self.features_dim,
            net_arch=self.net_arch,
            activation_fn=self.activation_fn,
            device=self.device
        )

class TransformerFeatureExtractor(nn.Module):
    """
    Extrator de features usando Transformer para séries temporais
    """
    def __init__(self, feature_dim: int, net_arch: List[int], activation_fn, device):
        super().__init__()
        self.feature_dim = feature_dim
        self.device = device
        
        # Transformer layers
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=feature_dim,
                nhead=8,
                dim_feedforward=256,
                dropout=0.1,
                batch_first=True
            ),
            num_layers=2
        )
        
        # Policy and value networks
        self.policy_net = nn.Sequential(
            nn.Linear(feature_dim, net_arch[0]),
            activation_fn(),
            nn.Linear(net_arch[0], net_arch[1]),
            activation_fn()
        )
        
        self.value_net = nn.Sequential(
            nn.Linear(feature_dim, net_arch[0]),
            activation_fn(),
            nn.Linear(net_arch[0], net_arch[1]),
            activation_fn()
        )
        
        self.latent_dim_pi = net_arch[1]
        self.latent_dim_vf = net_arch[1]
    
    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass através do Transformer
        """
        # Reshape para sequência temporal se necessário
        if len(features.shape) == 2:
            features = features.unsqueeze(1)  # Add sequence dimension
        
        # Transformer encoding
        transformer_out = self.transformer(features)
        
        # Use a última saída da sequência
        last_output = transformer_out[:, -1, :]
        
        # Policy and value networks
        policy_features = self.policy_net(last_output)
        value_features = self.value_net(last_output)
        
        return policy_features, value_features

class LSTMPolicy(ActorCriticPolicy):
    """
    Política customizada usando LSTM para memória temporal
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def _build_mlp_extractor(self) -> None:
        """Constrói o extrator de features usando LSTM"""
        self.mlp_extractor = LSTMFeatureExtractor(
            feature_dim=self.features_dim,
            net_arch=self.net_arch,
            activation_fn=self.activation_fn,
            device=self.device
        )

class LSTMFeatureExtractor(nn.Module):
    """
    Extrator de features usando LSTM para séries temporais
    """
    def __init__(self, feature_dim: int, net_arch: List[int], activation_fn, device):
        super().__init__()
        self.feature_dim = feature_dim
        self.device = device
        self.hidden_size = 128
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=feature_dim,
            hidden_size=self.hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=0.1
        )
        
        # Policy and value networks
        self.policy_net = nn.Sequential(
            nn.Linear(self.hidden_size, net_arch[0]),
            activation_fn(),
            nn.Linear(net_arch[0], net_arch[1]),
            activation_fn()
        )
        
        self.value_net = nn.Sequential(
            nn.Linear(self.hidden_size, net_arch[0]),
            activation_fn(),
            nn.Linear(net_arch[0], net_arch[1]),
            activation_fn()
        )
        
        self.latent_dim_pi = net_arch[1]
        self.latent_dim_vf = net_arch[1]
        
        # Hidden states
        self.hidden_state = None
    
    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass através do LSTM
        """
        batch_size = features.shape[0]
        
        # Reshape para sequência temporal se necessário
        if len(features.shape) == 2:
            features = features.unsqueeze(1)  # Add sequence dimension
        
        # Initialize hidden state if needed
        if self.hidden_state is None or self.hidden_state[0].shape[1] != batch_size:
            self.hidden_state = (
                torch.zeros(2, batch_size, self.hidden_size, device=self.device),
                torch.zeros(2, batch_size, self.hidden_size, device=self.device)
            )
        
        # LSTM forward
        lstm_out, self.hidden_state = self.lstm(features, self.hidden_state)
        
        # Use a última saída da sequência
        last_output = lstm_out[:, -1, :]
        
        # Policy and value networks
        policy_features = self.policy_net(last_output)
        value_features = self.value_net(last_output)
        
        return policy_features, value_features

class EnsembleAgent:
    """
    Agente ensemble que combina múltiplos algoritmos de RL
    """
    def __init__(self, env, ensemble_config: Dict[str, Any]):
        self.env = env
        self.config = ensemble_config
        self.agents = {}
        self.weights = {}
        self.performance_history = {}
        
        # Inicializa agentes
        self._initialize_agents()
    
    def _initialize_agents(self):
        """Inicializa todos os agentes do ensemble"""
        
        # PPO padrão
        self.agents['ppo_standard'] = PPO(
            'MlpPolicy',
            self.env,
            learning_rate=0.0003,
            n_steps=128,
            batch_size=64,
            gamma=0.99,
            verbose=0
        )
        
        # PPO com Transformer
        try:
            self.agents['ppo_transformer'] = PPO(
                TransformerPolicy,
                self.env,
                learning_rate=0.0001,
                n_steps=256,
                batch_size=32,
                gamma=0.99,
                verbose=0
            )
        except Exception as e:
            print(f"Aviso: Não foi possível criar PPO com Transformer: {e}")
        
        # PPO com LSTM
        try:
            self.agents['ppo_lstm'] = PPO(
                LSTMPolicy,
                self.env,
                learning_rate=0.0001,
                n_steps=256,
                batch_size=32,
                gamma=0.99,
                verbose=0
            )
        except Exception as e:
            print(f"Aviso: Não foi possível criar PPO com LSTM: {e}")
        
        # A2C
        self.agents['a2c'] = A2C(
            'MlpPolicy',
            self.env,
            learning_rate=0.001,
            n_steps=5,
            gamma=0.99,
            verbose=0
        )
        
        # SAC (para ambientes contínuos)
        try:
            self.agents['sac'] = SAC(
                'MlpPolicy',
                self.env,
                learning_rate=0.0003,
                buffer_size=100000,
                gamma=0.99,
                verbose=0
            )
        except Exception as e:
            print(f"Aviso: SAC não suportado para este ambiente: {e}")
        
        # Inicializa pesos uniformes
        num_agents = len(self.agents)
        for agent_name in self.agents.keys():
            self.weights[agent_name] = 1.0 / num_agents
            self.performance_history[agent_name] = []
    
    def train(self, total_timesteps: int, eval_freq: int = 10000):
        """
        Treina todos os agentes do ensemble
        """
        print(f"Treinando ensemble com {len(self.agents)} agentes...")
        
        for agent_name, agent in self.agents.items():
            print(f"Treinando {agent_name}...")
            try:
                agent.learn(total_timesteps=total_timesteps)
                print(f"{agent_name} treinado com sucesso!")
            except Exception as e:
                print(f"Erro ao treinar {agent_name}: {e}")
        
        # Avalia e ajusta pesos
        self._evaluate_and_adjust_weights()
    
    def _evaluate_and_adjust_weights(self):
        """
        Avalia cada agente e ajusta os pesos baseado na performance
        """
        performances = {}
        
        for agent_name, agent in self.agents.items():
            try:
                # Avaliação simples - pode ser expandida
                obs = self.env.reset()
                total_reward = 0
                done = False
                steps = 0
                
                while not done and steps < 100:
                    action, _ = agent.predict(obs, deterministic=True)
                    obs, reward, done, info = self.env.step(action)
                    total_reward += reward
                    steps += 1
                
                performances[agent_name] = total_reward
                self.performance_history[agent_name].append(total_reward)
                
            except Exception as e:
                print(f"Erro ao avaliar {agent_name}: {e}")
                performances[agent_name] = -np.inf
        
        # Ajusta pesos baseado na performance (softmax)
        if performances:
            performance_values = np.array(list(performances.values()))
            # Evita overflow no softmax
            performance_values = performance_values - np.max(performance_values)
            exp_values = np.exp(performance_values)
            softmax_weights = exp_values / np.sum(exp_values)
            
            for i, agent_name in enumerate(performances.keys()):
                self.weights[agent_name] = softmax_weights[i]
    
    def predict(self, observation, deterministic=True):
        """
        Faz predição usando ensemble ponderado
        """
        if len(self.agents) == 0:
            raise ValueError("Nenhum agente disponível no ensemble")
        
        # Coleta predições de todos os agentes
        predictions = {}
        
        for agent_name, agent in self.agents.items():
            try:
                action, _ = agent.predict(observation, deterministic=deterministic)
                predictions[agent_name] = action
            except Exception as e:
                print(f"Erro na predição do {agent_name}: {e}")
                continue
        
        if not predictions:
            # Fallback para ação aleatória
            return np.array([1]), None  # Ação de manter
        
        # Combina predições usando pesos
        weighted_action = 0
        total_weight = 0
        
        for agent_name, action in predictions.items():
            weight = self.weights.get(agent_name, 0)
            weighted_action += weight * action
            total_weight += weight
        
        if total_weight > 0:
            final_action = weighted_action / total_weight
        else:
            final_action = list(predictions.values())[0]  # Usa primeira predição disponível
        
        return np.array([int(np.round(final_action))]), None
    
    def save(self, path: str):
        """
        Salva todos os agentes do ensemble
        """
        import os
        os.makedirs(path, exist_ok=True)
        
        for agent_name, agent in self.agents.items():
            try:
                agent.save(f"{path}/{agent_name}")
            except Exception as e:
                print(f"Erro ao salvar {agent_name}: {e}")
        
        # Salva pesos e histórico
        import pickle
        with open(f"{path}/ensemble_metadata.pkl", 'wb') as f:
            pickle.dump({
                'weights': self.weights,
                'performance_history': self.performance_history
            }, f)
    
    def load(self, path: str):
        """
        Carrega todos os agentes do ensemble
        """
        import os
        import pickle
        
        # Carrega metadados
        try:
            with open(f"{path}/ensemble_metadata.pkl", 'rb') as f:
                metadata = pickle.load(f)
                self.weights = metadata['weights']
                self.performance_history = metadata['performance_history']
        except Exception as e:
            print(f"Erro ao carregar metadados: {e}")
        
        # Carrega agentes
        for agent_name in self.agents.keys():
            try:
                if os.path.exists(f"{path}/{agent_name}.zip"):
                    if agent_name.startswith('ppo'):
                        self.agents[agent_name] = PPO.load(f"{path}/{agent_name}")
                    elif agent_name.startswith('a2c'):
                        self.agents[agent_name] = A2C.load(f"{path}/{agent_name}")
                    elif agent_name.startswith('sac'):
                        self.agents[agent_name] = SAC.load(f"{path}/{agent_name}")
            except Exception as e:
                print(f"Erro ao carregar {agent_name}: {e}")
    
    def get_ensemble_stats(self) -> Dict[str, Any]:
        """
        Retorna estatísticas do ensemble
        """
        return {
            'num_agents': len(self.agents),
            'weights': self.weights,
            'performance_history': self.performance_history,
            'agent_names': list(self.agents.keys())
        }