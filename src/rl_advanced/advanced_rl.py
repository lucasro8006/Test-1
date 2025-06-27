"""
Técnicas Avançadas de Reinforcement Learning
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Tuple, Any, Optional
from collections import deque
import random
import warnings
warnings.filterwarnings('ignore')

class CuriosityDrivenAgent:
    """
    Agente com curiosity-driven exploration usando Intrinsic Curiosity Module (ICM)
    """
    def __init__(self, state_dim: int, action_dim: int, device='cpu'):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        
        # Redes do ICM
        self.feature_network = self._build_feature_network()
        self.inverse_model = self._build_inverse_model()
        self.forward_model = self._build_forward_model()
        
        # Otimizadores
        self.feature_optimizer = optim.Adam(self.feature_network.parameters(), lr=0.001)
        self.inverse_optimizer = optim.Adam(self.inverse_model.parameters(), lr=0.001)
        self.forward_optimizer = optim.Adam(self.forward_model.parameters(), lr=0.001)
        
        # Hiperparâmetros
        self.intrinsic_reward_scale = 0.1
        self.feature_loss_scale = 0.2
        
    def _build_feature_network(self) -> nn.Module:
        """Constrói a rede de features"""
        return nn.Sequential(
            nn.Linear(self.state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        ).to(self.device)
    
    def _build_inverse_model(self) -> nn.Module:
        """Constrói o modelo inverso (prediz ação dado estados)"""
        return nn.Sequential(
            nn.Linear(64, 64),  # 32 + 32 features de dois estados
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, self.action_dim)
        ).to(self.device)
    
    def _build_forward_model(self) -> nn.Module:
        """Constrói o modelo forward (prediz próximo estado dado estado e ação)"""
        return nn.Sequential(
            nn.Linear(32 + self.action_dim, 64),  # features + action
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 32)  # prediz features do próximo estado
        ).to(self.device)
    
    def calculate_intrinsic_reward(self, state: np.ndarray, 
                                 action: np.ndarray, 
                                 next_state: np.ndarray) -> float:
        """
        Calcula recompensa intrínseca baseada na curiosidade
        """
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
            action_tensor = torch.FloatTensor([action]).unsqueeze(0).to(self.device)
            
            # Extrai features
            state_features = self.feature_network(state_tensor)
            next_state_features = self.feature_network(next_state_tensor)
            
            # Prediz próximo estado
            predicted_next_features = self.forward_model(
                torch.cat([state_features, action_tensor], dim=1)
            )
            
            # Erro de predição = curiosidade
            prediction_error = nn.MSELoss()(predicted_next_features, next_state_features)
            
            return prediction_error.item() * self.intrinsic_reward_scale

class MultiAgentTradingEnvironment:
    """
    Ambiente multi-agente para simular múltiplos traders
    """
    def __init__(self, market_data: pd.DataFrame, num_agents: int = 3):
        self.market_data = market_data
        self.num_agents = num_agents
        self.agents = {}
        self.current_step = 0
        self.market_impact = 0.001  # Impacto no mercado por operação
        
        # Inicializa agentes
        for i in range(num_agents):
            self.agents[f'agent_{i}'] = {
                'balance': 100000,
                'position': 0,
                'net_worth': 100000,
                'strategy_type': ['momentum', 'mean_reversion', 'arbitrage'][i % 3]
            }
    
    def step(self, actions: Dict[str, int]) -> Tuple[Dict, Dict, bool, Dict]:
        """
        Executa um passo no ambiente multi-agente
        """
        if self.current_step >= len(self.market_data) - 1:
            return {}, {}, True, {}
        
        current_price = self.market_data.iloc[self.current_step]['close']
        
        # Calcula impacto agregado no mercado
        total_buy_pressure = sum(1 for action in actions.values() if action == 1)
        total_sell_pressure = sum(1 for action in actions.values() if action == 2)
        
        # Ajusta preço baseado na pressão de compra/venda
        net_pressure = total_buy_pressure - total_sell_pressure
        adjusted_price = current_price * (1 + net_pressure * self.market_impact)
        
        # Executa ações de cada agente
        observations = {}
        rewards = {}
        
        for agent_id, action in actions.items():
            if agent_id in self.agents:
                reward = self._execute_agent_action(agent_id, action, adjusted_price)
                rewards[agent_id] = reward
                observations[agent_id] = self._get_agent_observation(agent_id)
        
        self.current_step += 1
        done = self.current_step >= len(self.market_data) - 1
        
        return observations, rewards, done, {}
    
    def _execute_agent_action(self, agent_id: str, action: int, price: float) -> float:
        """
        Executa ação de um agente específico
        """
        agent = self.agents[agent_id]
        prev_net_worth = agent['net_worth']
        
        if action == 1 and agent['position'] == 0:  # Buy
            shares = agent['balance'] / price
            agent['position'] = shares
            agent['balance'] = 0
            agent['net_worth'] = shares * price
            
        elif action == 2 and agent['position'] > 0:  # Sell
            agent['balance'] = agent['position'] * price
            agent['position'] = 0
            agent['net_worth'] = agent['balance']
        
        # Calcula recompensa
        return (agent['net_worth'] - prev_net_worth) / prev_net_worth
    
    def _get_agent_observation(self, agent_id: str) -> np.ndarray:
        """
        Obtém observação para um agente específico
        """
        agent = self.agents[agent_id]
        market_obs = self.market_data.iloc[self.current_step].values
        
        # Adiciona informações específicas do agente
        agent_obs = np.array([
            agent['balance'] / 100000,  # Normalizado
            agent['position'],
            agent['net_worth'] / 100000  # Normalizado
        ])
        
        return np.concatenate([market_obs, agent_obs])

class ImitationLearningAgent:
    """
    Agente que aprende por imitação de traders experientes
    """
    def __init__(self, state_dim: int, action_dim: int, device='cpu'):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        
        # Rede de política
        self.policy_network = self._build_policy_network()
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=0.001)
        
        # Buffer de demonstrações
        self.demonstration_buffer = []
        
    def _build_policy_network(self) -> nn.Module:
        """Constrói a rede de política"""
        return nn.Sequential(
            nn.Linear(self.state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, self.action_dim),
            nn.Softmax(dim=-1)
        ).to(self.device)
    
    def generate_expert_demonstrations(self, market_data: pd.DataFrame, 
                                     strategy: str = 'momentum') -> List[Tuple]:
        """
        Gera demonstrações de um trader expert simulado
        """
        demonstrations = []
        
        for i in range(len(market_data) - 1):
            state = market_data.iloc[i].values
            
            if strategy == 'momentum':
                action = self._momentum_strategy_action(market_data, i)
            elif strategy == 'mean_reversion':
                action = self._mean_reversion_strategy_action(market_data, i)
            else:
                action = 0  # Hold
            
            demonstrations.append((state, action))
        
        return demonstrations
    
    def _momentum_strategy_action(self, data: pd.DataFrame, idx: int) -> int:
        """
        Estratégia de momentum para gerar demonstrações
        """
        if idx < 20:
            return 0
        
        short_ma = data['close'].iloc[idx-5:idx].mean()
        long_ma = data['close'].iloc[idx-20:idx].mean()
        
        if short_ma > long_ma * 1.02:  # 2% acima
            return 1  # Buy
        elif short_ma < long_ma * 0.98:  # 2% abaixo
            return 2  # Sell
        else:
            return 0  # Hold
    
    def _mean_reversion_strategy_action(self, data: pd.DataFrame, idx: int) -> int:
        """
        Estratégia de reversão à média para gerar demonstrações
        """
        if idx < 50:
            return 0
        
        current_price = data['close'].iloc[idx]
        long_ma = data['close'].iloc[idx-50:idx].mean()
        std = data['close'].iloc[idx-50:idx].std()
        
        if current_price < long_ma - 2 * std:  # Muito abaixo da média
            return 1  # Buy
        elif current_price > long_ma + 2 * std:  # Muito acima da média
            return 2  # Sell
        else:
            return 0  # Hold
    
    def train_on_demonstrations(self, demonstrations: List[Tuple], num_epochs: int = 100):
        """
        Treina a política usando demonstrações
        """
        if not demonstrations:
            print("Nenhuma demonstração disponível")
            return
        
        for epoch in range(num_epochs):
            total_loss = 0
            
            # Embaralha demonstrações
            random.shuffle(demonstrations)
            
            for state, action in demonstrations:
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                action_tensor = torch.LongTensor([action]).to(self.device)
                
                # Forward pass
                action_probs = self.policy_network(state_tensor)
                
                # Loss de classificação
                loss = nn.CrossEntropyLoss()(action_probs, action_tensor)
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
            
            if epoch % 20 == 0:
                avg_loss = total_loss / len(demonstrations)
                print(f"Epoch {epoch}, Average Loss: {avg_loss:.4f}")
    
    def predict(self, state: np.ndarray) -> int:
        """
        Prediz ação baseada no estado
        """
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            action_probs = self.policy_network(state_tensor)
            action = torch.argmax(action_probs, dim=-1).item()
            return action

class DistributionalRLAgent:
    """
    Agente que modela a distribuição completa dos retornos (C51/Rainbow)
    """
    def __init__(self, state_dim: int, action_dim: int, 
                 num_atoms: int = 51, v_min: float = -10, v_max: float = 10):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_atoms = num_atoms
        self.v_min = v_min
        self.v_max = v_max
        
        # Suporte da distribuição
        self.support = torch.linspace(v_min, v_max, num_atoms)
        self.delta_z = (v_max - v_min) / (num_atoms - 1)
        
        # Rede de distribuição
        self.distribution_network = self._build_distribution_network()
        self.optimizer = optim.Adam(self.distribution_network.parameters(), lr=0.001)
        
        # Buffer de replay
        self.replay_buffer = deque(maxlen=10000)
        
    def _build_distribution_network(self) -> nn.Module:
        """
        Constrói rede que prediz distribuição de valores
        """
        return nn.Sequential(
            nn.Linear(self.state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, self.action_dim * self.num_atoms)
        )
    
    def predict(self, state: np.ndarray, epsilon: float = 0.1) -> int:
        """
        Prediz ação usando a distribuição de valores
        """
        if random.random() < epsilon:
            return random.randint(0, self.action_dim - 1)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            logits = self.distribution_network(state_tensor)
            
            # Reshape para (batch, action, atoms)
            logits = logits.view(-1, self.action_dim, self.num_atoms)
            
            # Aplica softmax para obter distribuição
            distributions = torch.softmax(logits, dim=-1)
            
            # Calcula valor esperado para cada ação
            q_values = torch.sum(distributions * self.support, dim=-1)
            
            return torch.argmax(q_values).item()

class OfflineRLAgent:
    """
    Agente para Offline RL - aprende apenas com dados históricos
    """
    def __init__(self, state_dim: int, action_dim: int):
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Redes Q conservadoras (CQL)
        self.q_network1 = self._build_q_network()
        self.q_network2 = self._build_q_network()
        
        # Otimizadores
        self.q_optimizer1 = optim.Adam(self.q_network1.parameters(), lr=0.001)
        self.q_optimizer2 = optim.Adam(self.q_network2.parameters(), lr=0.001)
        
        # Hiperparâmetros CQL
        self.cql_alpha = 1.0
        
    def _build_q_network(self) -> nn.Module:
        """Constrói rede Q"""
        return nn.Sequential(
            nn.Linear(self.state_dim + 1, 128),  # state + action
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    
    def predict(self, state: np.ndarray) -> int:
        """
        Prediz ação usando Q-networks
        """
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            
            # Testa todas as ações
            q_values = []
            for action in range(self.action_dim):
                action_tensor = torch.FloatTensor([action]).unsqueeze(0)
                state_action = torch.cat([state_tensor, action_tensor], dim=1)
                
                q1 = self.q_network1(state_action)
                q2 = self.q_network2(state_action)
                q_value = torch.min(q1, q2)
                q_values.append(q_value.item())
            
            return np.argmax(q_values)