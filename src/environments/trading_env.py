"""
Ambiente de trading para Reinforcement Learning
"""

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces


class TradingEnv(gym.Env):
    """
    Ambiente de trading para Reinforcement Learning.
    
    Este ambiente simula um mercado de trading onde um agente pode comprar,
    vender ou manter uma posição em um ativo financeiro.
    """
    
    def __init__(
        self, 
        df, 
        initial_balance=100000, 
        commission_pct=0.001, 
        slippage_pct=0.0005,
        reward_type='pnl',
        window_size=1
    ):
        """
        Inicializa o ambiente de trading.
        
        Args:
            df (pd.DataFrame): DataFrame com dados de preços e features
            initial_balance (float): Saldo inicial
            commission_pct (float): Percentual de comissão por operação
            slippage_pct (float): Percentual de slippage por operação
            reward_type (str): Tipo de recompensa ('pnl', 'sharpe', 'calmar')
            window_size (int): Tamanho da janela de observação
        """
        super(TradingEnv, self).__init__()
        
        self.df = df
        self.initial_balance = initial_balance
        self.commission_pct = commission_pct
        self.slippage_pct = slippage_pct
        self.reward_type = reward_type
        self.window_size = window_size
        
        # Espaço de ações: 0 = Manter, 1 = Comprar, 2 = Vender
        self.action_space = spaces.Discrete(3)
        
        # Espaço de observação: features + posição atual + PnL da posição
        n_features = len(df.columns)
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(n_features + 2,),
            dtype=np.float32
        )
        
        # Variáveis de estado
        self.current_step = 0
        self.balance = initial_balance
        self.position = 0  # 0 = sem posição, 1 = comprado
        self.position_price = 0
        self.position_pnl = 0
        self.net_worth = initial_balance
        self.history = []
        self.returns_history = []
        
    def reset(self, seed=None, options=None):
        """
        Reinicia o ambiente para um novo episódio.
        
        Args:
            seed (int, optional): Semente para geração de números aleatórios
            options (dict, optional): Opções adicionais
            
        Returns:
            tuple: Observação inicial e informações
        """
        super().reset(seed=seed)
        
        self.current_step = 0
        self.balance = self.initial_balance
        self.position = 0
        self.position_price = 0
        self.position_pnl = 0
        self.net_worth = self.initial_balance
        self.history = []
        self.returns_history = []
        
        # Registra o estado inicial
        self._update_history()
        
        return self._get_obs(), {}
    
    def step(self, action):
        """
        Executa uma ação no ambiente.
        
        Args:
            action (int): Ação a ser executada (0=Manter, 1=Comprar, 2=Vender)
            
        Returns:
            tuple: Observação, recompensa, terminado, truncado, info
        """
        # Verifica se o episódio terminou
        done = self.current_step >= len(self.df) - 1
        
        if done:
            return self._get_obs(), 0, True, False, {}
        
        # Obtém o preço atual
        current_price = self._get_current_price()
        
        # Executa a ação
        reward = self._take_action(action, current_price)
        
        # Atualiza o histórico
        self._update_history()
        
        # Avança para o próximo passo
        self.current_step += 1
        
        # Obtém a nova observação
        obs = self._get_obs()
        
        return obs, reward, done, False, {}
    
    def _get_obs(self):
        """
        Obtém a observação atual.
        
        Returns:
            np.array: Vetor de observação
        """
        # Certifica-se de não ultrapassar o tamanho do DataFrame
        step = min(self.current_step, len(self.df) - 1)
        obs = self.df.iloc[step].values
        obs = np.append(obs, [self.position, self.position_pnl])
        return obs.astype(np.float32)
    
    def _get_current_price(self):
        """
        Obtém o preço atual.
        
        Returns:
            float: Preço atual
        """
        return self.df.iloc[self.current_step]['close']
    
    def _take_action(self, action, current_price):
        """
        Executa uma ação de trading.
        
        Args:
            action (int): Ação a ser executada
            current_price (float): Preço atual
            
        Returns:
            float: Recompensa pela ação
        """
        prev_net_worth = self.net_worth
        prev_position = self.position
        
        # Ação: 0 = Manter, 1 = Comprar, 2 = Vender
        if action == 1 and self.position == 0:  # Comprar
            # Aplica slippage (preço de compra é maior)
            execution_price = current_price * (1 + self.slippage_pct)
            # Calcula comissão
            commission = execution_price * self.commission_pct
            # Quantidade que pode comprar
            shares = (self.balance - commission) / execution_price
            # Atualiza estado
            self.position = 1
            self.position_price = execution_price
            self.balance = 0  # Todo o saldo é usado para comprar
            self.net_worth = shares * current_price
            self.position_pnl = 0
            
        elif action == 2 and self.position == 1:  # Vender
            # Aplica slippage (preço de venda é menor)
            execution_price = current_price * (1 - self.slippage_pct)
            # Calcula comissão
            commission = execution_price * self.commission_pct
            # Calcula valor da venda
            shares = self.net_worth / current_price
            sale_value = shares * execution_price - commission
            # Atualiza estado
            self.position = 0
            self.balance = sale_value
            self.net_worth = sale_value
            self.position_pnl = (execution_price - self.position_price) / self.position_price
            self.position_price = 0
            
        else:  # Manter posição
            if self.position == 1:
                # Atualiza PnL da posição
                self.position_pnl = (current_price - self.position_price) / self.position_price
                # Atualiza valor líquido
                shares = self.net_worth / self.df.iloc[max(0, self.current_step - 1)]['close']
                self.net_worth = shares * current_price
        
        # Calcula retorno
        current_return = (self.net_worth - prev_net_worth) / prev_net_worth
        self.returns_history.append(current_return)
        
        # Calcula recompensa com base no tipo escolhido
        if self.reward_type == 'pnl':
            reward = current_return
        elif self.reward_type == 'sharpe':
            reward = self._calculate_sharpe_reward()
        elif self.reward_type == 'calmar':
            reward = self._calculate_calmar_reward()
        else:
            reward = current_return
        
        # Adiciona incentivo para operações (para evitar que o agente fique parado)
        # Pequeno bônus para mudar de posição
        if self.position != prev_position:
            reward += 0.001  # Pequeno incentivo para operações
            
        # Pequena penalidade para ficar parado por muito tempo
        if action == 0 and self.current_step > 20:
            last_20_actions = [h.get('action', 0) for h in self.history[-20:]]
            if sum(last_20_actions) == 0:  # Se ficou 20 passos sem fazer nada
                reward -= 0.0005  # Pequena penalidade
            
        return reward
        
    def _update_history(self):
        """Atualiza o histórico de trading."""
        self.history.append({
            'step': self.current_step,
            'net_worth': self.net_worth,
            'position': self.position,
            'price': self.df.iloc[self.current_step]['close'],
            'action': 1 if self.position > 0 else 0  # Registra a ação para análise
        })
    
    def _calculate_sharpe_reward(self):
        """
        Calcula a recompensa baseada no Sharpe Ratio.
        
        Returns:
            float: Recompensa baseada no Sharpe Ratio
        """
        if len(self.returns_history) < 2:
            return 0
            
        returns = np.array(self.returns_history)
        sharpe = returns.mean() / (returns.std() + 1e-9)  # Evita divisão por zero
        return sharpe
    
    def _calculate_calmar_reward(self):
        """
        Calcula a recompensa baseada no Calmar Ratio.
        
        Returns:
            float: Recompensa baseada no Calmar Ratio
        """
        if len(self.returns_history) < 2:
            return 0
            
        returns = np.array(self.returns_history)
        
        # Calcula o drawdown máximo
        cumulative_returns = np.cumprod(1 + returns) - 1
        peak = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - peak) / (peak + 1)
        max_drawdown = abs(drawdown.min()) + 1e-9  # Evita divisão por zero
        
        # Calcula o Calmar Ratio
        annual_return = returns.mean() * 252  # Anualiza o retorno
        calmar = annual_return / max_drawdown
        
        return calmar
    
    def _update_history(self):
        """Atualiza o histórico de trading."""
        self.history.append({
            'step': self.current_step,
            'net_worth': self.net_worth,
            'position': self.position,
            'price': self.df.iloc[self.current_step]['close']
        })
    
    def render(self, mode='human'):
        """
        Renderiza o ambiente.
        
        Args:
            mode (str): Modo de renderização
        """
        print(f"Step: {self.current_step}")
        print(f"Price: {self._get_current_price():.2f}")
        print(f"Position: {self.position}")
        print(f"Balance: {self.balance:.2f}")
        print(f"Net Worth: {self.net_worth:.2f}")
        print(f"PnL: {self.position_pnl:.2%}")
        print("-" * 50)