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
        window_size=20,
        max_drawdown_penalty=0.5,
        transaction_cost_model='percentage',
        market_impact_model='linear',
        risk_free_rate=0.03,
        enable_fractional=True,
        max_position_size=1.0
    ):
        """
        Inicializa o ambiente de trading.
        
        Args:
            df (pd.DataFrame): DataFrame com dados de preços e features
            initial_balance (float): Saldo inicial
            commission_pct (float): Percentual de comissão por operação
            slippage_pct (float): Percentual de slippage por operação
            reward_type (str): Tipo de recompensa ('pnl', 'sharpe', 'calmar', 'sortino', 'information_ratio')
            window_size (int): Tamanho da janela para cálculo de métricas
            max_drawdown_penalty (float): Penalidade para drawdown máximo
            transaction_cost_model (str): Modelo de custo de transação ('percentage', 'fixed', 'tiered')
            market_impact_model (str): Modelo de impacto de mercado ('linear', 'square_root', 'none')
            risk_free_rate (float): Taxa livre de risco anualizada para cálculo de métricas
            enable_fractional (bool): Se True, permite posições fracionárias
            max_position_size (float): Tamanho máximo da posição como fração do saldo (0.0-1.0)
        """
        super(TradingEnv, self).__init__()
        
        self.df = df
        self.initial_balance = initial_balance
        self.commission_pct = commission_pct
        self.slippage_pct = slippage_pct
        self.reward_type = reward_type
        self.window_size = window_size
        self.max_drawdown_penalty = max_drawdown_penalty
        self.transaction_cost_model = transaction_cost_model
        self.market_impact_model = market_impact_model
        self.risk_free_rate = risk_free_rate
        self.enable_fractional = enable_fractional
        self.max_position_size = max_position_size
        
        # Configuração do ambiente
        self.config = {
            'initial_balance': initial_balance,
            'commission_pct': commission_pct,
            'slippage_pct': slippage_pct,
            'reward_type': reward_type,
            'window_size': window_size,
            'max_drawdown_penalty': max_drawdown_penalty,
            'transaction_cost_model': transaction_cost_model,
            'market_impact_model': market_impact_model,
            'risk_free_rate': risk_free_rate,
            'enable_fractional': enable_fractional,
            'max_position_size': max_position_size
        }
        
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
            # Calcula o preço de execução com slippage e impacto de mercado
            execution_price = self._calculate_execution_price(current_price, True)
            
            # Calcula custos de transação
            transaction_cost = self._calculate_transaction_cost(execution_price, self.balance)
            
            # Determina o tamanho da posição
            position_size = self.balance * self.max_position_size if self.enable_fractional else self.balance
            
            # Quantidade que pode comprar
            shares = (position_size - transaction_cost) / execution_price
            
            # Atualiza estado
            self.position = 1
            self.position_price = execution_price
            self.balance = self.balance - (shares * execution_price + transaction_cost)
            self.shares = shares
            self.net_worth = self.balance + (shares * current_price)
            self.position_pnl = 0
            
        elif action == 2 and self.position == 1:  # Vender
            # Calcula o preço de execução com slippage e impacto de mercado
            execution_price = self._calculate_execution_price(current_price, False)
            
            # Valor bruto da venda
            sale_value = self.shares * execution_price
            
            # Calcula custos de transação
            transaction_cost = self._calculate_transaction_cost(execution_price, sale_value)
            
            # Valor líquido da venda
            net_sale_value = sale_value - transaction_cost
            
            # Atualiza estado
            self.position = 0
            self.balance += net_sale_value
            self.net_worth = self.balance
            self.position_pnl = (execution_price - self.position_price) / self.position_price
            self.position_price = 0
            self.shares = 0
            
        else:  # Manter posição
            if self.position == 1:
                # Atualiza PnL da posição
                self.position_pnl = (current_price - self.position_price) / self.position_price
                # Atualiza valor líquido
                self.net_worth = self.balance + (self.shares * current_price)
        
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
        elif self.reward_type == 'sortino':
            reward = self._calculate_sortino_reward()
        elif self.reward_type == 'information_ratio':
            reward = self._calculate_information_ratio()
        else:
            reward = current_return
        
        # Penalidade para drawdown excessivo
        if len(self.returns_history) > self.window_size:
            max_drawdown = self._calculate_max_drawdown()
            if max_drawdown > 0.1:  # Penaliza drawdowns maiores que 10%
                reward -= max_drawdown * self.max_drawdown_penalty
        
        # Adiciona incentivo para operações (para evitar que o agente fique parado)
        # Pequeno bônus para mudar de posição
        if self.position != prev_position:
            reward += 0.001  # Pequeno incentivo para operações
            
        # Pequena penalidade para ficar parado por muito tempo
        if action == 0 and self.current_step > self.window_size:
            last_n_actions = [h.get('action', 0) for h in self.history[-self.window_size:]]
            if sum(last_n_actions) == 0:  # Se ficou window_size passos sem fazer nada
                reward -= 0.0005  # Pequena penalidade
            
        return reward
        
    def _calculate_execution_price(self, current_price, is_buy):
        """
        Calcula o preço de execução considerando slippage e impacto de mercado.
        
        Args:
            current_price (float): Preço atual
            is_buy (bool): Se True, é uma compra; se False, é uma venda
            
        Returns:
            float: Preço de execução
        """
        # Aplica slippage básico
        if is_buy:
            execution_price = current_price * (1 + self.slippage_pct)
        else:
            execution_price = current_price * (1 - self.slippage_pct)
            
        # Aplica impacto de mercado se configurado
        if self.market_impact_model != 'none':
            # Estima o volume como uma fração do volume médio diário
            # Aqui estamos usando um valor fixo para simplificar
            volume_fraction = 0.01
            
            # Calcula o impacto de mercado
            if self.market_impact_model == 'linear':
                # Modelo linear: impacto proporcional ao volume
                impact = volume_fraction * 0.1  # 0.1% de impacto para cada 1% do volume
            elif self.market_impact_model == 'square_root':
                # Modelo raiz quadrada: impacto proporcional à raiz do volume
                impact = np.sqrt(volume_fraction) * 0.1
            else:
                impact = 0
                
            # Aplica o impacto ao preço
            if is_buy:
                execution_price *= (1 + impact)
            else:
                execution_price *= (1 - impact)
                
        return execution_price
        
    def _calculate_transaction_cost(self, price, value):
        """
        Calcula o custo de transação com base no modelo escolhido.
        
        Args:
            price (float): Preço de execução
            value (float): Valor da transação
            
        Returns:
            float: Custo de transação
        """
        if self.transaction_cost_model == 'percentage':
            # Modelo percentual: custo é uma porcentagem do valor
            return value * self.commission_pct
        elif self.transaction_cost_model == 'fixed':
            # Modelo fixo: custo é um valor fixo por operação
            return min(10.0, value * 0.01)  # Máximo de $10 ou 1% do valor
        elif self.transaction_cost_model == 'tiered':
            # Modelo em camadas: custo varia com o valor
            if value < 1000:
                return value * 0.002  # 0.2% para valores pequenos
            elif value < 10000:
                return value * 0.001  # 0.1% para valores médios
            else:
                return value * 0.0005  # 0.05% para valores grandes
        else:
            # Modelo padrão
            return value * self.commission_pct
        
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
            
        returns = np.array(self.returns_history[-self.window_size:])
        
        # Calcula o drawdown máximo
        cumulative_returns = np.cumprod(1 + returns) - 1
        peak = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - peak) / (peak + 1)
        max_drawdown = abs(drawdown.min()) + 1e-9  # Evita divisão por zero
        
        # Calcula o Calmar Ratio
        annual_return = returns.mean() * 252  # Anualiza o retorno
        calmar = annual_return / max_drawdown
        
        return calmar
        
    def _calculate_sortino_reward(self):
        """
        Calcula a recompensa baseada no Sortino Ratio.
        
        Returns:
            float: Recompensa baseada no Sortino Ratio
        """
        if len(self.returns_history) < 2:
            return 0
            
        returns = np.array(self.returns_history[-self.window_size:])
        
        # Calcula o retorno médio
        mean_return = returns.mean()
        
        # Calcula o desvio padrão dos retornos negativos
        negative_returns = returns[returns < 0]
        if len(negative_returns) == 0:
            downside_std = 1e-9  # Evita divisão por zero
        else:
            downside_std = np.std(negative_returns) + 1e-9
            
        # Calcula o Sortino Ratio
        sortino = (mean_return - self.risk_free_rate / 252) / downside_std
        
        return sortino
        
    def _calculate_information_ratio(self):
        """
        Calcula a recompensa baseada no Information Ratio.
        
        Returns:
            float: Recompensa baseada no Information Ratio
        """
        if len(self.returns_history) < 2:
            return 0
            
        # Obtém os retornos do agente
        agent_returns = np.array(self.returns_history[-self.window_size:])
        
        # Obtém os retornos do benchmark (aqui usamos o próprio ativo)
        benchmark_returns = []
        end_idx = min(self.current_step, len(self.df) - 1)
        start_idx = max(0, end_idx - self.window_size + 1)
        
        for i in range(start_idx, end_idx + 1):
            if i > 0:
                benchmark_return = (self.df.iloc[i]['close'] / self.df.iloc[i-1]['close']) - 1
                benchmark_returns.append(benchmark_return)
                
        if len(benchmark_returns) < 2:
            return 0
            
        benchmark_returns = np.array(benchmark_returns)
        
        # Calcula o retorno em excesso
        excess_returns = agent_returns - benchmark_returns
        
        # Calcula o Information Ratio
        mean_excess = excess_returns.mean()
        std_excess = excess_returns.std() + 1e-9  # Evita divisão por zero
        
        information_ratio = mean_excess / std_excess
        
        return information_ratio
        
    def _calculate_max_drawdown(self):
        """
        Calcula o drawdown máximo.
        
        Returns:
            float: Drawdown máximo
        """
        if len(self.returns_history) < 2:
            return 0
            
        returns = np.array(self.returns_history)
        
        # Calcula o drawdown máximo
        cumulative_returns = np.cumprod(1 + returns) - 1
        peak = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - peak) / (peak + 1)
        max_drawdown = abs(drawdown.min())
        
        return max_drawdown
    
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