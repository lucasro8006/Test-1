"""
Estratégias Avançadas de Trading - Portfolio, Multi-timeframe, Risk Parity
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

class PortfolioOptimizer:
    """
    Otimizador de portfolio usando diferentes métodos
    """
    def __init__(self, method: str = 'markowitz'):
        self.method = method
        self.weights_history = []
        
    def optimize_portfolio(self, returns: pd.DataFrame, 
                          risk_aversion: float = 1.0,
                          constraints: Dict = None) -> np.ndarray:
        """
        Otimiza o portfolio baseado no método escolhido
        """
        if self.method == 'markowitz':
            return self._markowitz_optimization(returns, risk_aversion)
        elif self.method == 'risk_parity':
            return self._risk_parity_optimization(returns)
        elif self.method == 'black_litterman':
            return self._black_litterman_optimization(returns)
        elif self.method == 'hierarchical_risk_parity':
            return self._hierarchical_risk_parity(returns)
        else:
            return self._equal_weight(returns)
    
    def _markowitz_optimization(self, returns: pd.DataFrame, 
                               risk_aversion: float) -> np.ndarray:
        """
        Otimização de Markowitz (média-variância)
        """
        n_assets = len(returns.columns)
        mean_returns = returns.mean()
        cov_matrix = returns.cov()
        
        # Função objetivo: maximizar utilidade (retorno - risco)
        def objective(weights):
            portfolio_return = np.sum(weights * mean_returns)
            portfolio_risk = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            return -(portfolio_return - risk_aversion * portfolio_risk)
        
        # Restrições
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},  # Soma = 1
        ]
        
        # Limites (sem vendas a descoberto)
        bounds = tuple((0, 1) for _ in range(n_assets))
        
        # Otimização
        result = minimize(
            objective, 
            np.ones(n_assets) / n_assets,  # Peso inicial igual
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        return result.x if result.success else np.ones(n_assets) / n_assets
    
    def _risk_parity_optimization(self, returns: pd.DataFrame) -> np.ndarray:
        """
        Risk Parity - cada ativo contribui igualmente para o risco total
        """
        cov_matrix = returns.cov().values
        n_assets = len(returns.columns)
        
        def risk_budget_objective(weights):
            portfolio_vol = np.sqrt(np.dot(weights, np.dot(cov_matrix, weights)))
            marginal_contrib = np.dot(cov_matrix, weights) / portfolio_vol
            contrib = weights * marginal_contrib
            return np.sum((contrib - contrib.mean()) ** 2)
        
        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
        bounds = tuple((0.01, 0.99) for _ in range(n_assets))
        
        result = minimize(
            risk_budget_objective,
            np.ones(n_assets) / n_assets,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        return result.x if result.success else np.ones(n_assets) / n_assets
    
    def _black_litterman_optimization(self, returns: pd.DataFrame) -> np.ndarray:
        """
        Black-Litterman com views implícitas
        """
        # Implementação simplificada do Black-Litterman
        n_assets = len(returns.columns)
        
        # Prior (market cap weights - assumindo iguais por simplicidade)
        w_market = np.ones(n_assets) / n_assets
        
        # Parâmetros
        tau = 0.025  # Incerteza sobre o prior
        
        # Matriz de covariância
        sigma = returns.cov().values
        
        # Retornos implícitos (equilibrium returns)
        risk_aversion = 3.0
        pi = risk_aversion * np.dot(sigma, w_market)
        
        # Black-Litterman sem views específicas (volta ao prior)
        mu_bl = pi
        sigma_bl = sigma
        
        # Otimização
        inv_sigma = np.linalg.inv(sigma_bl)
        ones = np.ones((n_assets, 1))
        
        # Pesos ótimos
        w_opt = np.dot(inv_sigma, mu_bl) / (risk_aversion * np.dot(ones.T, np.dot(inv_sigma, ones)))
        
        return w_opt.flatten()
    
    def _hierarchical_risk_parity(self, returns: pd.DataFrame) -> np.ndarray:
        """
        Hierarchical Risk Parity (HRP)
        """
        try:
            from scipy.cluster.hierarchy import linkage, dendrogram
            from scipy.spatial.distance import squareform
            
            # Matriz de correlação
            corr_matrix = returns.corr()
            
            # Distância baseada na correlação
            distance_matrix = np.sqrt(0.5 * (1 - corr_matrix))
            
            # Clustering hierárquico
            condensed_distances = squareform(distance_matrix)
            linkage_matrix = linkage(condensed_distances, method='ward')
            
            # Função recursiva para calcular pesos HRP
            def _get_cluster_var(cov, cluster_items):
                cov_slice = cov.loc[cluster_items, cluster_items]
                inv_diag = 1 / np.diag(cov_slice)
                parity_w = inv_diag * (1 / np.sum(inv_diag))
                return np.dot(parity_w, np.dot(cov_slice, parity_w))
            
            def _get_rec_bipart(cov, sort_ix):
                w = pd.Series(1, index=sort_ix)
                cluster_items = [sort_ix]
                
                while len(cluster_items) > 0:
                    cluster_items = [i[j:k] for i in cluster_items 
                                   for j, k in ((0, len(i) // 2), (len(i) // 2, len(i))) 
                                   if len(i) > 1]
                    
                    for i in range(0, len(cluster_items), 2):
                        cluster0 = cluster_items[i]
                        cluster1 = cluster_items[i + 1]
                        
                        var0 = _get_cluster_var(cov, cluster0)
                        var1 = _get_cluster_var(cov, cluster1)
                        
                        alpha = 1 - var0 / (var0 + var1)
                        
                        w[cluster0] *= alpha
                        w[cluster1] *= 1 - alpha
                
                return w
            
            # Ordenação baseada no clustering
            sort_ix = returns.columns.tolist()  # Simplificado
            
            # Calcula pesos HRP
            cov_matrix = returns.cov()
            weights = _get_rec_bipart(cov_matrix, sort_ix)
            
            return weights.values
        except ImportError:
            print("Scipy não disponível, usando pesos iguais")
            return self._equal_weight(returns)
    
    def _equal_weight(self, returns: pd.DataFrame) -> np.ndarray:
        """
        Portfolio com pesos iguais
        """
        n_assets = len(returns.columns)
        return np.ones(n_assets) / n_assets

class MultiTimeframeStrategy:
    """
    Estratégia que opera em múltiplos timeframes
    """
    def __init__(self, timeframes: List[str] = ['1D', '1W', '1M']):
        self.timeframes = timeframes
        self.signals = {}
        self.weights = {'1D': 0.5, '1W': 0.3, '1M': 0.2}  # Pesos por timeframe
        
    def generate_signals(self, data: pd.DataFrame) -> Dict[str, int]:
        """
        Gera sinais para cada timeframe
        """
        signals = {}
        
        for tf in self.timeframes:
            # Resample data para o timeframe
            if tf == '1D':
                tf_data = data
            elif tf == '1W':
                tf_data = data.resample('W').agg({
                    'open': 'first',
                    'high': 'max',
                    'low': 'min',
                    'close': 'last',
                    'volume': 'sum'
                }).dropna()
            elif tf == '1M':
                tf_data = data.resample('M').agg({
                    'open': 'first',
                    'high': 'max',
                    'low': 'min',
                    'close': 'last',
                    'volume': 'sum'
                }).dropna()
            else:
                tf_data = data
            
            # Gera sinal para este timeframe
            signal = self._generate_timeframe_signal(tf_data)
            signals[tf] = signal
        
        return signals
    
    def _generate_timeframe_signal(self, data: pd.DataFrame) -> int:
        """
        Gera sinal para um timeframe específico
        """
        if len(data) < 20:
            return 0  # Hold
        
        # Estratégia simples: média móvel
        short_ma = data['close'].rolling(5).mean()
        long_ma = data['close'].rolling(20).mean()
        
        if short_ma.iloc[-1] > long_ma.iloc[-1]:
            return 1  # Buy
        elif short_ma.iloc[-1] < long_ma.iloc[-1]:
            return -1  # Sell
        else:
            return 0  # Hold
    
    def combine_signals(self, signals: Dict[str, int]) -> int:
        """
        Combina sinais de diferentes timeframes
        """
        weighted_signal = 0
        total_weight = 0
        
        for tf, signal in signals.items():
            weight = self.weights.get(tf, 0)
            weighted_signal += weight * signal
            total_weight += weight
        
        if total_weight > 0:
            final_signal = weighted_signal / total_weight
        else:
            final_signal = 0
        
        # Converte para ação discreta
        if final_signal > 0.3:
            return 1  # Buy
        elif final_signal < -0.3:
            return 2  # Sell
        else:
            return 0  # Hold

class HedgingStrategy:
    """
    Estratégia de hedging automático
    """
    def __init__(self, hedge_ratio: float = 0.5):
        self.hedge_ratio = hedge_ratio
        self.hedge_instruments = {}
        
    def calculate_hedge_ratio(self, primary_asset: pd.Series, 
                            hedge_asset: pd.Series) -> float:
        """
        Calcula o ratio de hedge ótimo usando regressão
        """
        # Calcula retornos
        primary_returns = primary_asset.pct_change().dropna()
        hedge_returns = hedge_asset.pct_change().dropna()
        
        # Alinha as séries
        aligned_data = pd.concat([primary_returns, hedge_returns], axis=1).dropna()
        
        if len(aligned_data) < 10:
            return self.hedge_ratio
        
        # Regressão linear
        X = aligned_data.iloc[:, 1].values.reshape(-1, 1)
        y = aligned_data.iloc[:, 0].values
        
        # Calcula beta (hedge ratio)
        covariance = np.cov(X.flatten(), y)[0, 1]
        variance = np.var(X.flatten())
        
        if variance > 0:
            beta = covariance / variance
            return -beta  # Negativo para hedge
        
        return self.hedge_ratio
    
    def generate_hedge_positions(self, portfolio_value: float,
                               primary_position: float) -> Dict[str, float]:
        """
        Gera posições de hedge
        """
        hedge_positions = {}
        
        # Hedge simples: posição oposta proporcional
        hedge_size = abs(primary_position) * self.hedge_ratio
        
        if primary_position > 0:  # Long position
            hedge_positions['hedge'] = -hedge_size  # Short hedge
        elif primary_position < 0:  # Short position
            hedge_positions['hedge'] = hedge_size  # Long hedge
        
        return hedge_positions

class FactorInvestingStrategy:
    """
    Estratégia baseada em fatores (value, momentum, quality, etc.)
    """
    def __init__(self, factors: List[str] = ['momentum', 'mean_reversion']):
        self.factors = factors
        self.factor_weights = {factor: 1.0/len(factors) for factor in factors}
        
    def calculate_factor_scores(self, data: pd.DataFrame) -> Dict[str, float]:
        """
        Calcula scores para cada fator
        """
        scores = {}
        
        if 'momentum' in self.factors:
            scores['momentum'] = self._momentum_score(data)
        
        if 'mean_reversion' in self.factors:
            scores['mean_reversion'] = self._mean_reversion_score(data)
        
        if 'volatility' in self.factors:
            scores['volatility'] = self._volatility_score(data)
        
        return scores
    
    def _momentum_score(self, data: pd.DataFrame) -> float:
        """
        Score de momentum (12-1 meses)
        """
        if len(data) < 252:  # Menos de 1 ano de dados
            return 0.0
        
        # Retorno dos últimos 12 meses excluindo o último mês
        start_price = data['close'].iloc[-252]
        end_price = data['close'].iloc[-21]  # Exclui último mês
        
        momentum = (end_price / start_price) - 1
        
        # Normaliza entre -1 e 1
        return np.tanh(momentum * 2)
    
    def _mean_reversion_score(self, data: pd.DataFrame) -> float:
        """
        Score de reversão à média
        """
        if len(data) < 50:
            return 0.0
        
        # Distância da média móvel de longo prazo
        long_ma = data['close'].rolling(50).mean().iloc[-1]
        current_price = data['close'].iloc[-1]
        
        deviation = (current_price - long_ma) / long_ma
        
        # Score negativo para reversão (comprar quando abaixo da média)
        return -np.tanh(deviation * 3)
    
    def _volatility_score(self, data: pd.DataFrame) -> float:
        """
        Score baseado na volatilidade
        """
        if len(data) < 21:
            return 0.0
        
        # Volatilidade realizada
        returns = data['close'].pct_change().dropna()
        volatility = returns.rolling(21).std().iloc[-1]
        
        # Score negativo para alta volatilidade
        return -np.tanh(volatility * 50)
    
    def generate_factor_signal(self, data: pd.DataFrame) -> float:
        """
        Gera sinal combinado baseado em fatores
        """
        scores = self.calculate_factor_scores(data)
        
        # Combina scores com pesos
        combined_score = 0
        total_weight = 0
        
        for factor, score in scores.items():
            weight = self.factor_weights.get(factor, 0)
            combined_score += weight * score
            total_weight += weight
        
        if total_weight > 0:
            return combined_score / total_weight
        
        return 0.0

class MarketMakingStrategy:
    """
    Estratégia de market making simplificada
    """
    def __init__(self, spread_target: float = 0.002):
        self.spread_target = spread_target
        self.inventory_limit = 0.1  # 10% do portfolio
        self.current_inventory = 0.0
        
    def calculate_optimal_quotes(self, mid_price: float, 
                               volatility: float,
                               inventory: float) -> Tuple[float, float]:
        """
        Calcula bid e ask ótimos
        """
        # Spread base ajustado pela volatilidade
        base_spread = self.spread_target * (1 + volatility * 10)
        
        # Ajuste por inventário (inventory skew)
        inventory_skew = inventory / self.inventory_limit * 0.001
        
        # Calcula bid e ask
        half_spread = base_spread / 2
        bid = mid_price - half_spread + inventory_skew
        ask = mid_price + half_spread + inventory_skew
        
        return bid, ask
    
    def should_make_market(self, market_conditions: Dict[str, float]) -> bool:
        """
        Decide se deve fazer market making baseado nas condições
        """
        # Não faz market making em alta volatilidade
        if market_conditions.get('volatility', 0) > 0.05:
            return False
        
        # Não faz market making se inventário muito alto
        if abs(self.current_inventory) > self.inventory_limit:
            return False
        
        # Verifica liquidez mínima
        if market_conditions.get('volume', 0) < 1000:
            return False
        
        return True
    
    def update_inventory(self, trade_size: float):
        """
        Atualiza inventário após trade
        """
        self.current_inventory += trade_size