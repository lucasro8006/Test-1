"""
Sistema Avançado de Gestão de Risco
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from scipy import stats
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

class VaRCalculator:
    """
    Calculador de Value at Risk (VaR) e Conditional VaR (CVaR)
    """
    def __init__(self, confidence_level: float = 0.95):
        self.confidence_level = confidence_level
        self.alpha = 1 - confidence_level
        
    def calculate_parametric_var(self, returns: pd.Series, 
                                portfolio_value: float) -> float:
        """
        Calcula VaR paramétrico (assumindo distribuição normal)
        """
        mean_return = returns.mean()
        std_return = returns.std()
        
        # Z-score para o nível de confiança
        z_score = stats.norm.ppf(self.alpha)
        
        # VaR = -(média + z_score * desvio padrão) * valor do portfolio
        var = -(mean_return + z_score * std_return) * portfolio_value
        
        return var
    
    def calculate_historical_var(self, returns: pd.Series, 
                               portfolio_value: float) -> float:
        """
        Calcula VaR histórico
        """
        # Ordena retornos em ordem crescente
        sorted_returns = returns.sort_values()
        
        # Encontra o percentil correspondente ao nível de confiança
        percentile_index = int(len(sorted_returns) * self.alpha)
        var_return = sorted_returns.iloc[percentile_index]
        
        # VaR = -percentil * valor do portfolio
        var = -var_return * portfolio_value
        
        return var
    
    def calculate_monte_carlo_var(self, returns: pd.Series, 
                                portfolio_value: float,
                                num_simulations: int = 10000) -> float:
        """
        Calcula VaR usando simulação Monte Carlo
        """
        mean_return = returns.mean()
        std_return = returns.std()
        
        # Gera simulações
        simulated_returns = np.random.normal(mean_return, std_return, num_simulations)
        
        # Calcula VaR
        var_return = np.percentile(simulated_returns, self.alpha * 100)
        var = -var_return * portfolio_value
        
        return var
    
    def calculate_cvar(self, returns: pd.Series, 
                      portfolio_value: float) -> float:
        """
        Calcula Conditional VaR (Expected Shortfall)
        """
        # Ordena retornos
        sorted_returns = returns.sort_values()
        
        # Encontra o percentil do VaR
        percentile_index = int(len(sorted_returns) * self.alpha)
        
        # CVaR é a média dos retornos piores que o VaR
        tail_returns = sorted_returns.iloc[:percentile_index]
        cvar_return = tail_returns.mean()
        
        # CVaR = -média da cauda * valor do portfolio
        cvar = -cvar_return * portfolio_value
        
        return cvar

class StressTesting:
    """
    Sistema de stress testing para portfolios
    """
    def __init__(self):
        self.stress_scenarios = {}
        
    def add_scenario(self, name: str, scenario: Dict[str, float]):
        """
        Adiciona um cenário de stress
        """
        self.stress_scenarios[name] = scenario
        
    def create_market_crash_scenario(self) -> Dict[str, float]:
        """
        Cria cenário de crash do mercado
        """
        return {
            'equity_shock': -0.30,  # Queda de 30% nas ações
            'volatility_shock': 2.0,  # Dobra a volatilidade
            'correlation_shock': 0.8,  # Correlações aumentam para 0.8
            'liquidity_shock': 0.5   # Reduz liquidez pela metade
        }
    
    def run_stress_test(self, portfolio_returns: pd.Series,
                       scenario_name: str) -> Dict[str, float]:
        """
        Executa teste de stress em um portfolio
        """
        if scenario_name not in self.stress_scenarios:
            raise ValueError(f"Cenário {scenario_name} não encontrado")
        
        scenario = self.stress_scenarios[scenario_name]
        results = {}
        
        # Aplica choques aos retornos
        stressed_returns = portfolio_returns.copy()
        
        if 'equity_shock' in scenario:
            # Aplica choque único
            shock_return = scenario['equity_shock']
            stressed_returns.iloc[-1] += shock_return
            
        if 'volatility_shock' in scenario:
            # Aumenta volatilidade
            vol_multiplier = scenario['volatility_shock']
            mean_return = stressed_returns.mean()
            stressed_returns = mean_return + (stressed_returns - mean_return) * vol_multiplier
        
        # Calcula métricas de risco no cenário estressado
        var_calc = VaRCalculator()
        
        results['stressed_var'] = var_calc.calculate_historical_var(stressed_returns, 100000)
        results['stressed_cvar'] = var_calc.calculate_cvar(stressed_returns, 100000)
        results['max_drawdown'] = self._calculate_max_drawdown(stressed_returns)
        results['volatility'] = stressed_returns.std() * np.sqrt(252)
        
        return results
    
    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """
        Calcula o máximo drawdown
        """
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        
        return drawdown.min()

class KellyCriterion:
    """
    Implementação do Kelly Criterion para position sizing
    """
    def __init__(self):
        self.historical_trades = []
        
    def calculate_kelly_fraction(self, win_rate: float, 
                                avg_win: float, 
                                avg_loss: float) -> float:
        """
        Calcula a fração Kelly ótima
        """
        if avg_loss == 0:
            return 0
        
        # Fórmula Kelly: f = (bp - q) / b
        # onde b = avg_win/avg_loss, p = win_rate, q = 1-win_rate
        b = avg_win / abs(avg_loss)
        p = win_rate
        q = 1 - win_rate
        
        kelly_fraction = (b * p - q) / b
        
        # Limita a fração Kelly para evitar over-leverage
        return max(0, min(kelly_fraction, 0.25))  # Máximo 25%
    
    def estimate_kelly_from_returns(self, returns: pd.Series) -> float:
        """
        Estima Kelly a partir de série de retornos
        """
        positive_returns = returns[returns > 0]
        negative_returns = returns[returns < 0]
        
        if len(positive_returns) == 0 or len(negative_returns) == 0:
            return 0
        
        win_rate = len(positive_returns) / len(returns)
        avg_win = positive_returns.mean()
        avg_loss = negative_returns.mean()
        
        return self.calculate_kelly_fraction(win_rate, avg_win, avg_loss)

class AnomalyDetector:
    """
    Detector de anomalias em tempo real para trading
    """
    def __init__(self, window_size: int = 50):
        self.window_size = window_size
        self.price_history = []
        self.volume_history = []
        self.return_history = []
        
    def add_observation(self, price: float, volume: float):
        """
        Adiciona nova observação
        """
        if len(self.price_history) > 0:
            return_val = (price - self.price_history[-1]) / self.price_history[-1]
            self.return_history.append(return_val)
        
        self.price_history.append(price)
        self.volume_history.append(volume)
        
        # Mantém apenas a janela especificada
        if len(self.price_history) > self.window_size:
            self.price_history.pop(0)
            self.volume_history.pop(0)
            self.return_history.pop(0)
    
    def detect_price_anomaly(self, threshold: float = 3.0) -> bool:
        """
        Detecta anomalias de preço usando z-score
        """
        if len(self.return_history) < 10:
            return False
        
        recent_returns = np.array(self.return_history[-10:])
        historical_returns = np.array(self.return_history[:-10])
        
        if len(historical_returns) == 0:
            return False
        
        # Calcula z-score do retorno mais recente
        mean_return = historical_returns.mean()
        std_return = historical_returns.std()
        
        if std_return == 0:
            return False
        
        z_score = abs((recent_returns[-1] - mean_return) / std_return)
        
        return z_score > threshold
    
    def detect_volume_anomaly(self, threshold: float = 2.5) -> bool:
        """
        Detecta anomalias de volume
        """
        if len(self.volume_history) < 10:
            return False
        
        recent_volume = self.volume_history[-1]
        historical_volumes = np.array(self.volume_history[:-1])
        
        # Calcula z-score do volume
        mean_volume = historical_volumes.mean()
        std_volume = historical_volumes.std()
        
        if std_volume == 0:
            return False
        
        z_score = abs((recent_volume - mean_volume) / std_volume)
        
        return z_score > threshold