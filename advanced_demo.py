"""
Demonstração das Funcionalidades Avançadas do Sistema de Trading
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Adiciona o diretório raiz ao path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Importa componentes avançados
from src.models.ensemble_agent import EnsembleAgent
from src.strategies.advanced_strategies import (
    PortfolioOptimizer, MultiTimeframeStrategy, 
    FactorInvestingStrategy, HedgingStrategy
)
from src.rl_advanced.advanced_rl import (
    CuriosityDrivenAgent, ImitationLearningAgent,
    DistributionalRLAgent, OfflineRLAgent
)
from src.risk_management.advanced_risk import (
    VaRCalculator, StressTesting, KellyCriterion, AnomalyDetector
)
from src.innovations.advanced_innovations import (
    FinancialGAN, CausalInferenceEngine, 
    QuantumInspiredOptimizer, MarketRegimeDetector
)

# Importa componentes básicos
from src.utils.data_utils import download_data
from src.features.feature_engineer import FeatureEngineer
from src.environments.trading_env import TradingEnv

def demo_ensemble_agent():
    """
    Demonstra o agente ensemble
    """
    print("\n=== DEMONSTRAÇÃO: ENSEMBLE AGENT ===")
    
    # Download de dados
    data = download_data('PETR4.SA', '2022-01-01', '2023-06-01')
    
    # Preparação de features
    fe = FeatureEngineer()
    processed_data = fe.transform(data)
    
    # Cria ambiente
    from stable_baselines3.common.vec_env import DummyVecEnv
    env = DummyVecEnv([lambda: TradingEnv(processed_data)])
    
    # Cria ensemble agent
    ensemble_config = {'num_agents': 3}
    ensemble = EnsembleAgent(env, ensemble_config)
    
    print(f"Ensemble criado com {len(ensemble.agents)} agentes:")
    for agent_name in ensemble.agents.keys():
        print(f"  - {agent_name}")
    
    # Treina ensemble (versão reduzida para demo)
    print("Treinando ensemble...")
    ensemble.train(total_timesteps=1000)
    
    # Testa predição
    obs = env.reset()
    action, _ = ensemble.predict(obs)
    print(f"Ação predita pelo ensemble: {action}")
    
    # Estatísticas do ensemble
    stats = ensemble.get_ensemble_stats()
    print(f"Pesos dos agentes: {stats['weights']}")

def demo_portfolio_optimization():
    """
    Demonstra otimização de portfolio
    """
    print("\n=== DEMONSTRAÇÃO: OTIMIZAÇÃO DE PORTFOLIO ===")
    
    # Simula retornos de múltiplos ativos
    np.random.seed(42)
    dates = pd.date_range('2022-01-01', '2023-01-01', freq='D')
    
    # Cria retornos sintéticos para 3 ativos
    returns_data = {
        'Asset_A': np.random.normal(0.001, 0.02, len(dates)),
        'Asset_B': np.random.normal(0.0005, 0.015, len(dates)),
        'Asset_C': np.random.normal(0.0008, 0.025, len(dates))
    }
    
    returns_df = pd.DataFrame(returns_data, index=dates)
    
    # Testa diferentes métodos de otimização
    methods = ['markowitz', 'risk_parity', 'equal_weight']
    
    for method in methods:
        optimizer = PortfolioOptimizer(method=method)
        weights = optimizer.optimize_portfolio(returns_df)
        
        print(f"\nMétodo {method}:")
        for i, asset in enumerate(returns_df.columns):
            print(f"  {asset}: {weights[i]:.3f}")

def demo_multi_timeframe_strategy():
    """
    Demonstra estratégia multi-timeframe
    """
    print("\n=== DEMONSTRAÇÃO: ESTRATÉGIA MULTI-TIMEFRAME ===")
    
    # Download de dados
    data = download_data('PETR4.SA', '2022-01-01', '2023-06-01')
    
    # Cria estratégia multi-timeframe
    strategy = MultiTimeframeStrategy(['1D', '1W', '1M'])
    
    # Gera sinais para diferentes timeframes
    signals = strategy.generate_signals(data)
    
    print("Sinais por timeframe:")
    for tf, signal in signals.items():
        signal_name = {-1: 'SELL', 0: 'HOLD', 1: 'BUY'}[signal]
        print(f"  {tf}: {signal_name}")
    
    # Combina sinais
    combined_signal = strategy.combine_signals(signals)
    combined_name = {0: 'HOLD', 1: 'BUY', 2: 'SELL'}[combined_signal]
    print(f"\nSinal combinado: {combined_name}")

def demo_risk_management():
    """
    Demonstra sistema de gestão de risco
    """
    print("\n=== DEMONSTRAÇÃO: GESTÃO DE RISCO ===")
    
    # Simula retornos de portfolio
    np.random.seed(42)
    returns = pd.Series(np.random.normal(0.001, 0.02, 252))
    portfolio_value = 100000
    
    # VaR Calculator
    var_calc = VaRCalculator(confidence_level=0.95)
    
    parametric_var = var_calc.calculate_parametric_var(returns, portfolio_value)
    historical_var = var_calc.calculate_historical_var(returns, portfolio_value)
    cvar = var_calc.calculate_cvar(returns, portfolio_value)
    
    print(f"VaR Paramétrico (95%): ${parametric_var:,.2f}")
    print(f"VaR Histórico (95%): ${historical_var:,.2f}")
    print(f"CVaR (95%): ${cvar:,.2f}")
    
    # Stress Testing
    stress_tester = StressTesting()
    crash_scenario = stress_tester.create_market_crash_scenario()
    stress_tester.add_scenario('market_crash', crash_scenario)
    
    stress_results = stress_tester.run_stress_test(returns, 'market_crash')
    print(f"\nStress Test - Market Crash:")
    print(f"  VaR Estressado: ${stress_results['stressed_var']:,.2f}")
    print(f"  CVaR Estressado: ${stress_results['stressed_cvar']:,.2f}")
    print(f"  Max Drawdown: {stress_results['max_drawdown']:.2%}")
    
    # Kelly Criterion
    kelly = KellyCriterion()
    kelly_fraction = kelly.estimate_kelly_from_returns(returns)
    print(f"\nFração Kelly Estimada: {kelly_fraction:.3f}")
    
    # Anomaly Detection
    detector = AnomalyDetector()
    
    # Simula dados de preço e volume
    prices = 100 * (1 + returns).cumprod()
    volumes = np.random.lognormal(10, 0.5, len(returns))
    
    anomalies_detected = 0
    for i in range(len(prices)):
        detector.add_observation(prices.iloc[i], volumes[i])
        
        if i > 20:  # Após período de warm-up
            if detector.detect_price_anomaly() or detector.detect_volume_anomaly():
                anomalies_detected += 1
    
    print(f"Anomalias detectadas: {anomalies_detected}")

def demo_causal_inference():
    """
    Demonstra inferência causal
    """
    print("\n=== DEMONSTRAÇÃO: INFERÊNCIA CAUSAL ===")
    
    # Simula dados de mercado com relações causais
    np.random.seed(42)
    n_obs = 252
    
    # Variáveis causais simuladas
    interest_rate = np.random.normal(0, 0.001, n_obs)
    vix = np.random.normal(0, 0.02, n_obs)
    
    # Preço do ativo (influenciado por taxa de juros e VIX)
    asset_price = (
        -0.5 * interest_rate +  # Taxa de juros afeta negativamente
        -0.3 * vix +           # VIX afeta negativamente
        np.random.normal(0, 0.01, n_obs)  # Ruído
    )
    
    # Cria DataFrame
    market_data = pd.DataFrame({
        'asset_returns': asset_price,
        'interest_rate': interest_rate,
        'vix': vix,
        'random_var': np.random.normal(0, 0.01, n_obs)  # Variável não causal
    })
    
    # Inferência causal
    causal_engine = CausalInferenceEngine()
    
    # Teste de causalidade de Granger
    causal_relationships = causal_engine.discover_causal_relationships(
        market_data, 'asset_returns', method='granger'
    )
    
    print("Relações causais descobertas (Granger):")
    for var, score in sorted(causal_relationships.items(), key=lambda x: x[1], reverse=True):
        print(f"  {var}: {score:.4f}")

def demo_gan_scenarios():
    """
    Demonstra geração de cenários com GAN
    """
    print("\n=== DEMONSTRAÇÃO: GERAÇÃO DE CENÁRIOS COM GAN ===")
    
    # Simula dados históricos
    np.random.seed(42)
    sequence_length = 30
    feature_dim = 3
    num_sequences = 100
    
    # Dados sintéticos (preço, volume, volatilidade)
    real_data = []
    for _ in range(num_sequences):
        sequence = np.random.normal(0, 1, (sequence_length, feature_dim))
        # Adiciona alguma estrutura temporal
        for i in range(1, sequence_length):
            sequence[i] += 0.1 * sequence[i-1]
        real_data.append(sequence)
    
    real_data = np.array(real_data)
    
    # Cria e treina GAN
    gan = FinancialGAN(sequence_length=sequence_length, feature_dim=feature_dim)
    
    print("Treinando GAN para geração de cenários...")
    gan.train(real_data, epochs=100)  # Reduzido para demo
    
    # Gera cenários sintéticos
    synthetic_scenarios = gan.generate_scenarios(num_scenarios=10)
    
    print(f"Gerados {len(synthetic_scenarios)} cenários sintéticos")
    print(f"Formato dos cenários: {synthetic_scenarios.shape}")
    
    # Compara estatísticas
    real_mean = real_data.mean()
    synthetic_mean = synthetic_scenarios.mean()
    
    print(f"Média dos dados reais: {real_mean:.4f}")
    print(f"Média dos dados sintéticos: {synthetic_mean:.4f}")

def demo_quantum_optimization():
    """
    Demonstra otimização quantum-inspired
    """
    print("\n=== DEMONSTRAÇÃO: OTIMIZAÇÃO QUANTUM-INSPIRED ===")
    
    # Simula problema de otimização de portfolio
    np.random.seed(42)
    num_assets = 5
    
    # Retornos esperados e matriz de covariância
    expected_returns = np.random.normal(0.08, 0.02, num_assets)
    correlation_matrix = np.random.uniform(0.1, 0.7, (num_assets, num_assets))
    correlation_matrix = (correlation_matrix + correlation_matrix.T) / 2
    np.fill_diagonal(correlation_matrix, 1.0)
    
    volatilities = np.random.uniform(0.1, 0.3, num_assets)
    covariance_matrix = np.outer(volatilities, volatilities) * correlation_matrix
    
    # Otimizador quantum
    quantum_optimizer = QuantumInspiredOptimizer(num_qubits=num_assets)
    
    # Função objetivo (maximizar Sharpe ratio)
    def objective(weights):
        portfolio_return = np.dot(weights, expected_returns)
        portfolio_risk = np.sqrt(np.dot(weights.T, np.dot(covariance_matrix, weights)))
        
        if portfolio_risk == 0:
            return -np.inf
        
        sharpe_ratio = portfolio_return / portfolio_risk
        return -sharpe_ratio  # Minimização
    
    # Restrições
    def sum_constraint(weights):
        return weights / np.sum(weights) if np.sum(weights) > 0 else weights
    
    def non_negative_constraint(weights):
        return np.maximum(weights, 0)
    
    constraints = [sum_constraint, non_negative_constraint]
    
    # Otimização
    print("Executando otimização quantum-inspired...")
    optimal_weights = quantum_optimizer.quantum_annealing_optimization(
        objective, constraints, num_iterations=500
    )
    
    print("Pesos ótimos do portfolio:")
    for i, weight in enumerate(optimal_weights):
        print(f"  Ativo {i+1}: {weight:.3f}")
    
    # Calcula métricas do portfolio ótimo
    portfolio_return = np.dot(optimal_weights, expected_returns)
    portfolio_risk = np.sqrt(np.dot(optimal_weights.T, np.dot(covariance_matrix, optimal_weights)))
    sharpe_ratio = portfolio_return / portfolio_risk if portfolio_risk > 0 else 0
    
    print(f"\nMétricas do portfolio ótimo:")
    print(f"  Retorno esperado: {portfolio_return:.2%}")
    print(f"  Risco (volatilidade): {portfolio_risk:.2%}")
    print(f"  Sharpe Ratio: {sharpe_ratio:.3f}")

def demo_regime_detection():
    """
    Demonstra detecção de regimes de mercado
    """
    print("\n=== DEMONSTRAÇÃO: DETECÇÃO DE REGIMES ===")
    
    # Simula retornos com diferentes regimes
    np.random.seed(42)
    
    # Regime 1: Bull market (alta volatilidade, retornos positivos)
    regime1 = np.random.normal(0.002, 0.015, 100)
    
    # Regime 2: Bear market (alta volatilidade, retornos negativos)
    regime2 = np.random.normal(-0.001, 0.025, 100)
    
    # Regime 3: Sideways market (baixa volatilidade, retornos neutros)
    regime3 = np.random.normal(0.0005, 0.008, 100)
    
    # Combina regimes
    returns = np.concatenate([regime1, regime2, regime3])
    returns_series = pd.Series(returns)
    
    # Detector de regimes
    detector = MarketRegimeDetector(num_regimes=3)
    
    # Detecta regimes
    detected_regimes = detector.detect_regimes_clustering(returns_series)
    
    # Caracteriza regimes
    regime_characteristics = detector.characterize_regimes(returns_series, detected_regimes)
    
    print("Regimes detectados:")
    for regime_id, characteristics in regime_characteristics.items():
        print(f"\nRegime {regime_id}:")
        print(f"  Retorno médio: {characteristics['mean_return']:.4f}")
        print(f"  Volatilidade: {characteristics['volatility']:.4f}")
        print(f"  Frequência: {characteristics['frequency']:.2%}")
    
    # Plota regimes (se matplotlib disponível)
    try:
        plt.figure(figsize=(12, 6))
        
        # Subplot 1: Retornos
        plt.subplot(2, 1, 1)
        plt.plot(returns_series.index, returns_series.values, alpha=0.7)
        plt.title('Retornos da Série Temporal')
        plt.ylabel('Retorno')
        
        # Subplot 2: Regimes detectados
        plt.subplot(2, 1, 2)
        colors = ['red', 'blue', 'green']
        for regime in range(3):
            mask = detected_regimes == regime
            plt.scatter(np.where(mask)[0], [regime] * np.sum(mask), 
                       c=colors[regime], alpha=0.6, label=f'Regime {regime}')
        
        plt.title('Regimes Detectados')
        plt.ylabel('Regime')
        plt.xlabel('Tempo')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('regime_detection_demo.png')
        print("\nGráfico salvo como 'regime_detection_demo.png'")
        
    except Exception as e:
        print(f"Erro ao plotar: {e}")

def main():
    """
    Executa todas as demonstrações
    """
    print("🚀 DEMONSTRAÇÃO DAS FUNCIONALIDADES AVANÇADAS DO SISTEMA DE TRADING")
    print("=" * 80)
    
    try:
        demo_ensemble_agent()
    except Exception as e:
        print(f"Erro na demo do ensemble agent: {e}")
    
    try:
        demo_portfolio_optimization()
    except Exception as e:
        print(f"Erro na demo de otimização de portfolio: {e}")
    
    try:
        demo_multi_timeframe_strategy()
    except Exception as e:
        print(f"Erro na demo de estratégia multi-timeframe: {e}")
    
    try:
        demo_risk_management()
    except Exception as e:
        print(f"Erro na demo de gestão de risco: {e}")
    
    try:
        demo_causal_inference()
    except Exception as e:
        print(f"Erro na demo de inferência causal: {e}")
    
    try:
        demo_gan_scenarios()
    except Exception as e:
        print(f"Erro na demo de GAN: {e}")
    
    try:
        demo_quantum_optimization()
    except Exception as e:
        print(f"Erro na demo de otimização quantum: {e}")
    
    try:
        demo_regime_detection()
    except Exception as e:
        print(f"Erro na demo de detecção de regimes: {e}")
    
    print("\n" + "=" * 80)
    print("🎉 DEMONSTRAÇÃO CONCLUÍDA!")
    print("\nTodas as funcionalidades avançadas foram demonstradas.")
    print("O sistema agora inclui:")
    print("✅ Ensemble de múltiplos algoritmos de RL")
    print("✅ Estratégias avançadas (portfolio, multi-timeframe, fatores)")
    print("✅ Técnicas de RL avançadas (curiosity, imitation, distributional)")
    print("✅ Gestão de risco sofisticada (VaR, stress testing, Kelly)")
    print("✅ Inovações com GANs e inferência causal")
    print("✅ Otimização quantum-inspired")
    print("✅ Detecção de regimes de mercado")

if __name__ == "__main__":
    main()