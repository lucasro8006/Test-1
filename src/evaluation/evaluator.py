"""
Módulo para avaliação de agentes de trading
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import seaborn as sns

# Tenta importar SHAP, mas não falha se não estiver disponível
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False


def evaluate_agent(agent, test_env, test_data):
    """
    Avalia o desempenho de um agente em um ambiente de teste.
    
    Args:
        agent: Agente treinado
        test_env: Ambiente de teste
        test_data (pd.DataFrame): Dados de teste
        
    Returns:
        tuple: DataFrame com histórico de trading e dicionário de métricas
    """
    # Reset do ambiente
    obs, _ = test_env.reset()
    done = False
    
    # Executa o agente no ambiente
    while not done:
        # Reshape da observação para o modelo
        obs_reshaped = np.array([obs])
        action, _ = agent.predict(obs_reshaped, deterministic=True)
        obs, reward, terminated, truncated, _ = test_env.step(action[0])
        done = terminated or truncated
    
    # Obtém o histórico
    history = test_env.history
    print(f"Histórico de trading: {len(history)} passos")
    
    if not history:
        print("AVISO: Nenhum histórico de trading foi registrado!")
        return pd.DataFrame(), {}
    
    # Cria DataFrame com o histórico
    history_df = pd.DataFrame(history)
    
    # Verifica se o índice do DataFrame de teste é compatível
    if len(history_df) <= len(test_data):
        history_df.index = test_data.index[:len(history_df)]
    else:
        print(f"AVISO: Tamanho do histórico ({len(history_df)}) maior que dados de teste ({len(test_data)})")
        # Usa índice numérico
        history_df.index = range(len(history_df))
    
    # Imprime informações de debug
    if len(history_df) > 0:
        print(f"History length: {len(history_df)}")
        print(f"First history entry: {history[0]}")
    
    # Calcula métricas
    metrics = calculate_metrics(history_df)
    print_metrics(metrics)
    
    # Plota resultados
    try:
        plot_performance(history_df, test_data, test_env.initial_balance)
    except Exception as e:
        print(f"Erro ao plotar resultados: {e}")
    
    return history_df, metrics


def calculate_metrics(history_df):
    """
    Calcula métricas de desempenho a partir do histórico de trading.
    
    Args:
        history_df (pd.DataFrame): DataFrame com histórico de trading
        
    Returns:
        dict: Dicionário com métricas calculadas
    """
    metrics = {}
    
    if 'net_worth' not in history_df.columns:
        print("Aviso: Coluna 'net_worth' não encontrada no histórico")
        return metrics
    
    # Retornos diários
    returns = history_df['net_worth'].pct_change().fillna(0)
    
    # Retorno total
    metrics['total_return'] = (history_df['net_worth'].iloc[-1] / history_df['net_worth'].iloc[0] - 1) * 100
    
    # Sharpe Ratio (anualizado)
    metrics['sharpe_ratio'] = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0
    
    # Drawdown
    peak = history_df['net_worth'].cummax()
    drawdown = (history_df['net_worth'] - peak) / peak
    metrics['max_drawdown'] = drawdown.min() * 100
    
    # Calmar Ratio
    annual_return = returns.mean() * 252
    metrics['calmar_ratio'] = annual_return / (abs(metrics['max_drawdown']) / 100) if metrics['max_drawdown'] != 0 else 0
    
    # Número de operações
    if 'position' in history_df.columns:
        position_changes = history_df['position'].diff().fillna(0)
        metrics['n_trades'] = (position_changes != 0).sum()
        
        # Operações lucrativas
        buys = history_df[history_df['position'] > history_df['position'].shift(1).fillna(0)]
        sells = history_df[history_df['position'] < history_df['position'].shift(1).fillna(0)]
        
        if not buys.empty and not sells.empty:
            # Mapeia compras e vendas para calcular P&L por operação
            trades = []
            buy_price = None
            
            for idx, row in history_df.iterrows():
                if row['position'] > history_df.loc[idx:idx].shift(1).fillna(0)['position'].iloc[0]:
                    buy_price = row['price']
                elif row['position'] < history_df.loc[idx:idx].shift(1).fillna(0)['position'].iloc[0] and buy_price is not None:
                    sell_price = row['price']
                    pnl = (sell_price / buy_price - 1) * 100
                    trades.append(pnl)
                    buy_price = None
            
            if trades:
                metrics['win_rate'] = (np.array(trades) > 0).mean() * 100
                metrics['avg_profit'] = np.mean([t for t in trades if t > 0]) if any(t > 0 for t in trades) else 0
                metrics['avg_loss'] = np.mean([t for t in trades if t <= 0]) if any(t <= 0 for t in trades) else 0
                metrics['profit_factor'] = abs(sum([t for t in trades if t > 0]) / sum([t for t in trades if t <= 0])) if sum([t for t in trades if t <= 0]) != 0 else float('inf')
    
    return metrics


def print_metrics(metrics):
    """
    Imprime as métricas de desempenho.
    
    Args:
        metrics (dict): Dicionário com métricas calculadas
    """
    print("\n--- MÉTRICAS DE DESEMPENHO ---")
    
    if 'total_return' in metrics:
        print(f"Retorno Total: {metrics['total_return']:.2f}%")
    
    if 'sharpe_ratio' in metrics:
        print(f"Sharpe Ratio (Anualizado): {metrics['sharpe_ratio']:.2f}")
    
    if 'max_drawdown' in metrics:
        print(f"Drawdown Máximo: {metrics['max_drawdown']:.2f}%")
    
    if 'calmar_ratio' in metrics:
        print(f"Calmar Ratio: {metrics['calmar_ratio']:.2f}")
    
    if 'n_trades' in metrics:
        print(f"Número de Operações: {metrics['n_trades']}")
    
    if 'win_rate' in metrics:
        print(f"Taxa de Acerto: {metrics['win_rate']:.2f}%")
    
    if 'avg_profit' in metrics and 'avg_loss' in metrics:
        print(f"Lucro Médio: {metrics['avg_profit']:.2f}%")
        print(f"Prejuízo Médio: {metrics['avg_loss']:.2f}%")
    
    if 'profit_factor' in metrics:
        print(f"Fator de Lucro: {metrics['profit_factor']:.2f}")


def plot_performance(history_df, test_data, initial_balance):
    """
    Plota o desempenho do agente.
    
    Args:
        history_df (pd.DataFrame): DataFrame com histórico de trading
        test_data (pd.DataFrame): Dados de teste
        initial_balance (float): Saldo inicial
    """
    if history_df.empty:
        print("Aviso: Não é possível plotar o desempenho com histórico vazio")
        return
        
    if 'net_worth' not in history_df.columns:
        print("Aviso: Não é possível plotar o desempenho sem a coluna 'net_worth'")
        return
    
    # Configura o estilo
    try:
        plt.style.use('dark_background')
    except:
        pass  # Usa o estilo padrão se dark_background não estiver disponível
    
    # Cria a figura
    fig, ax = plt.subplots(figsize=(15, 7))
    
    # Plota o patrimônio líquido
    ax.plot(history_df.index, history_df['net_worth'], color='blue', linewidth=2, label='Agente RL')
    
    # Plota buy & hold
    if 'close' in test_data.columns and len(test_data) >= len(history_df):
        try:
            buy_and_hold_data = test_data['close'].iloc[:len(history_df)]
            buy_and_hold = (buy_and_hold_data / buy_and_hold_data.iloc[0]) * initial_balance
            ax.plot(history_df.index, buy_and_hold.values, color='gray', linestyle='--', linewidth=1.5, label='Buy & Hold')
        except Exception as e:
            print(f"Erro ao plotar buy & hold: {e}")
    
    # Plota marcadores de compra/venda
    if 'position' in history_df.columns:
        try:
            buys = history_df[history_df['position'] > history_df['position'].shift(1).fillna(0)]
            sells = history_df[history_df['position'] < history_df['position'].shift(1).fillna(0)]
            
            if not buys.empty:
                ax.scatter(buys.index, buys['net_worth'], color='green', marker='^', s=100, label='Compra')
            
            if not sells.empty:
                ax.scatter(sells.index, sells['net_worth'], color='red', marker='v', s=100, label='Venda')
                
            # Imprime número de operações
            print(f"Número de compras: {len(buys)}")
            print(f"Número de vendas: {len(sells)}")
        except Exception as e:
            print(f"Erro ao plotar marcadores de compra/venda: {e}")
    
    # Configura o gráfico
    ax.set_title('Desempenho do Agente de Trading RL', fontsize=16)
    ax.set_ylabel('Patrimônio Líquido ($)', fontsize=12)
    ax.set_xlabel('Data', fontsize=12)
    ax.grid(True, alpha=0.2)
    ax.legend(loc='upper left', fontsize=10)
    
    # Formata o eixo x para datas se o índice for datetime
    if isinstance(history_df.index, pd.DatetimeIndex):
        date_format = DateFormatter('%Y-%m-%d')
        ax.xaxis.set_major_formatter(date_format)
        fig.autofmt_xdate()
    
    # Salva a figura
    plt.tight_layout()
    plt.savefig('performance.png')
    print("Gráfico de desempenho salvo como 'performance.png'")
    
    # Plota a distribuição de retornos
    try:
        plot_returns_distribution(history_df)
    except Exception as e:
        print(f"Erro ao plotar distribuição de retornos: {e}")


def plot_returns_distribution(history_df):
    """
    Plota a distribuição dos retornos diários.
    
    Args:
        history_df (pd.DataFrame): DataFrame com histórico de trading
    """
    if 'net_worth' not in history_df.columns:
        return
    
    # Calcula retornos diários
    returns = history_df['net_worth'].pct_change().fillna(0)
    
    # Configura o estilo
    plt.style.use('dark_background')
    
    # Cria a figura
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plota o histograma
    sns.histplot(returns, bins=30, kde=True, color='cyan', ax=ax)
    
    # Adiciona linha vertical em zero
    ax.axvline(x=0, color='red', linestyle='--', alpha=0.7)
    
    # Adiciona linha vertical na média
    ax.axvline(x=returns.mean(), color='lime', linestyle='-', alpha=0.7, 
               label=f'Média: {returns.mean():.4f}')
    
    # Configura o gráfico
    ax.set_title('Distribuição dos Retornos Diários', fontsize=16)
    ax.set_xlabel('Retorno Diário', fontsize=12)
    ax.set_ylabel('Frequência', fontsize=12)
    ax.grid(True, alpha=0.2)
    ax.legend()
    
    # Salva a figura
    plt.tight_layout()
    plt.savefig('returns_distribution.png')
    print("Distribuição de retornos salva como 'returns_distribution.png'")


def analyze_feature_importance(agent, test_env, feature_names):
    """
    Analisa a importância das features usando SHAP.
    
    Args:
        agent: Agente treinado
        test_env: Ambiente de teste
        feature_names (list): Nomes das features
        
    Returns:
        dict: Valores SHAP
    """
    if not SHAP_AVAILABLE:
        print("Aviso: Pacote SHAP não está disponível. Instale com 'pip install shap'")
        return None
    
    try:
        # Cria um explainer SHAP
        explainer = shap.Explainer(
            lambda x: agent.predict(x)[0], 
            np.zeros((1, len(feature_names)))
        )
        
        # Coleta observações do ambiente
        observations = []
        obs, _ = test_env.reset()
        done = False
        
        while not done and len(observations) < 100:  # Limita a 100 observações
            observations.append(obs)
            action, _ = agent.predict(np.array([obs]), deterministic=True)
            obs, _, terminated, truncated, _ = test_env.step(action[0])
            done = terminated or truncated
        
        # Calcula valores SHAP
        observations = np.array(observations)
        shap_values = explainer(observations)
        
        # Plota importância global das features
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, observations, feature_names=feature_names)
        plt.tight_layout()
        plt.savefig('shap_summary.png')
        print("Gráfico de importância de features salvo como 'shap_summary.png'")
        
        # Plota um force plot para uma decisão específica
        plt.figure(figsize=(20, 3))
        shap.initjs()
        force_plot = shap.force_plot(
            explainer.expected_value[0], 
            shap_values.values[0, :, 0], 
            observations[0], 
            feature_names=feature_names,
            matplotlib=True,
            show=False
        )
        plt.savefig('shap_force_plot.png', bbox_inches='tight')
        print("Force plot SHAP salvo como 'shap_force_plot.png'")
        
        return shap_values
        
    except Exception as e:
        print(f"Erro ao analisar importância das features: {e}")
        return None