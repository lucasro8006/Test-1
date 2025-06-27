"""
Módulo para validação cruzada em séries temporais financeiras.

Este módulo implementa técnicas de validação cruzada específicas para dados financeiros,
incluindo Purged K-Fold Cross-Validation e Walk-Forward Validation.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.model_selection._split import _BaseKFold
import datetime as dt


class PurgedKFold(_BaseKFold):
    """
    Implementa Purged K-Fold Cross-Validation para dados financeiros.
    
    Esta técnica evita vazamento de dados (data leakage) em séries temporais
    financeiras, removendo amostras que podem causar contaminação entre
    os conjuntos de treino e teste.
    """
    
    def __init__(self, n_splits=5, pct_embargo=0.0, purge_method='time'):
        """
        Inicializa o PurgedKFold.
        
        Args:
            n_splits (int): Número de folds
            pct_embargo (float): Percentual de dados a serem embargados após cada fold
            purge_method (str): Método de purga ('time' ou 'index')
        """
        super(PurgedKFold, self).__init__(n_splits=n_splits, shuffle=False, random_state=None)
        self.pct_embargo = pct_embargo
        self.purge_method = purge_method
        
    def split(self, X, y=None, groups=None):
        """
        Gera índices para dividir os dados em conjuntos de treino/teste.
        
        Args:
            X: Dados de entrada
            y: Alvo (não usado)
            groups: Timestamps ou índices para purga
            
        Yields:
            tuple: Índices de treino e teste para cada fold
        """
        if self.n_splits > len(X):
            raise ValueError(f"Não é possível ter n_splits={self.n_splits} com {len(X)} amostras.")
            
        # Verifica se groups é um array de timestamps
        if groups is None:
            if isinstance(X.index[0], (dt.datetime, pd.Timestamp)):
                groups = X.index
            else:
                groups = np.arange(len(X))
                
        # Converte para array numpy se for pandas Series/DatetimeIndex
        if isinstance(groups, (pd.Series, pd.DatetimeIndex)):
            groups = groups.values
            
        # Índices para divisão
        indices = np.arange(len(X))
        
        # Tamanho do embargo
        embargo_size = int(len(X) * self.pct_embargo)
        
        # Divisão em folds
        fold_size = len(X) // self.n_splits
        
        # Gera os folds
        for i in range(self.n_splits):
            # Índices de teste para este fold
            test_start = i * fold_size
            test_end = (i + 1) * fold_size if i < self.n_splits - 1 else len(X)
            test_indices = indices[test_start:test_end]
            
            # Índices de treino (todos exceto os de teste)
            train_indices = np.setdiff1d(indices, test_indices)
            
            # Aplica purga
            if self.purge_method == 'time' and len(groups) > 0:
                # Obtém timestamps de teste
                test_times = groups[test_indices]
                
                # Obtém timestamps de treino
                train_times = groups[train_indices]
                
                # Encontra amostras de treino que estão muito próximas das de teste
                if isinstance(test_times[0], (dt.datetime, pd.Timestamp)):
                    # Calcula a diferença mínima de tempo para purga (1 dia)
                    min_time_diff = pd.Timedelta(days=1)
                    
                    # Identifica amostras a serem purgadas
                    for test_time in test_times:
                        # Calcula diferença de tempo
                        time_diffs = np.abs(train_times - test_time)
                        
                        # Encontra amostras muito próximas
                        to_purge = np.where(time_diffs < min_time_diff)[0]
                        
                        # Remove do conjunto de treino
                        if len(to_purge) > 0:
                            train_indices = np.setdiff1d(train_indices, train_indices[to_purge])
            
            # Aplica embargo
            if embargo_size > 0:
                # Encontra índices após o conjunto de teste
                if i < self.n_splits - 1:
                    embargo_indices = indices[test_end:test_end + embargo_size]
                    # Remove do conjunto de treino
                    train_indices = np.setdiff1d(train_indices, embargo_indices)
            
            yield train_indices, test_indices


class WalkForwardValidation:
    """
    Implementa Walk-Forward Validation para dados financeiros.
    
    Esta técnica simula o processo de trading em tempo real, treinando
    o modelo com dados históricos e testando em dados futuros, avançando
    a janela de tempo a cada iteração.
    """
    
    def __init__(self, n_splits=5, train_size=0.7, step_size=None, purge_days=0, embargo_days=0):
        """
        Inicializa o WalkForwardValidation.
        
        Args:
            n_splits (int): Número de folds
            train_size (float): Proporção de dados para treino em cada fold
            step_size (int): Tamanho do passo para avançar a janela (em dias)
            purge_days (int): Número de dias a serem purgados entre treino e teste
            embargo_days (int): Número de dias a serem embargados após o teste
        """
        self.n_splits = n_splits
        self.train_size = train_size
        self.step_size = step_size
        self.purge_days = purge_days
        self.embargo_days = embargo_days
        
    def split(self, X, y=None, groups=None):
        """
        Gera índices para dividir os dados em conjuntos de treino/teste.
        
        Args:
            X: Dados de entrada
            y: Alvo (não usado)
            groups: Timestamps ou índices (não usado)
            
        Yields:
            tuple: Índices de treino e teste para cada fold
        """
        # Verifica se o índice é de timestamps
        if isinstance(X.index[0], (dt.datetime, pd.Timestamp)):
            dates = X.index
        else:
            dates = pd.date_range(start='2000-01-01', periods=len(X))
            
        # Calcula tamanho total
        total_size = len(X)
        
        # Calcula tamanho do treino e teste
        train_size = int(total_size * self.train_size)
        test_size = total_size - train_size
        
        # Calcula tamanho do passo
        if self.step_size is None:
            step_size = test_size
        else:
            step_size = self.step_size
            
        # Gera os folds
        for i in range(self.n_splits):
            # Calcula índices de início e fim
            start_idx = i * step_size
            train_end_idx = start_idx + train_size
            test_end_idx = min(train_end_idx + test_size, total_size)
            
            # Verifica se ainda temos dados suficientes
            if test_end_idx >= total_size:
                break
                
            # Índices de treino e teste
            train_indices = np.arange(start_idx, train_end_idx)
            test_indices = np.arange(train_end_idx, test_end_idx)
            
            # Aplica purga
            if self.purge_days > 0:
                # Calcula data de corte
                cutoff_date = dates[train_end_idx - 1] + pd.Timedelta(days=self.purge_days)
                
                # Remove amostras de treino após a data de corte
                train_indices = train_indices[dates[train_indices] <= cutoff_date]
                
                # Remove amostras de teste antes da data de corte
                test_indices = test_indices[dates[test_indices] >= cutoff_date]
                
            # Aplica embargo
            if self.embargo_days > 0 and i < self.n_splits - 1:
                # Calcula data de embargo
                embargo_date = dates[test_end_idx - 1] + pd.Timedelta(days=self.embargo_days)
                
                # Encontra índices a serem embargados
                embargo_indices = np.where((dates > dates[test_end_idx - 1]) & (dates < embargo_date))[0]
                
                # Remove do próximo conjunto de treino
                next_train_start = test_end_idx
                next_train_end = min(next_train_start + train_size, total_size)
                next_train_indices = np.arange(next_train_start, next_train_end)
                next_train_indices = np.setdiff1d(next_train_indices, embargo_indices)
                
            yield train_indices, test_indices


def cross_validate_agent(agent_class, env_class, data, cv, model_params, env_params, 
                         total_timesteps=10000, metric='sharpe_ratio'):
    """
    Realiza validação cruzada para um agente de trading.
    
    Args:
        agent_class: Classe do agente de trading
        env_class: Classe do ambiente de trading
        data (pd.DataFrame): Dados para validação
        cv: Objeto de validação cruzada (PurgedKFold ou WalkForwardValidation)
        model_params (dict): Parâmetros do modelo
        env_params (dict): Parâmetros do ambiente
        total_timesteps (int): Número de passos de treinamento
        metric (str): Métrica a ser avaliada
        
    Returns:
        dict: Resultados da validação cruzada
    """
    from stable_baselines3.common.vec_env import DummyVecEnv
    
    # Resultados
    results = {
        'train_metrics': [],
        'test_metrics': [],
        'fold_indices': []
    }
    
    # Realiza validação cruzada
    for i, (train_idx, test_idx) in enumerate(cv.split(data)):
        print(f"Fold {i+1}/{cv.n_splits}")
        
        # Divide os dados
        train_data = data.iloc[train_idx]
        test_data = data.iloc[test_idx]
        
        # Cria ambiente de treino
        train_env = DummyVecEnv([
            lambda: env_class(train_data, **env_params)
        ])
        
        # Cria ambiente de teste
        test_env = env_class(test_data, **env_params)
        
        # Cria e treina o agente
        agent = agent_class(model_params)
        agent.train(train_env, total_timesteps=total_timesteps)
        
        # Avalia no conjunto de treino
        train_metrics = agent.evaluate(train_env)
        
        # Avalia no conjunto de teste
        test_metrics = agent.evaluate(DummyVecEnv([lambda: test_env]))
        
        # Armazena resultados
        results['train_metrics'].append(train_metrics)
        results['test_metrics'].append(test_metrics)
        results['fold_indices'].append((train_idx, test_idx))
        
        # Imprime métricas
        print(f"  Train {metric}: {train_metrics.get(metric, 'N/A')}")
        print(f"  Test {metric}: {test_metrics.get(metric, 'N/A')}")
        
    # Calcula médias
    for m in results['train_metrics'][0].keys():
        train_values = [metrics.get(m, 0) for metrics in results['train_metrics']]
        test_values = [metrics.get(m, 0) for metrics in results['test_metrics']]
        
        results[f'mean_train_{m}'] = np.mean(train_values)
        results[f'std_train_{m}'] = np.std(train_values)
        results[f'mean_test_{m}'] = np.mean(test_values)
        results[f'std_test_{m}'] = np.std(test_values)
        
    return results