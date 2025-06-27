"""
Utilitários para manipulação de dados
"""

import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.model_selection import TimeSeriesSplit


def download_data(ticker, start_date, end_date, auto_adjust=True):
    """
    Baixa dados históricos de preços.
    
    Args:
        ticker (str): Símbolo do ativo
        start_date (str): Data de início (formato: 'YYYY-MM-DD')
        end_date (str): Data de fim (formato: 'YYYY-MM-DD')
        auto_adjust (bool): Ajustar preços para dividendos e splits
        
    Returns:
        pd.DataFrame: DataFrame com dados OHLCV
    """
    print(f"Baixando dados para {ticker}...")
    data = yf.download(ticker, start=start_date, end=end_date, auto_adjust=auto_adjust)
    
    if data.empty:
        raise ValueError(f"Não foi possível baixar dados para {ticker}")
        
    print(f"Dados baixados: {len(data)} registros de {data.index[0].date()} a {data.index[-1].date()}")
    
    return data


def split_data(data, train_size=0.7, val_size=0.15, test_size=0.15):
    """
    Divide os dados em conjuntos de treino, validação e teste.
    
    Args:
        data (pd.DataFrame): DataFrame com dados
        train_size (float): Proporção para treino
        val_size (float): Proporção para validação
        test_size (float): Proporção para teste
        
    Returns:
        tuple: DataFrames de treino, validação e teste
    """
    # Verifica se as proporções somam 1
    if abs(train_size + val_size + test_size - 1.0) > 1e-10:
        raise ValueError("As proporções devem somar 1")
        
    # Calcula os índices de divisão
    n = len(data)
    train_end = int(n * train_size)
    val_end = train_end + int(n * val_size)
    
    # Divide os dados
    train_data = data.iloc[:train_end].copy()
    val_data = data.iloc[train_end:val_end].copy()
    test_data = data.iloc[val_end:].copy()
    
    print(f"Dados divididos: Treino={len(train_data)}, Validação={len(val_data)}, Teste={len(test_data)}")
    
    return train_data, val_data, test_data


def create_purged_kfold(data, n_splits=5, embargo_size=5):
    """
    Cria uma validação cruzada encadeada com purga e embargo.
    
    Args:
        data (pd.DataFrame): DataFrame com dados
        n_splits (int): Número de folds
        embargo_size (int): Tamanho do embargo
        
    Returns:
        list: Lista de tuplas (train_idx, test_idx)
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    # Cria os folds com purga e embargo
    purged_folds = []
    
    for train_idx, test_idx in tscv.split(data):
        # Aplica embargo: remove pontos do treino que estão próximos ao teste
        if embargo_size > 0:
            test_start = test_idx[0]
            test_end = test_idx[-1]
            
            # Remove pontos do treino que estão no embargo
            embargo_train_idx = [i for i in train_idx if i < test_start - embargo_size]
            
            purged_folds.append((embargo_train_idx, test_idx))
        else:
            purged_folds.append((train_idx, test_idx))
    
    return purged_folds