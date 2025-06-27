"""
Módulo para carregamento e preparação de dados
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta


class DataLoader:
    """
    Classe para carregamento e preparação de dados.
    """
    
    def __init__(self, cache_dir='./data_cache'):
        """
        Inicializa o carregador de dados.
        
        Args:
            cache_dir (str): Diretório para cache de dados
        """
        self.cache_dir = cache_dir
        
    def load_ticker_data(self, ticker, start_date, end_date, use_cache=True, auto_adjust=True):
        """
        Carrega dados de um ticker.
        
        Args:
            ticker (str): Símbolo do ticker
            start_date (str): Data de início (formato: 'YYYY-MM-DD')
            end_date (str): Data de fim (formato: 'YYYY-MM-DD')
            use_cache (bool): Usar cache se disponível
            auto_adjust (bool): Ajustar preços para dividendos e splits
            
        Returns:
            pd.DataFrame: DataFrame com dados OHLCV
        """
        # Tenta carregar do cache
        cache_file = f"{self.cache_dir}/{ticker}_{start_date}_{end_date}.parquet"
        
        if use_cache:
            try:
                data = pd.read_parquet(cache_file)
                print(f"Dados carregados do cache: {len(data)} registros")
                return data
            except:
                print("Cache não encontrado, baixando dados...")
        
        # Baixa dados
        print(f"Baixando dados para {ticker}...")
        data = yf.download(ticker, start=start_date, end=end_date, auto_adjust=auto_adjust)
        
        if data.empty:
            raise ValueError(f"Não foi possível baixar dados para {ticker}")
            
        print(f"Dados baixados: {len(data)} registros de {data.index[0].date()} a {data.index[-1].date()}")
        
        # Salva no cache
        try:
            import os
            os.makedirs(self.cache_dir, exist_ok=True)
            data.to_parquet(cache_file)
            print(f"Dados salvos no cache: {cache_file}")
        except:
            print("Não foi possível salvar no cache")
        
        return data
    
    def load_multiple_tickers(self, tickers, start_date, end_date, use_cache=True, auto_adjust=True):
        """
        Carrega dados de múltiplos tickers.
        
        Args:
            tickers (list): Lista de símbolos de tickers
            start_date (str): Data de início (formato: 'YYYY-MM-DD')
            end_date (str): Data de fim (formato: 'YYYY-MM-DD')
            use_cache (bool): Usar cache se disponível
            auto_adjust (bool): Ajustar preços para dividendos e splits
            
        Returns:
            dict: Dicionário com DataFrames para cada ticker
        """
        data_dict = {}
        
        for ticker in tickers:
            try:
                data = self.load_ticker_data(ticker, start_date, end_date, use_cache, auto_adjust)
                data_dict[ticker] = data
            except Exception as e:
                print(f"Erro ao carregar {ticker}: {e}")
        
        return data_dict
    
    def merge_price_data(self, data_dict, column='close'):
        """
        Mescla dados de preços de múltiplos tickers.
        
        Args:
            data_dict (dict): Dicionário com DataFrames para cada ticker
            column (str): Coluna a ser mesclada
            
        Returns:
            pd.DataFrame: DataFrame com preços mesclados
        """
        merged_data = pd.DataFrame()
        
        for ticker, data in data_dict.items():
            if column in data.columns:
                merged_data[ticker] = data[column]
        
        return merged_data
    
    def add_market_data(self, data, market_ticker='^BVSP', start_date=None, end_date=None):
        """
        Adiciona dados de mercado ao DataFrame.
        
        Args:
            data (pd.DataFrame): DataFrame com dados
            market_ticker (str): Símbolo do índice de mercado
            start_date (str): Data de início (formato: 'YYYY-MM-DD')
            end_date (str): Data de fim (formato: 'YYYY-MM-DD')
            
        Returns:
            pd.DataFrame: DataFrame com dados de mercado adicionados
        """
        # Se start_date e end_date não forem fornecidos, usa o intervalo do DataFrame
        if start_date is None:
            start_date = data.index[0] - timedelta(days=10)
            start_date = start_date.strftime('%Y-%m-%d')
        
        if end_date is None:
            end_date = data.index[-1] + timedelta(days=10)
            end_date = end_date.strftime('%Y-%m-%d')
        
        # Baixa dados do mercado
        market_data = yf.download(market_ticker, start=start_date, end=end_date, auto_adjust=True)
        
        if market_data.empty:
            print(f"Aviso: Não foi possível baixar dados para {market_ticker}")
            return data
        
        # Adiciona retornos do mercado
        market_returns = market_data['Close'].pct_change()
        market_returns = market_returns.reindex(data.index)
        
        # Adiciona ao DataFrame original
        data_with_market = data.copy()
        data_with_market['market_return'] = market_returns
        
        # Calcula beta (regressão simples)
        if 'close' in data.columns:
            asset_returns = data['close'].pct_change()
            
            # Remove NaN
            valid_idx = ~(asset_returns.isna() | market_returns.isna())
            
            if valid_idx.sum() > 30:  # Precisa de pelo menos 30 pontos
                X = market_returns[valid_idx].values.reshape(-1, 1)
                y = asset_returns[valid_idx].values
                
                # Adiciona constante para o intercepto
                X = np.column_stack([np.ones(X.shape[0]), X])
                
                # Regressão linear (OLS)
                beta = np.linalg.lstsq(X, y, rcond=None)[0][1]
                
                # Adiciona beta como feature
                data_with_market['beta'] = beta
        
        return data_with_market