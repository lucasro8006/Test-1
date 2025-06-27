"""
Módulo para engenharia de features
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Tenta importar TA-Lib, se não estiver disponível, usa funções básicas
try:
    import talib as ta
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False
    print("Aviso: TA-Lib não está disponível. Usando implementações básicas de indicadores.")


class FeatureEngineer:
    """
    Classe responsável por toda a engenharia de features.
    
    Esta classe processa dados brutos de preços e cria features técnicas
    para uso em modelos de aprendizado de máquina.
    """
    
    def __init__(self):
        """Inicializa o engenheiro de features."""
        self.scaler = StandardScaler()
        self.feature_columns = []
        
    def transform(self, dataframe):
        """
        Transforma o DataFrame de dados brutos em features processadas.
        
        Args:
            dataframe (pd.DataFrame): DataFrame com dados OHLCV
            
        Returns:
            pd.DataFrame: DataFrame com todas as features calculadas e normalizadas
        """
        # Cria uma cópia para não modificar o original
        df = dataframe.copy()
        
        # Verifica se temos dados suficientes
        if len(df) < 30:
            raise ValueError("DataFrame precisa ter pelo menos 30 linhas para calcular indicadores")
        
        # Lida com MultiIndex do yfinance
        if isinstance(df.columns, pd.MultiIndex):
            print("Detectado MultiIndex do yfinance, convertendo para colunas simples...")
            # Converte para colunas simples
            df.columns = df.columns.to_flat_index()
            
            # Cria um novo DataFrame com colunas padronizadas
            new_df = pd.DataFrame(index=df.index)
            
            # Mapeia as colunas
            for col in df.columns:
                if 'Open' in col:
                    new_df['open'] = df[col]
                elif 'High' in col:
                    new_df['high'] = df[col]
                elif 'Low' in col:
                    new_df['low'] = df[col]
                elif 'Adj Close' in col:
                    new_df['close'] = df[col]
                elif 'Close' in col and 'Adj Close' not in str(df.columns):
                    new_df['close'] = df[col]
                elif 'Volume' in col:
                    new_df['volume'] = df[col]
            
            df = new_df
        else:
            # Padroniza os nomes das colunas (yfinance usa 'Open', 'High', etc.)
            column_mapping = {
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume',
                'Adj Close': 'close'  # Preferimos usar Adj Close
            }
            
            # Renomeia as colunas
            df = df.rename(columns=column_mapping)
        
        # Verifica se temos as colunas necessárias
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            raise ValueError(f"Colunas necessárias não encontradas: {missing_columns}")
            
        # Calcula indicadores técnicos
        self._add_technical_indicators(df)
        
        # Normaliza as features
        self._normalize_features(df)
        
        # Remove linhas com NaN (geralmente as primeiras por causa das médias móveis)
        df.dropna(inplace=True)
        
        # Armazena as colunas de features
        self.feature_columns = df.columns.tolist()
        
        return df
    
    def _add_technical_indicators(self, df):
        """
        Adiciona indicadores técnicos ao DataFrame.
        
        Args:
            df (pd.DataFrame): DataFrame com dados OHLCV
        """
        # Implementação básica de indicadores técnicos
        if TALIB_AVAILABLE:
            self._add_talib_indicators(df)
        else:
            self._add_basic_indicators(df)
            
    def _add_talib_indicators(self, df):
        """
        Adiciona indicadores técnicos usando TA-Lib.
        
        Args:
            df (pd.DataFrame): DataFrame com dados OHLCV
        """
        # Médias móveis
        df['sma_5'] = ta.SMA(df['close'], timeperiod=5)
        df['sma_20'] = ta.SMA(df['close'], timeperiod=20)
        df['sma_50'] = ta.SMA(df['close'], timeperiod=50)
        
        # Distância percentual das médias móveis
        df['dist_sma_5'] = (df['close'] / df['sma_5'] - 1) * 100
        df['dist_sma_20'] = (df['close'] / df['sma_20'] - 1) * 100
        df['dist_sma_50'] = (df['close'] / df['sma_50'] - 1) * 100
        
        # MACD
        df['macd'], df['macdsignal'], df['macdhist'] = ta.MACD(
            df['close'], fastperiod=12, slowperiod=26, signalperiod=9
        )
        
        # RSI
        df['rsi'] = ta.RSI(df['close'], timeperiod=14)
        
        # Bollinger Bands
        df['bbhigh'], df['bbmid'], df['bblow'] = ta.BBANDS(
            df['close'], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0
        )
        
        # Posição percentual nas Bollinger Bands (0-100%)
        df['bbpct'] = (df['close'] - df['bblow']) / (df['bbhigh'] - df['bblow']) * 100
        
        # Volatilidade (ATR normalizado)
        df['atr'] = ta.ATR(df['high'], df['low'], df['close'], timeperiod=14)
        df['atr_pct'] = df['atr'] / df['close'] * 100
        
        # Volume relativo (comparado com média de 20 dias)
        df['volume_sma_20'] = ta.SMA(df['volume'], timeperiod=20)
        df['volume_ratio'] = df['volume'] / df['volume_sma_20']
        
        # Momentum
        df['mom'] = ta.MOM(df['close'], timeperiod=10)
        
        # Retornos
        df['returns_1d'] = df['close'].pct_change(1)
        df['returns_5d'] = df['close'].pct_change(5)
        
    def _add_basic_indicators(self, df):
        """
        Adiciona indicadores técnicos usando implementações básicas (sem TA-Lib).
        
        Args:
            df (pd.DataFrame): DataFrame com dados OHLCV
        """
        # Médias móveis
        df['sma_5'] = df['close'].rolling(window=5).mean()
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['sma_50'] = df['close'].rolling(window=50).mean()
        
        # Distância percentual das médias móveis
        df['dist_sma_5'] = (df['close'] / df['sma_5'] - 1) * 100
        df['dist_sma_20'] = (df['close'] / df['sma_20'] - 1) * 100
        df['dist_sma_50'] = (df['close'] / df['sma_50'] - 1) * 100
        
        # Retornos
        df['returns_1d'] = df['close'].pct_change(1)
        df['returns_5d'] = df['close'].pct_change(5)
        
        # RSI simplificado
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands simplificadas
        df['bbmid'] = df['sma_20']
        df['bbstd'] = df['close'].rolling(window=20).std()
        df['bbhigh'] = df['bbmid'] + 2 * df['bbstd']
        df['bblow'] = df['bbmid'] - 2 * df['bbstd']
        df['bbpct'] = (df['close'] - df['bblow']) / (df['bbhigh'] - df['bblow']) * 100
        
        # Volume relativo
        df['volume_sma_20'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma_20']
        
        # Momentum
        df['mom'] = df['close'].diff(10)
        
        # MACD simplificado
        ema12 = df['close'].ewm(span=12, adjust=False).mean()
        ema26 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = ema12 - ema26
        df['macdsignal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macdhist'] = df['macd'] - df['macdsignal']
        
        # ATR simplificado
        high_low = df['high'] - df['low']
        high_close = (df['high'] - df['close'].shift()).abs()
        low_close = (df['low'] - df['close'].shift()).abs()
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        df['atr'] = true_range.rolling(14).mean()
        df['atr_pct'] = df['atr'] / df['close'] * 100
        
    def _normalize_features(self, df):
        """
        Normaliza as features para ter média 0 e desvio padrão 1.
        
        Args:
            df (pd.DataFrame): DataFrame com features calculadas
        """
        # Colunas para normalizar (excluindo OHLCV)
        cols_to_normalize = [col for col in df.columns if col not in ['open', 'high', 'low', 'close', 'volume']]
        
        if cols_to_normalize:
            # Ajusta o scaler e transforma
            df[cols_to_normalize] = self.scaler.fit_transform(df[cols_to_normalize].values)