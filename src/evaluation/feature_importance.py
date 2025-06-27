"""
Módulo para análise de importância de features em modelos de trading.

Este módulo implementa técnicas para analisar a importância das features
utilizadas pelo agente de trading, usando SHAP (SHapley Additive exPlanations)
e outras técnicas de interpretabilidade.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize

# Tenta importar SHAP
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("Aviso: SHAP não está disponível. Use 'pip install shap' para instalar.")


class FeatureImportanceAnalyzer:
    """
    Analisador de importância de features para modelos de trading.
    
    Esta classe fornece métodos para analisar e visualizar a importância
    das features utilizadas pelo agente de trading.
    """
    
    def __init__(self, model=None, feature_names=None):
        """
        Inicializa o analisador.
        
        Args:
            model: Modelo treinado (opcional)
            feature_names (list): Nomes das features (opcional)
        """
        self.model = model
        self.feature_names = feature_names
        self.shap_values = None
        self.feature_importance = None
        
    def analyze_shap(self, model, data, feature_names=None, n_samples=100, background_data=None):
        """
        Analisa a importância das features usando SHAP.
        
        Args:
            model: Modelo treinado
            data: Dados para análise
            feature_names (list): Nomes das features
            n_samples (int): Número de amostras para análise
            background_data: Dados de background para o explainer
            
        Returns:
            pd.DataFrame: DataFrame com importância das features
        """
        if not SHAP_AVAILABLE:
            print("Aviso: SHAP não está disponível. Use 'pip install shap' para instalar.")
            return None
            
        self.model = model
        
        if feature_names is not None:
            self.feature_names = feature_names
            
        try:
            # Seleciona amostras aleatórias
            if len(data) > n_samples:
                sample_indices = np.random.choice(len(data), n_samples, replace=False)
                samples = data.iloc[sample_indices] if hasattr(data, 'iloc') else data[sample_indices]
            else:
                samples = data
                
            # Prepara dados de background
            if background_data is None:
                background_data = samples
                
            # Cria um explainer SHAP
            if hasattr(model, 'policy') and hasattr(model.policy, 'predict'):
                # Para modelos de stable-baselines3
                explainer = shap.Explainer(model.policy.predict)
                
                # Prepara as observações
                if hasattr(samples, 'values'):
                    observations = samples.values
                else:
                    observations = samples
                    
                # Calcula valores SHAP
                self.shap_values = explainer(observations)
                
            elif hasattr(model, 'predict'):
                # Para modelos scikit-learn
                explainer = shap.Explainer(model.predict, background_data)
                self.shap_values = explainer(samples)
                
            else:
                # Tenta usar KernelExplainer como fallback
                explainer = shap.KernelExplainer(
                    lambda x: model.predict(x) if hasattr(model, 'predict') else model(x),
                    background_data
                )
                self.shap_values = explainer.shap_values(samples)
                
            # Cria DataFrame com importância
            if self.feature_names is None:
                if hasattr(data, 'columns'):
                    self.feature_names = data.columns.tolist()
                else:
                    self.feature_names = [f"feature_{i}" for i in range(self.shap_values.shape[1] if hasattr(self.shap_values, 'shape') else len(self.shap_values[0]))]
            
            # Calcula importância média
            if hasattr(self.shap_values, 'values'):
                importance_values = np.abs(self.shap_values.values).mean(axis=0)
            else:
                importance_values = np.abs(np.array(self.shap_values)).mean(axis=0)
                
            # Cria DataFrame
            self.feature_importance = pd.DataFrame({
                'feature': self.feature_names[:len(importance_values)],
                'importance': importance_values
            })
            
            # Ordena por importância
            self.feature_importance = self.feature_importance.sort_values('importance', ascending=False)
            
            return self.feature_importance
            
        except Exception as e:
            print(f"Erro ao analisar importância das features com SHAP: {e}")
            return None
            
    def plot_feature_importance(self, figsize=(10, 8), top_n=None, color='viridis'):
        """
        Plota a importância das features.
        
        Args:
            figsize (tuple): Tamanho da figura
            top_n (int): Número de features a serem mostradas
            color (str): Mapa de cores
            
        Returns:
            matplotlib.figure.Figure: Figura com o gráfico
        """
        if self.feature_importance is None:
            print("Aviso: Análise de importância das features não foi realizada")
            return None
            
        # Seleciona top N features
        if top_n is not None and top_n < len(self.feature_importance):
            plot_data = self.feature_importance.head(top_n)
        else:
            plot_data = self.feature_importance
            
        fig, ax = plt.subplots(figsize=figsize)
        
        # Cria mapa de cores
        cmap = cm.get_cmap(color)
        norm = Normalize(vmin=0, vmax=len(plot_data))
        
        # Plota importância das features
        bars = ax.barh(
            y=plot_data['feature'],
            width=plot_data['importance'],
            color=[cmap(norm(i)) for i in range(len(plot_data))]
        )
        
        # Adiciona valores
        for i, bar in enumerate(bars):
            ax.text(
                bar.get_width() + bar.get_width() * 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{plot_data['importance'].iloc[i]:.4f}",
                va='center'
            )
        
        ax.set_title('Importância das Features', fontsize=14)
        ax.set_xlabel('Importância', fontsize=12)
        ax.set_ylabel('Feature', fontsize=12)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
        
    def plot_shap_summary(self, figsize=(12, 8), max_display=20):
        """
        Plota o resumo dos valores SHAP.
        
        Args:
            figsize (tuple): Tamanho da figura
            max_display (int): Número máximo de features a serem mostradas
            
        Returns:
            matplotlib.figure.Figure: Figura com o gráfico
        """
        if not SHAP_AVAILABLE:
            print("Aviso: SHAP não está disponível. Use 'pip install shap' para instalar.")
            return None
            
        if self.shap_values is None:
            print("Aviso: Análise SHAP não foi realizada")
            return None
            
        plt.figure(figsize=figsize)
        
        # Plota resumo SHAP
        shap.summary_plot(
            self.shap_values,
            feature_names=self.feature_names,
            max_display=max_display,
            show=False
        )
        
        plt.tight_layout()
        return plt.gcf()
        
    def plot_shap_dependence(self, feature_idx, interaction_idx=None, figsize=(10, 6)):
        """
        Plota a dependência de uma feature com os valores SHAP.
        
        Args:
            feature_idx (int or str): Índice ou nome da feature
            interaction_idx (int or str): Índice ou nome da feature de interação
            figsize (tuple): Tamanho da figura
            
        Returns:
            matplotlib.figure.Figure: Figura com o gráfico
        """
        if not SHAP_AVAILABLE:
            print("Aviso: SHAP não está disponível. Use 'pip install shap' para instalar.")
            return None
            
        if self.shap_values is None:
            print("Aviso: Análise SHAP não foi realizada")
            return None
            
        # Converte nome para índice
        if isinstance(feature_idx, str) and self.feature_names is not None:
            try:
                feature_idx = self.feature_names.index(feature_idx)
            except ValueError:
                print(f"Aviso: Feature '{feature_idx}' não encontrada")
                return None
                
        if isinstance(interaction_idx, str) and self.feature_names is not None:
            try:
                interaction_idx = self.feature_names.index(interaction_idx)
            except ValueError:
                print(f"Aviso: Feature de interação '{interaction_idx}' não encontrada")
                interaction_idx = None
                
        plt.figure(figsize=figsize)
        
        # Plota dependência SHAP
        shap.dependence_plot(
            feature_idx,
            self.shap_values.values if hasattr(self.shap_values, 'values') else self.shap_values,
            features=self.shap_values.data if hasattr(self.shap_values, 'data') else None,
            feature_names=self.feature_names,
            interaction_index=interaction_idx,
            show=False
        )
        
        plt.tight_layout()
        return plt.gcf()
        
    def plot_shap_force(self, sample_idx=0, figsize=(20, 3)):
        """
        Plota o force plot SHAP para uma amostra específica.
        
        Args:
            sample_idx (int): Índice da amostra
            figsize (tuple): Tamanho da figura
            
        Returns:
            matplotlib.figure.Figure: Figura com o gráfico
        """
        if not SHAP_AVAILABLE:
            print("Aviso: SHAP não está disponível. Use 'pip install shap' para instalar.")
            return None
            
        if self.shap_values is None:
            print("Aviso: Análise SHAP não foi realizada")
            return None
            
        plt.figure(figsize=figsize)
        
        # Plota force plot SHAP
        if hasattr(self.shap_values, 'base_values'):
            # Para ExplanationSet
            shap.plots.force(
                self.shap_values[sample_idx],
                matplotlib=True,
                show=False
            )
        else:
            # Para valores SHAP antigos
            shap.force_plot(
                base_value=self.shap_values.base_values[sample_idx] if hasattr(self.shap_values, 'base_values') else 0,
                shap_values=self.shap_values.values[sample_idx] if hasattr(self.shap_values, 'values') else self.shap_values[sample_idx],
                features=self.shap_values.data[sample_idx] if hasattr(self.shap_values, 'data') else None,
                feature_names=self.feature_names,
                matplotlib=True,
                show=False
            )
        
        plt.tight_layout()
        return plt.gcf()
        
    def get_top_features(self, n=10):
        """
        Retorna as top N features mais importantes.
        
        Args:
            n (int): Número de features a serem retornadas
            
        Returns:
            pd.DataFrame: DataFrame com as top N features
        """
        if self.feature_importance is None:
            print("Aviso: Análise de importância das features não foi realizada")
            return None
            
        return self.feature_importance.head(n)
        
    def get_feature_impact(self, feature_name):
        """
        Retorna o impacto de uma feature específica.
        
        Args:
            feature_name (str): Nome da feature
            
        Returns:
            dict: Dicionário com informações de impacto
        """
        if self.feature_importance is None or self.shap_values is None:
            print("Aviso: Análise de importância das features não foi realizada")
            return None
            
        # Encontra índice da feature
        if feature_name in self.feature_names:
            feature_idx = self.feature_names.index(feature_name)
        else:
            print(f"Aviso: Feature '{feature_name}' não encontrada")
            return None
            
        # Obtém valores SHAP para a feature
        if hasattr(self.shap_values, 'values'):
            feature_shap = self.shap_values.values[:, feature_idx]
            feature_data = self.shap_values.data[:, feature_idx]
        else:
            feature_shap = np.array(self.shap_values)[:, feature_idx]
            feature_data = None
            
        # Calcula estatísticas
        impact = {
            'feature': feature_name,
            'importance': self.feature_importance[self.feature_importance['feature'] == feature_name]['importance'].values[0],
            'mean_impact': feature_shap.mean(),
            'abs_mean_impact': np.abs(feature_shap).mean(),
            'min_impact': feature_shap.min(),
            'max_impact': feature_shap.max(),
            'positive_impact_pct': (feature_shap > 0).mean() * 100,
            'negative_impact_pct': (feature_shap < 0).mean() * 100
        }
        
        return impact