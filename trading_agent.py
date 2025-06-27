"""
Trading Agent with Reinforcement Learning

This module implements a complete pipeline for training and evaluating a trading agent
using Reinforcement Learning (RL) with stable-baselines3 and a custom Gymnasium environment.
"""

import os
import pandas as pd
import numpy as np
import yfinance as yf
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
import matplotlib.pyplot as plt
import optuna
from sklearn.model_selection import TimeSeriesSplit
import warnings

# Suppress yfinance warning about auto_adjust
warnings.filterwarnings("ignore", category=FutureWarning, module="yfinance")

# Try to import SHAP, but make it optional
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("SHAP not available. Feature importance analysis will be disabled.")

class FeatureEngineer:
    """
    Class responsible for all feature engineering tasks.
    """
    def __init__(self):
        """Initialize the feature engineer with default parameters."""
        self.feature_cols = ['macd', 'macdhist', 'rsi', 'bbhigh', 'bblow']
        self.state_columns = self.feature_cols + ['open', 'high', 'low', 'close', 'volume']
        
    def transform(self, df):
        """
        Transform raw price data into a feature-rich DataFrame.
        
        Args:
            df (pd.DataFrame): Raw price data with OHLCV columns
            
        Returns:
            pd.DataFrame: DataFrame with calculated features
        """
        # Ensure column names are lowercase
        df = df.copy()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0].lower() for col in df.columns]
        else:
            df.columns = [col.lower() for col in df.columns]
        
        close = df['close']
        
        # Calculate MACD
        df['ema12'] = close.ewm(span=12, adjust=False).mean()
        df['ema26'] = close.ewm(span=26, adjust=False).mean()
        df['macd'] = df['ema12'] - df['ema26']
        df['macdsignal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macdhist'] = df['macd'] - df['macdsignal']
        
        # Calculate RSI
        delta = close.diff(1)
        gain = delta.clip(lower=0).ewm(span=14, adjust=False).mean()
        loss = abs(delta.clip(upper=0)).ewm(span=14, adjust=False).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Calculate Bollinger Bands
        df['bbhigh'] = close.rolling(20).mean() + (close.rolling(20).std() * 2)
        df['bblow'] = close.rolling(20).mean() - (close.rolling(20).std() * 2)
        
        # Normalize features
        for col in self.feature_cols:
            df[col] = (df[col] - df[col].mean()) / df[col].std()
            
        return df[self.state_columns].dropna()


class TradingEnv(gym.Env):
    """
    Custom Gymnasium environment for simulating trading with RL.
    """
    def __init__(self, df, initial_balance=100000, commission_pct=0.001, slippage_pct=0.0005, reward_type='sharpe'):
        """
        Initialize the trading environment.
        
        Args:
            df (pd.DataFrame): DataFrame with price data and features
            initial_balance (float): Starting capital
            commission_pct (float): Commission percentage per trade
            slippage_pct (float): Slippage percentage per trade
            reward_type (str): Type of reward function ('pnl', 'sharpe', or 'calmar')
        """
        super(TradingEnv, self).__init__()
        
        self.df = df
        self.initial_balance = initial_balance
        self.commission_pct = commission_pct
        self.slippage_pct = slippage_pct
        self.reward_type = reward_type
        self.current_step = 0
        
        # Define action space: 0 = hold, 1 = buy, 2 = sell
        self.action_space = spaces.Discrete(3)
        
        # Observation space includes features + position + position_pnl
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(len(df.columns) + 2,), 
            dtype=np.float32
        )
        
        # For Sharpe ratio calculation
        self.returns_history = []

    def _get_obs(self):
        """Get the current observation (state)."""
        # Make sure we don't go out of bounds
        step = min(self.current_step, len(self.df) - 1)
        obs = self.df.iloc[step].values
        obs = np.append(obs, [self.position, self.position_pnl])
        return obs.astype(np.float32)

    def reset(self, seed=None):
        """Reset the environment to initial state."""
        super().reset(seed=seed)
        self.balance = self.initial_balance
        self.net_worth = self.initial_balance
        self.position = 0  # 0 = no position, 1 = long
        self.entry_price = 0
        self.position_pnl = 0
        self.current_step = 0
        self.history = []
        self.returns_history = []
        return self._get_obs(), {}

    def _calculate_reward(self, pnl=0):
        """
        Calculate reward based on the specified reward type.
        
        Args:
            pnl (float): Profit and loss from the current action
            
        Returns:
            float: Calculated reward
        """
        if self.reward_type == 'pnl':
            return pnl
        
        elif self.reward_type == 'sharpe':
            # Add current return to history
            if pnl != 0:  # Only add non-zero returns (when we close a position)
                self.returns_history.append(pnl)
            
            # Calculate Sharpe ratio if we have enough data
            if len(self.returns_history) > 1:
                returns_array = np.array(self.returns_history)
                sharpe = returns_array.mean() / (returns_array.std() + 1e-9) * np.sqrt(252)
                return sharpe * 0.01  # Scale down the reward
            return 0
            
        elif self.reward_type == 'calmar':
            # Add current return to history
            if pnl != 0:
                self.returns_history.append(pnl)
                
            # Calculate Calmar ratio if we have enough data
            if len(self.returns_history) > 10:  # Need enough data for meaningful max drawdown
                returns_array = np.array(self.returns_history)
                cumulative_returns = np.cumprod(1 + returns_array) - 1
                peak = np.maximum.accumulate(cumulative_returns)
                drawdown = (cumulative_returns - peak) / (peak + 1)
                max_drawdown = abs(drawdown.min()) + 1e-9  # Avoid division by zero
                annual_return = returns_array.mean() * 252
                calmar = annual_return / max_drawdown
                return calmar * 0.01  # Scale down the reward
            return 0
            
        else:
            return pnl  # Default to PnL

    def step(self, action):
        """
        Execute one step in the environment.
        
        Args:
            action (int): Action to take (0=hold, 1=buy, 2=sell)
            
        Returns:
            tuple: (observation, reward, terminated, truncated, info)
        """
        # Check if we've reached the end of the data
        if self.current_step >= len(self.df) - 1:
            return self._get_obs(), 0, True, False, {}
            
        current_price = self.df['close'].iloc[self.current_step]
        reward = 0
        pnl = 0
        
        # Apply action
        if action == 1 and self.position == 0:  # Buy
            # Apply slippage to entry price
            entry_price_with_slippage = current_price * (1 + self.slippage_pct)
            # Apply commission
            entry_cost = entry_price_with_slippage * (1 + self.commission_pct)
            
            self.position = 1
            self.entry_price = entry_cost
            
        elif action == 2 and self.position == 1:  # Sell
            # Apply slippage to exit price
            exit_price_with_slippage = current_price * (1 - self.slippage_pct)
            # Apply commission
            exit_price_after_commission = exit_price_with_slippage * (1 - self.commission_pct)
            
            # Calculate PnL
            pnl = (exit_price_after_commission - self.entry_price) / self.entry_price
            reward = self._calculate_reward(pnl)
            
            # Update balance
            self.balance *= (1 + pnl)
            self.position = 0
            self.entry_price = 0

        # Update position PnL if we have a position
        if self.position == 1:
            self.position_pnl = (current_price - self.entry_price) / self.entry_price
        else:
            self.position_pnl = 0
            
        # Update net worth
        self.net_worth = self.balance * (1 + self.position_pnl)
        
        # Small penalty for holding a position (to encourage action)
        if self.position != 0 and reward == 0:
            reward -= 0.0001
            
        # Record history
        self.history.append({
            'step': self.current_step, 
            'net_worth': self.net_worth,
            'position': self.position,
            'price': current_price
        })
        
        # Move to next step
        self.current_step += 1
        terminated = self.current_step >= len(self.df) - 1
        truncated = False
        
        return self._get_obs(), reward, terminated, truncated, {}


class TradingAgent:
    """
    Wrapper class for the RL model that handles training and prediction.
    """
    def __init__(self, model_params=None):
        """
        Initialize the trading agent.
        
        Args:
            model_params (dict): Parameters for the PPO model
        """
        self.model = None
        self.model_params = model_params or {
            'policy': 'MlpPolicy',
            'verbose': 0,
            'tensorboard_log': "./ppo_trading_tensorboard/"
        }
        
    def train(self, environment, total_timesteps=100000, progress_bar=True):
        """
        Train the agent on the given environment.
        
        Args:
            environment: Vectorized gym environment
            total_timesteps (int): Number of timesteps to train for
            progress_bar (bool): Whether to show a progress bar
            
        Returns:
            self: The trained agent
        """
        # Initialize the model if it doesn't exist
        if self.model is None:
            self.model = PPO(env=environment, **self.model_params)
            
        # Train the model
        self.model.learn(total_timesteps=total_timesteps, progress_bar=progress_bar)
        return self
    
    def predict(self, observation, deterministic=True):
        """
        Make a prediction based on the current observation.
        
        Args:
            observation: The current state observation
            deterministic (bool): Whether to use deterministic actions
            
        Returns:
            tuple: (action, state)
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet")
        return self.model.predict(observation, deterministic=deterministic)
    
    def save(self, path):
        """
        Save the model to disk.
        
        Args:
            path (str): Path to save the model to
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet")
        self.model.save(path)
        
    def load(self, path):
        """
        Load a model from disk.
        
        Args:
            path (str): Path to load the model from
            
        Returns:
            self: The loaded agent
        """
        self.model = PPO.load(path)
        return self


class PurgedTimeSeriesSplit:
    """
    Time Series cross-validator with purging and embargo.
    
    Provides train/test indices to split time series data samples
    that are observed at fixed time intervals, in train/test sets.
    In each split, test indices must be higher than before, and thus shuffling
    in cross validator is inappropriate.
    
    This cross-validation object is a variation of TimeSeriesSplit with the following differences:
    - Purging: Removes from the training set samples that overlap with the test set
    - Embargo: Removes additional training samples immediately prior to the test set
    """
    
    def __init__(self, n_splits=5, purge_days=5, embargo_days=5):
        """
        Initialize the cross-validator.
        
        Args:
            n_splits (int): Number of splits
            purge_days (int): Number of days to purge from training set
            embargo_days (int): Number of days to embargo from training set
        """
        self.n_splits = n_splits
        self.purge_days = purge_days
        self.embargo_days = embargo_days
        
    def split(self, X, y=None, groups=None):
        """
        Generate indices to split data into training and test set.
        
        Args:
            X (array-like): Data to split
            y (array-like): Ignored
            groups (array-like): Ignored
            
        Yields:
            tuple: (train_index, test_index)
        """
        n_samples = len(X)
        indices = np.arange(n_samples)
        
        # Determine the size of each test split
        test_size = n_samples // (self.n_splits + 1)
        
        # Generate the splits
        for i in range(self.n_splits):
            # Test indices
            test_start = (i + 1) * test_size
            test_end = min(test_start + test_size, n_samples)
            test_indices = indices[test_start:test_end]
            
            # Training indices with purging and embargo
            purge_start = max(0, test_start - self.purge_days)
            embargo_end = max(0, purge_start - self.embargo_days)
            
            train_indices = np.concatenate([
                indices[:embargo_end],
                indices[test_end:]
            ])
            
            yield train_indices, test_indices


def optimize_agent(train_data, validation_data, n_trials=50):
    """
    Optimize hyperparameters using Optuna.
    
    Args:
        train_data (pd.DataFrame): Training data
        validation_data (pd.DataFrame): Validation data
        n_trials (int): Number of optimization trials
        
    Returns:
        dict: Best hyperparameters
    """
    def objective(trial):
        # Define hyperparameters to optimize
        learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)
        n_steps = trial.suggest_int('n_steps', 16, 2048, log=True)
        batch_size = trial.suggest_int('batch_size', 8, 256, log=True)
        gamma = trial.suggest_float('gamma', 0.9, 0.9999)
        commission = trial.suggest_float('commission', 0.0001, 0.01, log=True)
        slippage = trial.suggest_float('slippage', 0.0001, 0.005, log=True)
        reward_type = trial.suggest_categorical('reward_type', ['pnl', 'sharpe', 'calmar'])
        
        # Neural network architecture
        net_arch_str = trial.suggest_categorical('net_arch', ['[64, 64]', '[128, 64]', '[256, 128, 64]'])
        net_arch = eval(net_arch_str)
        
        # Cross-validation
        cv = PurgedTimeSeriesSplit(n_splits=3, purge_days=5, embargo_days=5)
        sharpe_ratios = []
        
        for train_idx, test_idx in cv.split(train_data):
            cv_train_data = train_data.iloc[train_idx]
            cv_test_data = train_data.iloc[test_idx]
            
            # Create environments
            train_env = DummyVecEnv([
                lambda: TradingEnv(
                    cv_train_data, 
                    commission_pct=commission, 
                    slippage_pct=slippage,
                    reward_type=reward_type
                )
            ])
            
            test_env = DummyVecEnv([
                lambda: TradingEnv(
                    cv_test_data, 
                    commission_pct=commission, 
                    slippage_pct=slippage,
                    reward_type=reward_type
                )
            ])
            
            # Create and train agent
            model_params = {
                'policy': 'MlpPolicy',
                'learning_rate': learning_rate,
                'n_steps': n_steps,
                'batch_size': batch_size,
                'gamma': gamma,
                'policy_kwargs': {'net_arch': net_arch},
                'verbose': 0
            }
            
            agent = TradingAgent(model_params)
            agent.train(train_env, total_timesteps=20000, progress_bar=False)
            
            # Evaluate on test set
            obs = test_env.reset()
            done = False
            while not done:
                action, _ = agent.predict(obs)
                obs, _, terminated, truncated, _ = test_env.step(action)
                done = terminated[0] or truncated[0]
            
            # Calculate Sharpe ratio
            history_df = pd.DataFrame(test_env.envs[0].history)
            returns = history_df['net_worth'].pct_change().fillna(0)
            sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0
            sharpe_ratios.append(sharpe_ratio)
        
        # Return mean Sharpe ratio across folds
        return np.mean(sharpe_ratios)
    
    # Create Optuna study
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)
    
    print(f"Best trial: {study.best_trial.number}")
    print(f"Best value: {study.best_trial.value}")
    print(f"Best hyperparameters: {study.best_trial.params}")
    
    return study.best_trial.params


def analyze_feature_importance(model, test_data):
    """
    Analyze feature importance using SHAP.
    
    Args:
        model: Trained model
        test_data (pd.DataFrame): Test data
        
    Returns:
        tuple: (shap_values, explainer) or None if SHAP is not available
    """
    if not SHAP_AVAILABLE:
        print("SHAP is not available. Skipping feature importance analysis.")
        return None
    
    try:
        # Create a background dataset for SHAP
        background_data = test_data.sample(min(100, len(test_data)))
        
        # Create explainer
        explainer = shap.Explainer(model.predict, background_data)
        
        # Calculate SHAP values
        shap_values = explainer(test_data)
        
        # Plot global feature importance
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, test_data, plot_type="bar")
        
        # Plot detailed SHAP values
        plt.figure(figsize=(15, 8))
        shap.summary_plot(shap_values, test_data)
        
        # Force plot for a specific decision
        plt.figure(figsize=(20, 3))
        shap.plots.force(shap_values[0])
        
        return shap_values, explainer
    except Exception as e:
        print(f"Error in SHAP analysis: {e}")
        return None


def evaluate_agent(agent, test_data, commission_pct=0.001, slippage_pct=0.0005, reward_type='sharpe'):
    """
    Evaluate the agent on test data.
    
    Args:
        agent: Trained agent
        test_data (pd.DataFrame): Test data
        commission_pct (float): Commission percentage
        slippage_pct (float): Slippage percentage
        reward_type (str): Reward type
        
    Returns:
        pd.DataFrame: History DataFrame
    """
    # Create a direct environment (not vectorized) for evaluation
    test_env = TradingEnv(
        test_data, 
        commission_pct=commission_pct, 
        slippage_pct=slippage_pct,
        reward_type=reward_type
    )
    
    # Reset environment
    obs, _ = test_env.reset()
    done = False
    
    # Run agent on test data
    while not done:
        # Reshape observation for the model
        obs_reshaped = np.array([obs])
        action, _ = agent.predict(obs_reshaped, deterministic=True)
        obs, reward, terminated, truncated, _ = test_env.step(action[0])
        done = terminated or truncated
    
    # Get history
    history = test_env.history
    print(f"History length: {len(history)}")
    print(f"First history entry: {history[0] if history else 'No history'}")
    
    # Create DataFrame from history
    history_df = pd.DataFrame(history)
    history_df.index = test_data.index[:len(history_df)]
    
    # Calculate metrics
    if 'net_worth' in history_df.columns:
        returns = history_df['net_worth'].pct_change().fillna(0)
        sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0
        total_return = (history_df['net_worth'].iloc[-1] / history_df['net_worth'].iloc[0] - 1) * 100
        
        # Calculate drawdown
        peak = history_df['net_worth'].cummax()
        drawdown = (history_df['net_worth'] - peak) / peak
        max_drawdown = drawdown.min() * 100
        
        # Calculate Calmar ratio
        annual_return = returns.mean() * 252
        calmar_ratio = annual_return / (abs(max_drawdown) / 100) if max_drawdown != 0 else 0
    else:
        print("Warning: 'net_worth' column not found in history DataFrame")
        sharpe_ratio = 0
        total_return = 0
        max_drawdown = 0
        calmar_ratio = 0
    
    print("\n--- MÉTRICAS DE DESEMPENHO NO PERÍODO DE TESTE ---")
    print(f"Retorno Total da Estratégia: {total_return:.2f}%")
    print(f"Sharpe Ratio (Anualizado): {sharpe_ratio:.2f}")
    print(f"Drawdown Máximo: {max_drawdown:.2f}%")
    print(f"Calmar Ratio: {calmar_ratio:.2f}")
    
    # Plot performance if net_worth is available
    if 'net_worth' in history_df.columns:
        plt.style.use('dark_background')
        fig, ax = plt.subplots(figsize=(15, 7))
        ax.plot(history_df.index, history_df['net_worth'], color='cyan', label='Desempenho do Agente de RL')
        
        # Plot buy and hold
        buy_and_hold_data = test_data['close']
        buy_and_hold = (buy_and_hold_data / buy_and_hold_data.iloc[0]) * test_env.initial_balance
        ax.plot(buy_and_hold.index, buy_and_hold.values, color='gray', linestyle='--', label='Buy & Hold')
        
        # Plot buy/sell markers if position column exists
        if 'position' in history_df.columns:
            buys = history_df[history_df['position'] > history_df['position'].shift(1).fillna(0)]
            sells = history_df[history_df['position'] < history_df['position'].shift(1).fillna(0)]
            
            if not buys.empty:
                ax.scatter(buys.index, buys['net_worth'], color='green', marker='^', s=100, label='Compra')
            if not sells.empty:
                ax.scatter(sells.index, sells['net_worth'], color='red', marker='v', s=100, label='Venda')
        
        ax.set_title(f'Desempenho do Agente de RL vs. Buy & Hold')
        ax.set_ylabel('Patrimônio Líquido')
        ax.set_xlabel('Data')
        ax.legend()
        plt.grid(True, alpha=0.2)
        plt.savefig('performance.png')
        print("Performance plot saved to 'performance.png'")
    else:
        print("Cannot plot performance: 'net_worth' column not found in history DataFrame")
    
    return history_df


def main():
    """Main function to run the complete pipeline."""
    # Download data
    ticker = "PETR4.SA"
    print(f"Downloading data for {ticker}...")
    data = yf.download(ticker, start="2018-01-01", end="2024-01-01", auto_adjust=True)
    
    # Split data into train, validation, and test sets
    print("Preparing data...")
    feature_engineer = FeatureEngineer()
    data_featured = feature_engineer.transform(data)
    
    # Split data: 70% train, 15% validation, 15% test
    train_size = int(len(data_featured) * 0.7)
    validation_size = int(len(data_featured) * 0.15)
    
    train_data = data_featured[:train_size]
    validation_data = data_featured[train_size:train_size+validation_size]
    test_data = data_featured[train_size+validation_size:]
    
    print(f"Data split: Train={len(train_data)}, Validation={len(validation_data)}, Test={len(test_data)}")
    
    # Hyperparameter optimization
    print("\n--- INICIANDO OTIMIZAÇÃO DE HIPERPARÂMETROS ---")
    best_params = optimize_agent(train_data, validation_data, n_trials=20)
    
    # Train final model on train + validation data
    print("\n--- TREINANDO MODELO FINAL COM OS MELHORES HIPERPARÂMETROS ---")
    train_val_data = pd.concat([train_data, validation_data])
    
    # Extract parameters
    commission = best_params.get('commission', 0.001)
    slippage = best_params.get('slippage', 0.0005)
    reward_type = best_params.get('reward_type', 'sharpe')
    
    # Create environment
    train_env = DummyVecEnv([
        lambda: TradingEnv(
            train_val_data, 
            commission_pct=commission, 
            slippage_pct=slippage,
            reward_type=reward_type
        )
    ])
    
    # Create model parameters
    model_params = {
        'policy': 'MlpPolicy',
        'learning_rate': best_params.get('learning_rate', 0.0003),
        'n_steps': best_params.get('n_steps', 2048),
        'batch_size': best_params.get('batch_size', 64),
        'gamma': best_params.get('gamma', 0.99),
        'policy_kwargs': {'net_arch': eval(best_params.get('net_arch', '[64, 64]'))},
        'tensorboard_log': "./ppo_trading_tensorboard/"
    }
    
    # Create and train agent
    agent = TradingAgent(model_params)
    agent.train(train_env, total_timesteps=100000, progress_bar=True)
    
    # Save the model
    agent.save("ppo_trading_agent_final")
    
    # Evaluate on test data
    print("\n--- AVALIANDO O AGENTE EM DADOS NÃO VISTOS ---")
    history_df = evaluate_agent(
        agent, 
        test_data, 
        commission_pct=commission, 
        slippage_pct=slippage,
        reward_type=reward_type
    )
    
    # Analyze feature importance
    print("\n--- ANALISANDO IMPORTÂNCIA DAS FEATURES ---")
    if SHAP_AVAILABLE:
        analyze_feature_importance(agent.model, test_data)
    else:
        print("SHAP não está disponível. Análise de importância de features desabilitada.")
    
    print("\n--- PIPELINE COMPLETO FINALIZADO ---")


if __name__ == "__main__":
    main()