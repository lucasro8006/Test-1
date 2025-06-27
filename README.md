# Trading Agent with Reinforcement Learning

This project implements a complete pipeline for training and evaluating a trading agent using Reinforcement Learning (RL) with stable-baselines3 and a custom Gymnasium environment.

## Overview

The trading agent uses Proximal Policy Optimization (PPO) to learn optimal trading strategies based on technical indicators. The system includes:

- Feature engineering for technical indicators
- Custom Gymnasium environment for trading simulation
- Hyperparameter optimization with Optuna
- Purged time-series cross-validation
- Feature importance analysis with SHAP
- Comprehensive performance evaluation

## Project Structure

- `trading_agent.py`: Main script containing the complete pipeline
- `robo.py`: Original prototype script (kept for reference)

## Key Components

### FeatureEngineer

Responsible for calculating and normalizing technical indicators:
- MACD (Moving Average Convergence Divergence)
- RSI (Relative Strength Index)
- Bollinger Bands

### TradingEnv

Custom Gymnasium environment that simulates trading with:
- Realistic transaction costs (commission and slippage)
- Multiple reward functions (PnL, Sharpe Ratio, Calmar Ratio)
- Detailed trade history tracking

### TradingAgent

Wrapper for the PPO model that handles:
- Training on historical data
- Making predictions for new observations
- Saving and loading trained models

### PurgedTimeSeriesSplit

Time-series cross-validation with:
- Purging to remove training samples that overlap with test data
- Embargo to prevent data leakage

## Usage

To run the complete pipeline:

```bash
python trading_agent.py
```

This will:
1. Download historical data for PETR4.SA
2. Split data into train, validation, and test sets
3. Optimize hyperparameters using Optuna
4. Train the final model with the best parameters
5. Evaluate performance on unseen test data
6. Analyze feature importance with SHAP

## Requirements

- pandas
- numpy
- yfinance
- gymnasium
- stable-baselines3
- matplotlib
- optuna
- shap

Install dependencies:

```bash
pip install pandas numpy yfinance gymnasium "stable-baselines3[extra]" matplotlib optuna shap
```

## Performance Metrics

The agent is evaluated using:
- Total Return
- Sharpe Ratio (risk-adjusted return)
- Maximum Drawdown
- Calmar Ratio (return relative to maximum drawdown)

## Future Improvements

- Implement multi-asset portfolio optimization
- Add more sophisticated features (sentiment analysis, macroeconomic indicators)
- Explore different RL algorithms (SAC, TD3)
- Implement online learning for continuous model updates