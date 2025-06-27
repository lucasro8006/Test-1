# Sistema de Trading com Reinforcement Learning

Este projeto implementa um sistema completo de trading baseado em Reinforcement Learning (RL), utilizando a biblioteca stable-baselines3 e um ambiente customizado baseado em Gymnasium.

## Visão Geral

O agente de trading utiliza Proximal Policy Optimization (PPO) para aprender estratégias ótimas de negociação baseadas em indicadores técnicos. O sistema inclui:

- Engenharia de features para indicadores técnicos
- Ambiente customizado Gymnasium para simulação de trading
- Otimização de hiperparâmetros com Optuna
- Validação cruzada encadeada com purga para séries temporais
- Análise de importância de features com SHAP
- Avaliação abrangente de desempenho

## Estrutura do Projeto

```
.
├── src/                      # Código-fonte principal
│   ├── data/                 # Módulos para carregamento e preparação de dados
│   ├── environments/         # Ambientes de trading para RL
│   ├── features/             # Engenharia de features
│   ├── models/               # Modelos de agentes de trading
│   ├── evaluation/           # Avaliação de desempenho
│   └── utils/                # Utilitários diversos
├── models/                   # Modelos treinados
├── results/                  # Resultados e gráficos
├── main.py                   # Script principal
├── quick_test_fixed.py       # Script para teste rápido
├── trading_agent.py          # Versão anterior do script principal
└── requirements.txt          # Dependências
```

## Componentes Principais

### 1. Engenharia de Features (`FeatureEngineer`)

Classe responsável por transformar dados brutos de preços em features técnicas para o modelo:
- Indicadores técnicos (MACD, RSI, Bollinger Bands, etc.)
- Normalização de features
- Preparação de dados para o ambiente de trading

### 2. Ambiente de Trading (`TradingEnv`)

Ambiente customizado baseado em Gymnasium que simula um mercado de trading:
- Suporte a diferentes tipos de recompensa (PnL, Sharpe, Calmar)
- Simulação realista com custos de transação (comissão e slippage)
- Rastreamento de histórico de operações e métricas

### 3. Agente de Trading (`TradingAgent`)

Wrapper para o modelo PPO da stable-baselines3:
- Métodos para treinamento, previsão e persistência
- Integração com callbacks para monitoramento
- Compatibilidade com diferentes políticas e arquiteturas de rede

### 4. Validação Cruzada Encadeada (`PurgedTimeSeriesSplit`)

Validação cruzada especializada para séries temporais financeiras:
- Purga para remover amostras de treino que se sobrepõem aos dados de teste
- Embargo para prevenir vazamento de dados

## Como Usar

### Instalação

```bash
pip install -r requirements.txt
```

### Execução Básica

```bash
python main.py --ticker PETR4.SA --start_date 2018-01-01 --end_date 2023-12-31
```

### Otimização de Hiperparâmetros

```bash
python main.py --ticker PETR4.SA --optimize --n_trials 50
```

### Teste Rápido

```bash
python quick_test_fixed.py
```

## Métricas de Desempenho

O agente é avaliado usando:
- Retorno Total
- Sharpe Ratio (retorno ajustado ao risco)
- Drawdown Máximo
- Calmar Ratio (retorno relativo ao drawdown máximo)
- Taxa de Acerto (win rate)
- Fator de Lucro (profit factor)

## Recursos Avançados

### 1. Validação Cruzada Encadeada

Implementação de validação cruzada com purga e embargo para evitar vazamento de dados em séries temporais financeiras.

### 2. Análise de Importância de Features

Utilização de SHAP (SHapley Additive exPlanations) para entender quais features são mais importantes para as decisões do agente.

### 3. Diferentes Tipos de Recompensa

Suporte a diferentes funções de recompensa:
- PnL simples
- Sharpe Ratio diferencial
- Calmar Ratio diferencial

### 4. Custos de Transação Realistas

Simulação de custos de transação:
- Comissão percentual
- Slippage baseado em spread

## Próximos Passos

- [ ] Suporte a múltiplos ativos (portfolio)
- [ ] Integração com dados fundamentais
- [ ] Análise de sentimento de mercado
- [ ] Backtesting mais robusto com dados de tick
- [ ] Interface web para visualização de resultados
- [ ] Implementação de outros algoritmos de RL (SAC, TD3, etc.)
- [ ] Suporte a operações de venda a descoberto (short)