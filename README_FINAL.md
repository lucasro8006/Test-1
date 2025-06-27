# Sistema de Trading com Reinforcement Learning

Este projeto implementa um sistema completo de trading baseado em Reinforcement Learning (RL), utilizando as bibliotecas stable-baselines3 e Gymnasium para criar um agente de trading que aprende a tomar decisões de compra e venda em mercados financeiros.

## Visão Geral

O sistema foi completamente refatorado para uma arquitetura modular e orientada a objetos, com os seguintes componentes principais:

1. **Engenharia de Features**: Processamento de dados brutos de preços em indicadores técnicos
2. **Ambiente de Trading**: Simulação realista de mercado com custos de transação
3. **Agente de Trading**: Modelo de RL para tomada de decisões
4. **Avaliação de Desempenho**: Métricas financeiras e visualizações
5. **Otimização de Hiperparâmetros**: Busca eficiente por parâmetros ótimos

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
├── main.py                   # Script principal
├── quick_test_fixed.py       # Script para teste rápido
├── test_final.py             # Script para teste final com PPO
├── test_final_v2.py          # Script para teste final com A2C
└── requirements.txt          # Dependências
```

## Componentes Principais

### 1. Engenharia de Features (`FeatureEngineer`)

A classe `FeatureEngineer` processa dados brutos de preços e cria features técnicas para o modelo:

```python
fe = FeatureEngineer()
processed_data = fe.transform(data)
```

Features implementadas:
- Médias móveis (SMA 5, 20, 50)
- Distância percentual das médias móveis
- MACD (Moving Average Convergence Divergence)
- RSI (Relative Strength Index)
- Bollinger Bands
- ATR (Average True Range)
- Volume relativo
- Momentum
- Retornos diários e semanais

### 2. Ambiente de Trading (`TradingEnv`)

O ambiente `TradingEnv` simula um mercado de trading com:

```python
env = TradingEnv(
    data, 
    commission_pct=0.001, 
    slippage_pct=0.0005,
    reward_type='sharpe'
)
```

Características:
- Suporte a diferentes tipos de recompensa (PnL, Sharpe, Calmar)
- Simulação realista com custos de transação (comissão e slippage)
- Rastreamento de histórico de operações
- Compatibilidade com a API Gymnasium

### 3. Agente de Trading (`TradingAgent`)

A classe `TradingAgent` encapsula o modelo de RL:

```python
agent = TradingAgent(model_params)
agent.train(env, total_timesteps=50000)
```

Suporta:
- Treinamento com diferentes algoritmos (PPO, A2C)
- Previsão para novas observações
- Persistência do modelo treinado

### 4. Avaliação de Desempenho

O módulo de avaliação calcula métricas financeiras e gera visualizações:

```python
history_df, metrics = evaluate_agent(agent, test_env, test_data)
```

Métricas implementadas:
- Retorno total
- Sharpe Ratio
- Drawdown máximo
- Calmar Ratio
- Número de operações
- Taxa de acerto
- Lucro médio por operação

## Como Usar

### Instalação

```bash
pip install -r requirements.txt
```

### Execução Rápida

```bash
python quick_test_fixed.py
```

### Execução Completa

```bash
python main.py --ticker PETR4.SA --start_date 2018-01-01 --end_date 2023-12-31 --optimize
```

## Resultados e Análises

Os testes realizados demonstraram que:

1. O agente consegue aprender padrões nos dados e realizar operações lucrativas
2. A escolha da função de recompensa tem grande impacto no comportamento do agente
3. Custos de transação realistas são essenciais para evitar overtrading
4. A exploração (entropy coefficient) é crucial para que o agente realize operações

### Exemplo de Desempenho

Em um dos testes, o agente alcançou:
- Retorno total: 9.36%
- Sharpe Ratio: 0.44
- Drawdown máximo: -39.10%
- Calmar Ratio: 0.48

## Melhorias Futuras

- [ ] Implementar suporte a múltiplos ativos (portfolio)
- [ ] Adicionar dados fundamentais e macroeconômicos
- [ ] Implementar análise de sentimento de mercado
- [ ] Desenvolver backtesting mais robusto com dados de tick
- [ ] Criar interface web para visualização de resultados
- [ ] Implementar outros algoritmos de RL (SAC, TD3)
- [ ] Adicionar suporte a operações de venda a descoberto (short)

## Conclusão

Este projeto demonstra a aplicação de técnicas de Reinforcement Learning para trading algorítmico, com uma arquitetura modular e extensível que permite fácil experimentação com diferentes modelos, features e estratégias.

A abordagem baseada em RL oferece vantagens sobre métodos tradicionais, como a capacidade de aprender diretamente a partir dos dados sem necessidade de regras explícitas, adaptação a diferentes condições de mercado, e otimização direta para métricas financeiras como o Sharpe Ratio.