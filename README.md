# Sentiment-Driven HSI Trading Strategy Backtesting

Commissioned by a Hong Kong fintech firm to design, evaluate, and recommend
sentiment-based trading strategies for the Hang Seng Index.

## Overview
Built an end-to-end Python pipeline evaluating three sentiment-driven strategies
against buy-and-hold using three years of proprietary market and social media data.

## Methods
- MICE multiple imputation for 26% missing sentiment values
- Three strategy variants: basic sentiment, contrarian extreme fear/greed, gap + sentiment
- Risk evaluation: Sharpe, Sortino, max drawdown, win rate

## Key Results
| Strategy | Return | Sharpe | Sortino | Max Drawdown |
|----------|--------|--------|---------|--------------|
| Contrarian | 26.35% | 0.63 | 1.00 | -16.58% |
| Gap/Sentiment | 8.79% | 0.26 | 0.39 | -24.97% |
| Basic | -7.92% | -0.03 | -0.05 | -36.76% |
| Buy & Hold | 3.05% | 0.17 | 0.26 | -35.49% |

## Note
Data is proprietary and not included. Code demonstrates methodology and pipeline architecture.

## Tools
Python, pandas, matplotlib, scikit-learn (IterativeImputer), QuantStats
