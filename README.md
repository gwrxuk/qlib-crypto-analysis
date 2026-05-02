# Crypto Protocol Health Analysis with Qlib

A quantitative research pipeline that uses [Microsoft qlib](https://github.com/microsoft/qlib) to analyse cryptocurrency market health across 15 major assets. The project covers data ingestion, Alpha158 feature engineering, LightGBM model training, portfolio backtesting, and statistical analysis including VAR, Granger causality, and volatility regime detection.

## Assets

BTC, ETH, SOL, BNB, AVAX, ADA, LINK, DOT, MATIC, UNI, ATOM, LTC, XRP, DOGE, NEAR  
Daily OHLCV sourced from Yahoo Finance via yfinance (2021-01-04 → 2025-04-27).

## Pipeline

| Script | Purpose |
|---|---|
| `01_build_qlib_data.py` | Download OHLCV and write qlib native binary data store |
| `02_qlib_pipeline.py` | Alpha158 features → LGBModel training → TopkDropout backtest |
| `03_statistical_analysis.py` | VAR(10), Granger causality, ADF, volatility regimes, Amihud liquidity |

## Key Results

- **Alpha158**: 158 features, 13,620 training samples across 15 assets
- **Model**: LGBModel (LightGBM, MSE loss), IC = −0.052 on test set (2024-04-01 → 2025-04-27)
- **Backtest**: TopkDropoutStrategy (top-5, dropout=0.25), benchmark BTCUSD
- **Granger causality**: BTC → ETH significant at p < 0.05; BTC/ETH jointly lead SOL
- **Regimes**: Low/mid/high volatility regimes identified via 30-day rolling annualised vol (33rd/67th percentile thresholds)
- **Amihud illiquidity**: DOGE and MATIC show highest illiquidity ratios; BTC lowest

## Setup

```bash
pip install qlib lightgbm yfinance statsmodels matplotlib pandas numpy
```

```bash
python 01_build_qlib_data.py   # build qlib data store (~2 min)
python 02_qlib_pipeline.py     # train model + backtest (~5 min)
python 03_statistical_analysis.py  # statistical tests + figures
```

## Outputs

- `qlib_data/` — qlib binary data store (calendars, instruments, feature `.bin` files)
- `mlruns/` — MLflow experiment tracking
- `alpha158_ic.csv`, `model_ic_daily.csv` — Information Coefficient time series
- `granger_causality.csv`, `regime_stats.csv`, `stats_results.json` — statistical results
- `risk_analysis_qlib.csv`, `factor_ic.csv`, `lgbm_feature_importance.csv` — model diagnostics

## Requirements

- Python 3.9+
- qlib 0.9.7
- LightGBM
- statsmodels
- yfinance
- matplotlib, pandas, numpy
