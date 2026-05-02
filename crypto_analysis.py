"""
Cryptocurrency Protocol Health Analysis Pipeline
Uses qlib factor framework + live market data for academic paper
"""

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats
from scipy.stats import jarque_bera, shapiro, kurtosis, skew
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.stats.diagnostic import acorr_ljungbox
import statsmodels.api as sm
import json
from datetime import datetime, timedelta

OUTPUT_DIR = "/home/ubuntu/ky/agents/qlib_paper"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─── 1. DATA ACQUISITION ────────────────────────────────────────────────────

TICKERS = {
    "BTC-USD": "Bitcoin",
    "ETH-USD": "Ethereum",
    "SOL-USD": "Solana",
    "BNB-USD": "BNB Chain",
    "AVAX-USD": "Avalanche",
}

START = "2021-01-01"
END   = "2025-04-30"

print("Fetching price data from Yahoo Finance …")
raw = yf.download(list(TICKERS.keys()), start=START, end=END, auto_adjust=True, progress=False)
close = raw["Close"].dropna(how="all")
volume = raw["Volume"].dropna(how="all")
high   = raw["High"].dropna(how="all")
low    = raw["Low"].dropna(how="all")

print(f"  Rows: {len(close)}, Cols: {list(close.columns)}")

# ─── 2. RETURNS & BASIC STATS ────────────────────────────────────────────────

returns = close.pct_change().dropna()
log_ret = np.log(close / close.shift(1)).dropna()

print("\n=== Descriptive Statistics (Daily Returns) ===")
desc = returns.describe()
desc.loc["skewness"] = returns.apply(skew)
desc.loc["kurtosis"] = returns.apply(kurtosis)
desc.loc["jarque_bera_p"] = returns.apply(lambda s: jarque_bera(s.dropna())[1])
print(desc.round(6).to_string())
desc.to_csv(f"{OUTPUT_DIR}/descriptive_stats.csv")

# ─── 3. QLIB-STYLE ALPHA FACTORS ────────────────────────────────────────────
# We replicate the qlib alpha158 factor library on crypto OHLCV data

def rolling_z(s, w):
    return (s - s.rolling(w).mean()) / s.rolling(w).std()

def rsi(close, window=14):
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = (-delta.clip(upper=0)).rolling(window).mean()
    rs   = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)

factors = pd.DataFrame(index=close.index)

for ticker in close.columns:
    c = close[ticker]
    v = volume[ticker]
    h = high[ticker]
    l = low[ticker]
    r = log_ret[ticker]

    # Momentum factors
    factors[f"{ticker}_mom5"]   = c / c.shift(5)  - 1
    factors[f"{ticker}_mom20"]  = c / c.shift(20) - 1
    factors[f"{ticker}_mom60"]  = c / c.shift(60) - 1

    # Volatility factors
    factors[f"{ticker}_vol10"]  = r.rolling(10).std() * np.sqrt(252)
    factors[f"{ticker}_vol30"]  = r.rolling(30).std() * np.sqrt(252)
    factors[f"{ticker}_vol60"]  = r.rolling(60).std() * np.sqrt(252)

    # Volume factors
    factors[f"{ticker}_volvol"] = np.log(v).rolling(5).std()
    factors[f"{ticker}_turnover"] = (v * c).rolling(20).mean()

    # Technical indicators
    factors[f"{ticker}_rsi14"]  = rsi(c, 14)
    factors[f"{ticker}_rsi28"]  = rsi(c, 28)

    # Mean-reversion factor (MACD signal)
    ema12 = c.ewm(span=12).mean()
    ema26 = c.ewm(span=26).mean()
    factors[f"{ticker}_macd"]   = (ema12 - ema26) / c

    # Bollinger Band position
    sma20 = c.rolling(20).mean()
    std20 = c.rolling(20).std()
    factors[f"{ticker}_bb_pos"] = (c - sma20) / (2 * std20)

    # Realized vs implied vol ratio (proxy: short/long vol)
    factors[f"{ticker}_vol_ratio"] = factors[f"{ticker}_vol10"] / factors[f"{ticker}_vol60"]

    # Liquidity: Amihud illiquidity ratio (|r|/volume_usd)
    vol_usd = v * c
    factors[f"{ticker}_amihud"] = (r.abs() / vol_usd.replace(0, np.nan)).rolling(20).mean() * 1e6

    # Price / 52-week high
    factors[f"{ticker}_52wk_pos"] = c / c.rolling(252).max()

factors = factors.dropna(how="all")
print(f"\nFactor matrix shape: {factors.shape}")

# ─── 4. PROTOCOL HEALTH METRICS ─────────────────────────────────────────────

print("\n=== Protocol Health Metrics ===")

# 4a. Annualised volatility (rolling 30-day)
ann_vol = log_ret.rolling(30).std() * np.sqrt(252) * 100
ann_vol_latest = ann_vol.iloc[-1]
print("\nLatest 30-day Annualised Volatility (%):")
print(ann_vol_latest.round(2).to_string())

# 4b. Sharpe ratio (252-day rolling)
rf_daily = 0.05 / 252
rolling_sharpe = (log_ret.rolling(252).mean() - rf_daily) / log_ret.rolling(252).std() * np.sqrt(252)
print("\nLatest Rolling Sharpe Ratio:")
print(rolling_sharpe.iloc[-1].round(3).to_string())

# 4c. Maximum drawdown
def max_drawdown(series):
    cum = (1 + series).cumprod()
    roll_max = cum.cummax()
    dd = (cum - roll_max) / roll_max
    return dd.min()

mdd = returns.apply(max_drawdown)
print("\nMax Drawdown (full period):")
print(mdd.round(4).to_string())

# 4d. Calmar ratio
total_return = (close.iloc[-1] / close.iloc[0]) - 1
years = (close.index[-1] - close.index[0]).days / 365.25
ann_return = (1 + total_return) ** (1 / years) - 1
calmar = ann_return / (-mdd)
print("\nCalmar Ratio:")
print(calmar.round(3).to_string())

# 4e. Tail risk: CVaR at 5%
def cvar(series, alpha=0.05):
    var = series.quantile(alpha)
    return series[series <= var].mean()

cvar_5 = returns.apply(cvar)
var_5  = returns.apply(lambda s: s.quantile(0.05))
print("\nVaR (5%) daily:")
print(var_5.round(4).to_string())
print("\nCVaR (5%) daily:")
print(cvar_5.round(4).to_string())

# 4f. Cross-asset correlation
corr = returns.corr()
print("\nReturn Correlation Matrix:")
print(corr.round(3).to_string())

# 4g. ADF stationarity test on log-prices
print("\nADF Test (log prices) – H0: unit root:")
for col in close.columns:
    adf_stat, pval, _, _, _, _ = adfuller(np.log(close[col].dropna()), autolag="AIC")
    print(f"  {col}: stat={adf_stat:.3f}, p={pval:.4f} ({'non-stationary' if pval>0.05 else 'stationary'})")

# ADF on returns
print("\nADF Test (log returns) – H0: unit root:")
for col in log_ret.columns:
    adf_stat, pval, _, _, _, _ = adfuller(log_ret[col].dropna(), autolag="AIC")
    print(f"  {col}: stat={adf_stat:.3f}, p={pval:.4f} ({'non-stationary' if pval>0.05 else 'stationary'})")

# 4h. VAR model for cross-asset dynamics
print("\nFitting VAR model …")
var_data = log_ret.dropna()
var_model = VAR(var_data)
var_lag_sel = var_model.select_order(maxlags=10)
best_lag = max(1, var_lag_sel.aic)
var_fit = var_model.fit(best_lag)
print(var_fit.summary())

# Granger causality summary
print("\n=== Granger Causality (p-values) ===")
gc_results = {}
pairs = [(a, b) for a in log_ret.columns for b in log_ret.columns if a != b]
for caused, causing in pairs:
    try:
        gc_test = var_fit.test_causality(caused, [causing], kind='f')
        gc_results[f"{causing}→{caused}"] = gc_test.pvalue
    except Exception:
        gc_results[f"{causing}→{caused}"] = np.nan

gc_df = pd.Series(gc_results).rename("p_value").sort_values()
print(gc_df.round(4).to_string())
gc_df.to_csv(f"{OUTPUT_DIR}/granger_causality.csv")

# ─── 5. QLIB FACTOR IC ANALYSIS ─────────────────────────────────────────────
print("\n=== Information Coefficient (IC) Analysis ===")

def compute_ic(factor_series, fwd_return, window=1):
    """Spearman rank IC between factor and forward return."""
    df = pd.concat([factor_series, fwd_return], axis=1).dropna()
    if len(df) < 30:
        return np.nan
    return df.iloc[:, 0].corr(df.iloc[:, 1], method="spearman")

ic_results = {}
for ticker in close.columns:
    fwd1  = log_ret[ticker].shift(-1)
    fwd5  = log_ret[ticker].rolling(5).sum().shift(-5)
    fwd20 = log_ret[ticker].rolling(20).sum().shift(-20)
    for fkey, fwd in [("1d", fwd1), ("5d", fwd5), ("20d", fwd20)]:
        for fcol in [c for c in factors.columns if c.startswith(ticker)]:
            ic = compute_ic(factors[fcol], fwd)
            ic_results[(ticker, fcol.replace(ticker+"_",""), fkey)] = ic

ic_df = pd.Series(ic_results).rename("IC")
ic_df.index.names = ["asset","factor","horizon"]
ic_df = ic_df.reset_index()
print(ic_df.sort_values("IC", key=abs, ascending=False).head(20).to_string(index=False))
ic_df.to_csv(f"{OUTPUT_DIR}/factor_ic.csv", index=False)

# ─── 6. REGIME DETECTION (HIDDEN MARKOV – simplified) ───────────────────────
print("\n=== Volatility Regime Clustering ===")

btc_ret = log_ret["BTC-USD"].dropna()
vol_30  = btc_ret.rolling(30).std() * np.sqrt(252)

# K-means style thresholds
low_vol_thresh  = vol_30.quantile(0.33)
high_vol_thresh = vol_30.quantile(0.67)

def regime(v):
    if pd.isna(v):  return "unknown"
    if v < low_vol_thresh:  return "low"
    if v < high_vol_thresh: return "medium"
    return "high"

vol_regime = vol_30.apply(regime)
regime_counts = vol_regime.value_counts()
print("BTC Volatility Regime Distribution:")
print(regime_counts.to_string())

# Returns by regime
regime_ret = pd.DataFrame({"return": btc_ret, "regime": vol_regime}).dropna()
regime_stats = regime_ret.groupby("regime")["return"].agg(["mean","std","count"])
regime_stats["annualized_mean"] = regime_stats["mean"] * 252
regime_stats["annualized_vol"]  = regime_stats["std"] * np.sqrt(252)
print("\nReturn stats by regime:")
print(regime_stats.round(6).to_string())
regime_stats.to_csv(f"{OUTPUT_DIR}/regime_stats.csv")

# ─── 7. MARKET MICROSTRUCTURE – LIQUIDITY ───────────────────────────────────
print("\n=== Market Microstructure / Liquidity Metrics ===")

# Bid-ask spread proxy: Roll measure
def roll_spread(returns):
    cov = returns.cov(returns.shift(1))
    if cov >= 0: return 0
    return 2 * np.sqrt(-cov)

roll = returns.apply(roll_spread)
print("Roll Spread Estimate (proxy for bid-ask spread):")
print(roll.round(6).to_string())

# Amihud illiquidity ratio
amihud_cols = [f"{t}_amihud" for t in close.columns]
amihud_latest = factors[amihud_cols].iloc[-30:].mean()
print("\nAmihud Illiquidity (20-day avg, latest month):")
print(amihud_latest.round(8).to_string())

# Volume trends
vol_change = volume.apply(lambda s: (s.rolling(30).mean().iloc[-1] / s.rolling(30).mean().iloc[-252] - 1) * 100)
print("\nVolume YoY Change (%):")
print(vol_change.round(2).to_string())

# ─── 8. CUMULATIVE PERFORMANCE & QLIB PORTFOLIO SIMULATION ──────────────────
print("\n=== Equal-Weight Portfolio vs. Individual Assets ===")

ew_ret = returns.mean(axis=1)
cum_ew  = (1 + ew_ret).cumprod()
cum_ind = (1 + returns).cumprod()

# Simple momentum strategy: long top-2 30d momentum each month
monthly_returns = returns.resample("M").apply(lambda x: (1+x).prod()-1)
mom_30 = close.resample("M").last().pct_change()

strategy_ret_list = []
for i in range(1, len(monthly_returns)):
    ranked = mom_30.iloc[i-1].dropna().rank(ascending=False)
    top2   = ranked[ranked <= 2].index
    if len(top2) == 0:
        strategy_ret_list.append(0)
    else:
        strategy_ret_list.append(monthly_returns.iloc[i][top2].mean())

strategy_monthly = pd.Series(strategy_ret_list, index=monthly_returns.index[1:])
cum_momentum = (1 + strategy_monthly).cumprod()

print(f"EW Portfolio CAGR:      {(cum_ew.iloc[-1]**(1/years)-1)*100:.2f}%")
print(f"Momentum Strategy CAGR: {(cum_momentum.iloc[-1]**(1/(years))-1)*100:.2f}%")

strat_sharpe = strategy_monthly.mean() / strategy_monthly.std() * np.sqrt(12)
print(f"Momentum Sharpe (annualised): {strat_sharpe:.3f}")

# ─── 9. SAVE NUMERICAL RESULTS ───────────────────────────────────────────────

results = {
    "period": f"{START} to {END}",
    "assets": list(TICKERS.keys()),
    "ann_volatility_latest": ann_vol_latest.round(2).to_dict(),
    "rolling_sharpe_latest": rolling_sharpe.iloc[-1].round(3).to_dict(),
    "max_drawdown": mdd.round(4).to_dict(),
    "calmar_ratio": calmar.round(3).to_dict(),
    "var_5pct": var_5.round(4).to_dict(),
    "cvar_5pct": cvar_5.round(4).to_dict(),
    "correlation_matrix": corr.round(3).to_dict(),
    "regime_distribution": regime_counts.to_dict(),
    "roll_spread": roll.round(6).to_dict(),
    "ew_portfolio_cagr": round((cum_ew.iloc[-1]**(1/years)-1)*100, 2),
    "momentum_strategy_cagr": round((cum_momentum.iloc[-1]**(1/years)-1)*100, 2),
    "momentum_sharpe": round(strat_sharpe, 3),
    "top_ic_factors": ic_df.sort_values("IC", key=abs, ascending=False).head(10).to_dict(orient="records"),
}

with open(f"{OUTPUT_DIR}/results.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"\nNumerical results saved to {OUTPUT_DIR}/results.json")

# ─── 10. FIGURES ─────────────────────────────────────────────────────────────

sns.set_theme(style="whitegrid", palette="husl")
COLORS = sns.color_palette("husl", len(close.columns))

# Figure 1: Normalised price performance
fig, axes = plt.subplots(2, 1, figsize=(14, 10))
norm_close = close / close.iloc[0] * 100
for i, col in enumerate(norm_close.columns):
    axes[0].plot(norm_close.index, norm_close[col], label=TICKERS[col], color=COLORS[i], linewidth=1.5)
axes[0].set_title("Normalised Price Performance (Base=100, Jan 2021 – Apr 2025)", fontsize=13)
axes[0].set_ylabel("Index (Base 100)")
axes[0].legend(loc="upper left")

# Annualised rolling volatility
for i, col in enumerate(ann_vol.columns):
    axes[1].plot(ann_vol.index, ann_vol[col], label=TICKERS[col], color=COLORS[i], linewidth=1.2, alpha=0.85)
axes[1].set_title("30-Day Rolling Annualised Volatility (%)", fontsize=13)
axes[1].set_ylabel("Volatility (%)")
axes[1].legend(loc="upper right")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/fig1_price_vol.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved fig1_price_vol.png")

# Figure 2: Correlation heatmap + return distributions
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdYlGn", center=0, vmin=-1, vmax=1,
            ax=axes[0], linewidths=0.5, mask=mask)
axes[0].set_title("Return Correlation Matrix", fontsize=13)
for i, col in enumerate(returns.columns):
    returns[col].plot.kde(ax=axes[1], label=TICKERS[col], color=COLORS[i], linewidth=1.5)
axes[1].axvline(0, color="black", linestyle="--", linewidth=0.8)
axes[1].set_title("Return Distribution (KDE)", fontsize=13)
axes[1].set_xlabel("Daily Return")
axes[1].legend()
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/fig2_corr_dist.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved fig2_corr_dist.png")

# Figure 3: Drawdown
fig, ax = plt.subplots(figsize=(14, 5))
for i, col in enumerate(returns.columns):
    cum = (1 + returns[col]).cumprod()
    dd  = (cum - cum.cummax()) / cum.cummax() * 100
    ax.fill_between(dd.index, dd, 0, alpha=0.35, color=COLORS[i], label=TICKERS[col])
ax.set_title("Drawdown from Peak (%)", fontsize=13)
ax.set_ylabel("Drawdown (%)")
ax.legend()
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/fig3_drawdown.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved fig3_drawdown.png")

# Figure 4: IC heatmap (top factors)
ic_pivot = ic_df[ic_df["horizon"]=="5d"].pivot_table(values="IC", index="factor", columns="asset")
fig, ax = plt.subplots(figsize=(12, 8))
sns.heatmap(ic_pivot, annot=True, fmt=".3f", cmap="RdYlGn", center=0,
            ax=ax, linewidths=0.4, vmin=-0.1, vmax=0.1)
ax.set_title("Factor Information Coefficient (5-day forward return)", fontsize=13)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/fig4_ic_heatmap.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved fig4_ic_heatmap.png")

# Figure 5: Cumulative performance + regime overlay
fig, axes = plt.subplots(2, 1, figsize=(14, 10))
for i, col in enumerate(cum_ind.columns):
    axes[0].plot(cum_ind.index, cum_ind[col], label=TICKERS[col], color=COLORS[i], linewidth=1.4)
axes[0].set_yscale("log")
axes[0].set_title("Cumulative Return (log scale)", fontsize=13)
axes[0].legend()

regime_num = vol_regime.map({"low": 0, "medium": 1, "high": 2, "unknown": np.nan})
axes[1].plot(btc_ret.index, (1+btc_ret).cumprod(), color="steelblue", linewidth=1.4, label="BTC cum. return")
high_vol_dates = vol_regime[vol_regime == "high"].index
for d in high_vol_dates:
    axes[1].axvspan(d, d + pd.Timedelta("1D"), alpha=0.08, color="red", linewidth=0)
axes[1].set_title("BTC Cumulative Return with High-Volatility Regimes (red shading)", fontsize=13)
axes[1].legend()
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/fig5_cumret_regime.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved fig5_cumret_regime.png")

# Figure 6: Qlib momentum strategy vs EW
fig, ax = plt.subplots(figsize=(12, 5))
cum_momentum.plot(ax=ax, label="Momentum Strategy", color="darkorange", linewidth=2)
ew_monthly = (1 + ew_ret).resample("M").prod()
ew_monthly_cum = ew_monthly.cumprod()
ew_monthly_cum.plot(ax=ax, label="Equal-Weight Portfolio", color="steelblue", linewidth=2, linestyle="--")
ax.set_title("Qlib Momentum Factor Strategy vs Equal-Weight Portfolio", fontsize=13)
ax.set_ylabel("Cumulative Return")
ax.legend()
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/fig6_strategy.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved fig6_strategy.png")

print("\n=== Analysis complete. All outputs written to", OUTPUT_DIR)
