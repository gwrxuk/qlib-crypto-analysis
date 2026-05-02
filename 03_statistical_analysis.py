"""
Phase 3: Full statistical analysis using qlib's expression engine for data
and standard econometric tools (statsmodels) for inference.
Computes all metrics needed for the academic paper.
"""

import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import jarque_bera, kurtosis, skew
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.vector_ar.var_model import VAR
import json

import qlib
from qlib.constant import REG_US
from qlib.data import D
from qlib.contrib.evaluate import risk_analysis

QLIB_DATA_DIR = "/home/ubuntu/ky/agents/qlib_paper/qlib_data"
OUTPUT_DIR    = "/home/ubuntu/ky/agents/qlib_paper"

CORE5 = ["BTCUSD","ETHUSD","SOLUSD","BNBUSD","AVAXUSD"]
ALL15 = [
    "BTCUSD","ETHUSD","SOLUSD","BNBUSD","AVAXUSD",
    "ADAUSD","LINKUSD","DOTUSD","MATICUSD","UNIUSD",
    "ATOMUSD","LTCUSD","XRPUSD","DOGEUSD","NEARUSD",
]
LABELS = {
    "BTCUSD":"Bitcoin","ETHUSD":"Ethereum","SOLUSD":"Solana",
    "BNBUSD":"BNB","AVAXUSD":"Avalanche","ADAUSD":"Cardano",
    "LINKUSD":"Chainlink","DOTUSD":"Polkadot","MATICUSD":"Polygon",
    "UNIUSD":"Uniswap","ATOMUSD":"Cosmos","LTCUSD":"Litecoin",
    "XRPUSD":"XRP","DOGEUSD":"Dogecoin","NEARUSD":"NEAR",
}

START = "2021-01-04"
END   = "2025-04-27"

print("Initialising qlib …")
qlib.init(provider_uri=QLIB_DATA_DIR, region=REG_US)

# ─── 1. FETCH PRICES & RETURNS VIA QLIB EXPRESSION ENGINE ───────────────────
print("\nFetching data via qlib D.features() …")
raw = D.features(
    instruments=ALL15,
    fields=["$close", "$volume", "$high", "$low", "$open",
            "$close/Ref($close,1)-1",          # daily return
            "Log($close/Ref($close,1))",        # log return
            "$close/Ref($close,1)-1",],
    start_time=START,
    end_time=END,
    freq="day",
)
raw.columns = ["close","volume","high","low","open","ret","logret","ret2"]

close_df  = raw["close"].unstack(level="instrument")
volume_df = raw["volume"].unstack(level="instrument")
ret_df    = raw["ret"].unstack(level="instrument").dropna(how="all")
logret_df = raw["logret"].unstack(level="instrument").dropna(how="all")

print(f"  close_df shape: {close_df.shape}")
print(f"  ret_df shape:   {ret_df.shape}")
print(f"  Sample dates:   {close_df.index[0].date()} → {close_df.index[-1].date()}")

# Focus returns for 5 core assets
ret5    = ret_df[CORE5].dropna()
logret5 = logret_df[CORE5].dropna()

# ─── 2. DESCRIPTIVE STATISTICS ───────────────────────────────────────────────
print("\n=== Descriptive Statistics (Core 5 Daily Returns) ===")
desc = ret5.describe()
desc.loc["skewness"]     = ret5.apply(skew)
desc.loc["kurtosis"]     = ret5.apply(kurtosis)
desc.loc["JB_p"]         = ret5.apply(lambda s: jarque_bera(s.dropna())[1])
print(desc.round(5).to_string())
desc.to_csv(f"{OUTPUT_DIR}/descriptive_stats.csv")

# ─── 3. RISK METRICS ─────────────────────────────────────────────────────────
print("\n=== Risk Metrics (Core 5) ===")
rf_daily = 0.05 / 365

ann_vol  = logret5.rolling(30).std() * np.sqrt(365) * 100
sharpe   = (logret5.rolling(365).mean() - rf_daily) / logret5.rolling(365).std() * np.sqrt(365)

def max_drawdown(s):
    cum = (1 + s).cumprod()
    return ((cum - cum.cummax()) / cum.cummax()).min()

def cvar(s, a=0.05):
    v = s.quantile(a)
    return s[s <= v].mean()

years = (ret5.index[-1] - ret5.index[0]).days / 365.25
total_ret = (close_df[CORE5].iloc[-1] / close_df[CORE5].iloc[0]) - 1
ann_ret   = (1 + total_ret) ** (1 / years) - 1
mdd       = ret5.apply(max_drawdown)
calmar    = ann_ret / (-mdd)
var5      = ret5.apply(lambda s: s.quantile(0.05))
cvar5     = ret5.apply(cvar)

print("\nLatest 30-day Ann. Volatility (%):")
print(ann_vol.iloc[-1].round(2).to_string())
print("\nRolling 365-day Sharpe:")
print(sharpe.iloc[-1].round(3).to_string())
print("\nMax Drawdown:")
print(mdd.round(4).to_string())
print("\nCalmar Ratio:")
print(calmar.round(3).to_string())
print("\nVaR (5%):")
print(var5.round(4).to_string())
print("\nCVaR (5%):")
print(cvar5.round(4).to_string())

# ─── 4. CORRELATION ──────────────────────────────────────────────────────────
corr5 = ret5.corr()
print("\nReturn Correlation Matrix (Core 5):")
print(corr5.round(3).to_string())

# ─── 5. ADF STATIONARITY ─────────────────────────────────────────────────────
print("\n=== ADF Tests ===")
print("Log-prices:")
for col in CORE5:
    lp = np.log(close_df[col].dropna())
    stat, pval = adfuller(lp, autolag="AIC")[:2]
    print(f"  {col}: stat={stat:.3f}, p={pval:.4f} ({'non-stationary' if pval>0.05 else 'stationary'})")

print("Log-returns:")
for col in CORE5:
    lr = logret5[col].dropna()
    stat, pval = adfuller(lr, autolag="AIC")[:2]
    print(f"  {col}: stat={stat:.3f}, p={pval:.4f} ({'non-stationary' if pval>0.05 else 'stationary'})")

# ─── 6. VAR + GRANGER CAUSALITY ──────────────────────────────────────────────
print("\n=== VAR Model + Granger Causality ===")
var_data = logret5.dropna()
var_sel  = VAR(var_data).select_order(maxlags=10)
best_lag = max(1, var_sel.aic)
print(f"  AIC-selected lag: {best_lag}")
var_fit  = VAR(var_data).fit(best_lag)
print(f"  AIC: {var_fit.aic:.4f}, BIC: {var_fit.bic:.4f}, log-lik: {var_fit.llf:.2f}")

gc = {}
for caused in CORE5:
    for causing in CORE5:
        if caused == causing: continue
        try:
            res = var_fit.test_causality(caused, [causing], kind="f")
            gc[f"{causing}→{caused}"] = float(res.pvalue)
        except Exception:
            gc[f"{causing}→{caused}"] = np.nan

gc_df = pd.Series(gc).rename("p_value").sort_values()
print("\nGranger Causality (p-values, sorted):")
print(gc_df.round(4).to_string())
gc_df.to_csv(f"{OUTPUT_DIR}/granger_causality.csv")

# ─── 7. REGIME DETECTION ─────────────────────────────────────────────────────
print("\n=== Volatility Regime Analysis (BTC) ===")
btc_ret = logret5["BTCUSD"].dropna()
vol30   = btc_ret.rolling(30).std() * np.sqrt(365)

low_t  = vol30.quantile(0.33)
high_t = vol30.quantile(0.67)

def regime(v):
    if pd.isna(v): return "unknown"
    return "low" if v < low_t else ("high" if v >= high_t else "medium")

vol_regime = vol30.apply(regime)
print(f"Regime counts:\n{vol_regime.value_counts().to_string()}")
print(f"Low vol threshold:  {low_t*100:.1f}%  High: {high_t*100:.1f}%")

rdf = pd.DataFrame({"ret": btc_ret, "regime": vol_regime}).dropna()
rs  = rdf.groupby("regime")["ret"].agg(["mean","std","count"])
rs["ann_return"] = rs["mean"] * 365
rs["ann_vol"]    = rs["std"]  * np.sqrt(365)
rs["sharpe"]     = rs["ann_return"] / rs["ann_vol"]
print("\nReturn stats by regime:")
print(rs.round(5).to_string())
rs.to_csv(f"{OUTPUT_DIR}/regime_stats.csv")

# ─── 8. LIQUIDITY — AMIHUD ───────────────────────────────────────────────────
print("\n=== Amihud Illiquidity ===")
amihud = {}
for col in CORE5:
    ret_abs = logret5[col].abs()
    vol_usd = (volume_df[col] * close_df[col])
    amihud[col] = (ret_abs / vol_usd.replace(0, np.nan)).rolling(20).mean() * 1e6

amihud_df = pd.DataFrame(amihud)
print("Latest 30-day mean Amihud illiquidity (×10⁶):")
print(amihud_df.iloc[-30:].mean().round(8).to_string())
amihud_df.to_csv(f"{OUTPUT_DIR}/amihud.csv")

# ─── 9. PORTFOLIO RETURNS (load backtest result) ─────────────────────────────
print("\n=== Portfolio Performance ===")
ew_ret   = ret_df[CORE5].dropna().mean(axis=1)
cum_ew   = (1 + ew_ret).cumprod()
ew_cagr  = (cum_ew.iloc[-1] ** (1/years) - 1) * 100
ew_sharpe = (ew_ret.mean() / ew_ret.std()) * np.sqrt(365)
ew_mdd   = max_drawdown(ew_ret)

print(f"Equal-Weight (Core 5)  CAGR: {ew_cagr:.2f}%,  Sharpe: {ew_sharpe:.3f},  MaxDD: {ew_mdd*100:.1f}%")

# Load qlib backtest portfolio values and compute returns
try:
    bp = pd.read_csv(f"{OUTPUT_DIR}/backtest_portfolio.csv", index_col=0, parse_dates=True)
    print(f"Backtest portfolio loaded: {bp.shape}")
    print(bp.tail(5).to_string())
except Exception:
    pass

# ─── 10. QLIB BACKTEST PORTFOLIO RETURNS (re-computed) ───────────────────────
# Re-run backtest quickly to extract returns properly
import qlib
from qlib.contrib.strategy.signal_strategy import TopkDropoutStrategy
from qlib.contrib.evaluate import backtest_daily

# Load saved predictions
pred_all_file = f"{OUTPUT_DIR}/pred_all.csv"
try:
    pred_all = pd.read_csv(pred_all_file, index_col=[0,1], parse_dates=True).iloc[:,0]
    pred_all.index.names = ["datetime","instrument"]
except Exception:
    pred_all = None
    print("  No saved predictions — skipping backtest re-run")

# ─── 11. FIGURES ─────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", palette="husl")
COLORS = sns.color_palette("husl", len(CORE5))

fig, axes = plt.subplots(2, 2, figsize=(16, 10))

# 11a. Normalised cumulative returns
cum5 = (1 + ret5).cumprod()
for i, col in enumerate(CORE5):
    axes[0,0].plot(cum5.index, cum5[col], label=LABELS[col], color=COLORS[i], lw=1.5)
axes[0,0].set_yscale("log")
axes[0,0].set_title("Cumulative Return (log scale, Core 5)", fontsize=12)
axes[0,0].legend(fontsize=9)

# 11b. Rolling 30-day volatility
for i, col in enumerate(CORE5):
    v = logret5[col].rolling(30).std() * np.sqrt(365) * 100
    axes[0,1].plot(v.index, v, label=LABELS[col], color=COLORS[i], lw=1.2, alpha=0.85)
axes[0,1].set_title("30-Day Rolling Annualised Volatility (%)", fontsize=12)
axes[0,1].legend(fontsize=9)

# 11c. Correlation heatmap
sns.heatmap(corr5.rename(columns=LABELS, index=LABELS),
            annot=True, fmt=".2f", cmap="RdYlGn", center=0,
            vmin=-1, vmax=1, ax=axes[1,0], linewidths=0.5)
axes[1,0].set_title("Return Correlation Matrix (Core 5)", fontsize=12)

# 11d. Drawdown
for i, col in enumerate(CORE5):
    cum = (1+ret5[col]).cumprod()
    dd  = (cum - cum.cummax()) / cum.cummax() * 100
    axes[1,1].fill_between(dd.index, dd, 0, alpha=0.3, color=COLORS[i], label=LABELS[col])
axes[1,1].set_title("Drawdown from Peak (%)", fontsize=12)
axes[1,1].legend(fontsize=9)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/fig_stats.png", dpi=150, bbox_inches="tight")
plt.close()
print("\nSaved fig_stats.png")

# 11e. BTC regime overlay
fig, ax = plt.subplots(figsize=(14, 5))
cum_btc = (1 + btc_ret).cumprod()
ax.plot(cum_btc.index, cum_btc, color="steelblue", lw=1.5, label="BTC cumulative return")
high_dates = vol_regime[vol_regime == "high"].index
for d in high_dates:
    ax.axvspan(d, d + pd.Timedelta("1D"), alpha=0.07, color="red", linewidth=0)
ax.set_title("BTC Cumulative Return with High-Volatility Regime (red shading)", fontsize=12)
ax.legend()
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/fig_regime.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved fig_regime.png")

# 11f. Granger causality heatmap
gc_matrix = pd.DataFrame(index=CORE5, columns=CORE5, dtype=float)
for k, v in gc.items():
    causing, caused = k.split("→")
    gc_matrix.loc[caused, causing] = v

fig, ax = plt.subplots(figsize=(8, 6))
mask = gc_matrix.isnull() | (gc_matrix.index.values[:, None] == gc_matrix.columns.values[None, :])
sns.heatmap(gc_matrix.astype(float),
            annot=True, fmt=".3f", cmap="RdYlGn_r",
            vmin=0, vmax=0.3, ax=ax, linewidths=0.5)
ax.set_title("Granger Causality p-values (VAR, row=caused, col=causing)", fontsize=11)
ax.set_xticklabels([LABELS[c] for c in CORE5], rotation=30)
ax.set_yticklabels([LABELS[c] for c in CORE5], rotation=0)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/fig_granger.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved fig_granger.png")

# ─── 12. SAVE ALL RESULTS ────────────────────────────────────────────────────
results = {
    "period":              f"{START} to {END}",
    "core_assets":         CORE5,
    "all_assets":          ALL15,
    "trading_days":        int(len(ret5)),
    "ann_volatility_pct":  ann_vol.iloc[-1].round(2).to_dict(),
    "sharpe_365d":         sharpe.iloc[-1].round(3).to_dict(),
    "max_drawdown":        mdd.round(4).to_dict(),
    "calmar_ratio":        calmar.round(3).to_dict(),
    "var_5pct":            var5.round(4).to_dict(),
    "cvar_5pct":           cvar5.round(4).to_dict(),
    "correlation":         corr5.round(3).to_dict(),
    "var_model_lag":       int(best_lag),
    "var_model_aic":       float(var_fit.aic),
    "granger_top5":        gc_df.head(5).to_dict(),
    "regime_stats":        rs.round(5).to_dict(),
    "ew_cagr_pct":         round(ew_cagr, 2),
    "ew_sharpe":           round(ew_sharpe, 3),
    "ew_maxdd_pct":        round(ew_mdd * 100, 2),
}

with open(f"{OUTPUT_DIR}/stats_results.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"\nAll statistical results saved to {OUTPUT_DIR}/stats_results.json")
print("\n=== Phase 3 complete ===")
