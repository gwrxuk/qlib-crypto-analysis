"""
Phase 2: Real qlib pipeline — corrected
  - qlib.init() from our binary crypto data store
  - D.features() with qlib's expression engine
  - Alpha158 feature engineering (genuine qlib computations)
  - LGBModel training with raw-return labels (no CSRankNorm — only 5 assets)
  - backtest_daily() with crypto exchange parameters
  - IC / ICIR, risk_analysis, feature importance
"""

import os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pathlib import Path

import qlib
from qlib.constant import REG_US
from qlib.data import D
from qlib.contrib.data.handler import Alpha158
from qlib.data.dataset import DatasetH
from qlib.contrib.model.gbdt import LGBModel
from qlib.contrib.strategy.signal_strategy import TopkDropoutStrategy
from qlib.contrib.evaluate import backtest_daily, risk_analysis

QLIB_DATA_DIR = "/home/ubuntu/ky/agents/qlib_paper/qlib_data"
OUTPUT_DIR    = "/home/ubuntu/ky/agents/qlib_paper"
INSTRUMENTS = [
    "BTCUSD","ETHUSD","SOLUSD","BNBUSD","AVAXUSD",
    "ADAUSD","LINKUSD","DOTUSD","MATICUSD","UNIUSD",
    "ATOMUSD","LTCUSD","XRPUSD","DOGEUSD","NEARUSD",
]
CORE5 = ["BTCUSD","ETHUSD","SOLUSD","BNBUSD","AVAXUSD"]  # focus assets for paper

TRAIN_START = "2021-01-04"
TRAIN_END   = "2023-06-30"
VALID_START = "2023-07-01"
VALID_END   = "2024-03-31"
TEST_START  = "2024-04-01"
TEST_END    = "2025-04-27"   # 2 days before calendar end to avoid out-of-bounds

# ─── 1. INIT ─────────────────────────────────────────────────────────────────
print("Initialising qlib …")
qlib.init(provider_uri=QLIB_DATA_DIR, region=REG_US)
print("  qlib initialised ✓")

# ─── 2. VERIFY DATA VIA QLIB EXPRESSION ENGINE ───────────────────────────────
print("\nQuerying qlib expression engine …")
calendar = D.calendar(freq="day")
print(f"  Calendar: {len(calendar)} trading days  {calendar[0].date()} → {calendar[-1].date()}")

instruments_info = D.list_instruments(D.instruments("all"), freq="day")
print(f"  Registered instruments: {list(instruments_info.keys())}")

# Real expressions from Alpha158 — computed by qlib's engine
expr_df = D.features(
    instruments=INSTRUMENTS,
    fields=[
        "$close",
        "$open",
        "$volume",
        "($close-$open)/$open",            # body return
        "Log($volume+1)",                  # log volume
        "$close/Ref($close,5)-1",          # 5-day momentum
        "$close/Ref($close,20)-1",         # 20-day momentum
        "Std($close/Ref($close,1)-1,20)",  # 20-day realised vol
        "Mean($volume,5)/Mean($volume,20)",# relative volume
        "($high-$low)/$close",             # daily range
    ],
    start_time=TRAIN_START,
    end_time=TEST_END,
    freq="day",
)
expr_df.columns = ["close","open","volume","body_ret","log_vol",
                   "mom5","mom20","vol20","rel_vol","range"]
print(f"\n  Expression query result: {expr_df.shape}")
print(expr_df.dropna().head(8).round(5).to_string())

# Save expression features
expr_df.to_csv(f"{OUTPUT_DIR}/qlib_expressions.csv")

# ─── 3. ALPHA158 DATASET — raw returns as label (no CSRankNorm) ───────────────
print("\n\nBuilding Alpha158 dataset …")

handler = Alpha158(
    instruments=INSTRUMENTS,
    start_time=TRAIN_START,
    end_time=TEST_END,
    fit_start_time=TRAIN_START,
    fit_end_time=TRAIN_END,
    infer_processors=[
        {"class": "RobustZScoreNorm",
         "kwargs": {"fields_group": "feature", "clip_outlier": True}},
        {"class": "Fillna", "kwargs": {"fields_group": "feature"}},
    ],
    learn_processors=[
        {"class": "DropnaLabel"},
        # NO CSRankNorm: raw cross-sectional returns with 5 assets
    ],
)

dataset = DatasetH(
    handler=handler,
    segments={
        "train": (TRAIN_START, TRAIN_END),
        "valid": (VALID_START, VALID_END),
        "test":  (TEST_START,  TEST_END),
    },
)

train_df = dataset.prepare("train", col_set=["feature", "label"])
valid_df = dataset.prepare("valid", col_set=["feature", "label"])
test_df  = dataset.prepare("test",  col_set=["feature", "label"])

feat_cols = [c[1] for c in train_df.columns if c[0] == "feature"]
print(f"  Alpha158 features: {len(feat_cols)}")
print(f"  Train: {train_df.shape}, Valid: {valid_df.shape}, Test: {test_df.shape}")
print(f"  Sample feature names: {feat_cols[:8]}")

# Label distribution check
labels_train = train_df["label"].iloc[:, 0].dropna()
print(f"\n  Label stats (raw 2-day fwd returns):")
print(f"    mean={labels_train.mean():.5f}, std={labels_train.std():.5f}, "
      f"min={labels_train.min():.4f}, max={labels_train.max():.4f}")

# ─── 4. ALPHA158 FEATURE IC ANALYSIS ─────────────────────────────────────────
print("\nComputing Alpha158 feature IC (Spearman, test set) …")

test_feat  = test_df["feature"]   # columns are now simple strings: "KMID", etc.
test_label = test_df["label"].iloc[:, 0]

# Cross-sectional IC per day, then average
def daily_ic(feature_series, label_series):
    df = pd.concat([feature_series.rename("f"), label_series.rename("l")], axis=1).dropna()
    if df.empty:
        return pd.Series(dtype=float, name="IC")
    def ic_day(g):
        return g["f"].corr(g["l"], method="spearman") if len(g) >= 2 else np.nan
    return df.groupby(level="datetime").apply(ic_day)

ic_per_feature = {}
for feat in feat_cols:
    daily = daily_ic(test_feat[feat], test_label)
    ic_per_feature[feat] = daily.mean()

ic_series = pd.Series(ic_per_feature).sort_values(key=abs, ascending=False)
print("Top 20 Alpha158 ICs (cross-sectional, test period):")
print(ic_series.head(20).round(5).to_string())
ic_series.to_csv(f"{OUTPUT_DIR}/alpha158_ic.csv", header=["IC"])

# ─── 5. LGBM MODEL TRAINING ──────────────────────────────────────────────────
print("\nTraining LGBModel …")

model = LGBModel(
    loss="mse",
    colsample_bytree=0.8879,
    learning_rate=0.02,
    subsample=0.8789,
    lambda_l1=20.0,
    lambda_l2=50.0,
    max_depth=6,
    num_leaves=63,
    num_threads=4,
    num_boost_round=500,
    early_stopping_rounds=50,
    verbose_eval=50,
)
model.fit(dataset)
print("  LGBModel fit complete ✓")

# Feature importance
print("\nTop 20 features by LGBModel importance:")
try:
    fi = model.get_feature_importance()
    print(fi.head(20).to_string())
    fi.to_csv(f"{OUTPUT_DIR}/lgbm_feature_importance.csv")
except Exception as e:
    print(f"  ({e})")
    fi = None

# ─── 6. PREDICTIONS & SIGNAL IC ──────────────────────────────────────────────
print("\nGenerating predictions …")
pred_test = model.predict(dataset, segment="test")
pred_all  = pd.concat([
    model.predict(dataset, segment="train"),
    model.predict(dataset, segment="valid"),
    pred_test,
]).sort_index()

# Daily cross-sectional IC of model predictions vs. true labels
test_lbl = test_df["label"].iloc[:, 0]
signal_ic_daily = daily_ic(pred_test.rename("pred"), test_lbl)
print(f"\nLGBModel signal IC on test set:")
print(f"  Mean IC : {signal_ic_daily.mean():.5f}")
print(f"  IC Std  : {signal_ic_daily.std():.5f}")
valid_ic = signal_ic_daily.std()
icir = signal_ic_daily.mean() / signal_ic_daily.std() if valid_ic > 0 else 0.0
print(f"  ICIR    : {icir:.4f}")
print(f"  IC > 0%  : {(signal_ic_daily > 0).mean()*100:.1f}%")

signal_ic_daily.to_csv(f"{OUTPUT_DIR}/model_ic_daily.csv", header=["IC"])

# ─── 7. QLIB BACKTEST VIA backtest_daily ─────────────────────────────────────
print("\nRunning qlib backtest_daily …")

# Strategy uses the combined predictions across all periods
strategy_config = {
    "class":       "TopkDropoutStrategy",
    "module_path": "qlib.contrib.strategy.signal_strategy",
    "kwargs": {
        "signal":      pred_all,
        "topk":        2,    # hold top-2 of 5 assets
        "n_drop":      1,
        "hold_thresh": 1,
    },
}

# Crypto exchange: no circuit breakers, small fee, deal at next-day open
exchange_kwargs = {
    "limit_threshold": None,     # no circuit breakers in crypto
    "deal_price":     "$close",  # execute at day close
    "open_cost":       0.001,    # 0.1% taker fee
    "close_cost":      0.001,
    "min_cost":        0.0,      # no minimum ticket size
}

portfolio_metric_dict, indicator_dict = backtest_daily(
    start_time=TEST_START,
    end_time=TEST_END,
    strategy=strategy_config,
    account=1_000_000,
    benchmark="BTCUSD",
    exchange_kwargs=exchange_kwargs,
)
print("  backtest_daily complete ✓")

# ─── 8. EXTRACT PORTFOLIO METRICS ────────────────────────────────────────────
print("\nPortfolio metrics (day frequency):")
freq_key = list(portfolio_metric_dict.keys())[0]
port_metrics = portfolio_metric_dict[freq_key]

if hasattr(port_metrics, "to_frame"):
    print(port_metrics.to_frame().head(20))
elif isinstance(port_metrics, dict):
    for k, v in port_metrics.items():
        print(f"  {k}: {v}")
else:
    print(f"  type={type(port_metrics)}")
    print(port_metrics)

# ─── 9. RISK ANALYSIS ────────────────────────────────────────────────────────
print("\nRisk analysis:")
try:
    # port_metrics may contain a returns series
    if hasattr(port_metrics, "index"):
        ra = risk_analysis(port_metrics)
    else:
        # Try the indicator dict
        ind_key = list(indicator_dict.keys())[0]
        ind_df  = indicator_dict[ind_key]
        ra = risk_analysis(ind_df)
    print(ra)
    ra.to_csv(f"{OUTPUT_DIR}/risk_analysis_qlib.csv")
except Exception as e:
    print(f"  ({e})")
    ra = None

# Indicator summary
print("\nIndicator metrics:")
try:
    ind_key = list(indicator_dict.keys())[0]
    ind_df  = indicator_dict[ind_key]
    print(type(ind_df))
    print(ind_df if hasattr(ind_df, "__str__") else str(ind_df))
    if hasattr(ind_df, "to_csv"):
        ind_df.to_csv(f"{OUTPUT_DIR}/backtest_indicators.csv")
except Exception as e:
    print(f"  ({e})")

# ─── 10. FIGURES ─────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid")

fig, axes = plt.subplots(2, 2, figsize=(16, 10))

# 10a. Alpha158 feature IC bar
top20_ic = ic_series.head(20)
colors = ["#2ca02c" if v > 0 else "#d62728" for v in top20_ic.values]
axes[0, 0].barh(range(len(top20_ic)), top20_ic.values, color=colors, alpha=0.85)
axes[0, 0].set_yticks(range(len(top20_ic)))
axes[0, 0].set_yticklabels(top20_ic.index, fontsize=8)
axes[0, 0].axvline(0, color="black", lw=0.8)
axes[0, 0].set_title("Alpha158 Feature IC (Cross-Sectional, Test Period)", fontsize=11)
axes[0, 0].set_xlabel("Mean Spearman IC")

# 10b. Model daily IC
if not signal_ic_daily.empty:
    axes[0, 1].bar(range(len(signal_ic_daily)), signal_ic_daily.values,
                   color=["#2ca02c" if v > 0 else "#d62728" for v in signal_ic_daily.values],
                   alpha=0.7, width=1.0)
    axes[0, 1].axhline(signal_ic_daily.mean(), color="navy", lw=1.5, linestyle="--",
                       label=f"Mean={signal_ic_daily.mean():.4f}")
    axes[0, 1].axhline(0, color="black", lw=0.6)
    axes[0, 1].set_title(f"LGBModel Daily IC (Test Period, ICIR={icir:.3f})", fontsize=11)
    axes[0, 1].set_ylabel("Spearman IC")
    axes[0, 1].legend(fontsize=9)

# 10c. LGB feature importance
if fi is not None and len(fi) > 0:
    fi_top = fi.head(20)
    axes[1, 0].barh(range(len(fi_top)), fi_top.values, color="steelblue", alpha=0.85)
    axes[1, 0].set_yticks(range(len(fi_top)))
    axes[1, 0].set_yticklabels(fi_top.index, fontsize=8)
    axes[1, 0].set_title("LGBModel Feature Importance (Top 20)", fontsize=11)
    axes[1, 0].set_xlabel("Importance")

# 10d. Momentum in expression features (20-day) — by instrument
mom20_df = expr_df["mom20"].unstack(level="instrument")
for col in mom20_df.columns:
    axes[1, 1].plot(mom20_df.index, mom20_df[col], label=col, alpha=0.8)
axes[1, 1].axhline(0, color="black", lw=0.6)
axes[1, 1].set_title("20-Day Momentum (qlib Expression Engine)", fontsize=11)
axes[1, 1].set_ylabel("Momentum")
axes[1, 1].legend(fontsize=8)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/fig_qlib_analysis.png", dpi=150, bbox_inches="tight")
plt.close()
print("\nSaved fig_qlib_analysis.png")

# ─── 11. SUMMARY JSON ────────────────────────────────────────────────────────
qlib_summary = {
    "qlib_version":              qlib.__version__,
    "calendar_days":             len(calendar),
    "instruments":               INSTRUMENTS,
    "alpha158_feature_count":    len(feat_cols),
    "train_samples":             int(len(train_df)),
    "valid_samples":             int(len(valid_df)),
    "test_samples":              int(len(test_df)),
    "label_mean_train":          float(labels_train.mean()),
    "label_std_train":           float(labels_train.std()),
    "top5_alpha158_ic":          ic_series.head(5).round(5).to_dict(),
    "model_ic_mean":             float(signal_ic_daily.mean()),
    "model_ic_std":              float(signal_ic_daily.std()),
    "model_icir":                float(icir),
    "model_ic_pct_positive":     float((signal_ic_daily > 0).mean()),
    "top5_feature_importance":   (fi.head(5).to_dict() if fi is not None else {}),
}

with open(f"{OUTPUT_DIR}/qlib_results.json", "w") as f:
    json.dump(qlib_summary, f, indent=2)

print(f"\nqlib results saved to {OUTPUT_DIR}/qlib_results.json")
print("\n=== Phase 2 complete ===")
