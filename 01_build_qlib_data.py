"""
Phase 1: Download real crypto OHLCV from Yahoo Finance and write it
into qlib's native binary format.

Expanded to 15 assets for a meaningful cross-section (Alpha158 needs ≥10).
"""

import os, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import yfinance as yf
from pathlib import Path

QLIB_DATA_DIR = "/home/ubuntu/ky/agents/qlib_paper/qlib_data"
OUTPUT_DIR    = "/home/ubuntu/ky/agents/qlib_paper"

SYMBOLS_MAP = {
    "BTC-USD":   "BTCUSD",
    "ETH-USD":   "ETHUSD",
    "SOL-USD":   "SOLUSD",
    "BNB-USD":   "BNBUSD",
    "AVAX-USD":  "AVAXUSD",
    "ADA-USD":   "ADAUSD",
    "LINK-USD":  "LINKUSD",
    "DOT-USD":   "DOTUSD",
    "MATIC-USD": "MATICUSD",
    "UNI-USD":   "UNIUSD",
    "ATOM-USD":  "ATOMUSD",
    "LTC-USD":   "LTCUSD",
    "XRP-USD":   "XRPUSD",
    "DOGE-USD":  "DOGEUSD",
    "NEAR-USD":  "NEARUSD",
}
START = "2021-01-01"
END   = "2025-04-30"

print(f"Downloading {len(SYMBOLS_MAP)} crypto assets from Yahoo Finance …")
raw = yf.download(list(SYMBOLS_MAP.keys()), start=START, end=END,
                  auto_adjust=True, progress=False)
print(f"  Rows: {len(raw)}")

all_dates = sorted(raw.index.strftime("%Y-%m-%d").tolist())
cal_array = np.array(all_dates)
print(f"  Calendar: {len(all_dates)} days  ({all_dates[0]} → {all_dates[-1]})")

# ─── CALENDAR ────────────────────────────────────────────────────────────────
import shutil
if Path(QLIB_DATA_DIR).exists():
    shutil.rmtree(QLIB_DATA_DIR)
    print("  Cleared old data store")

cal_dir = Path(QLIB_DATA_DIR) / "calendars"
cal_dir.mkdir(parents=True)
with open(cal_dir / "day.txt", "w") as f:
    f.write("\n".join(all_dates) + "\n")

# ─── INSTRUMENTS ─────────────────────────────────────────────────────────────
inst_dir = Path(QLIB_DATA_DIR) / "instruments"
inst_dir.mkdir()
with open(inst_dir / "all.txt", "w") as f:
    for ticker, symbol in SYMBOLS_MAP.items():
        close_series = raw["Close"][ticker].dropna()
        if len(close_series) == 0:
            print(f"  SKIP {symbol}: no data")
            continue
        first = close_series.index[0].strftime("%Y-%m-%d")
        last  = close_series.index[-1].strftime("%Y-%m-%d")
        f.write(f"{symbol}\t{first}\t{last}\n")

# ─── FEATURE BINARY FILES ────────────────────────────────────────────────────
written = []
for ticker, symbol in SYMBOLS_MAP.items():
    close  = raw["Close"][ticker].dropna()
    if len(close) < 100:
        print(f"  SKIP {symbol}: only {len(close)} rows")
        continue

    opens  = raw["Open"][ticker]
    high   = raw["High"][ticker]
    low    = raw["Low"][ticker]
    volume = raw["Volume"][ticker]

    df = pd.DataFrame({
        "open":   opens,
        "close":  close,
        "high":   high,
        "low":    low,
        "volume": volume,
        "factor": 1.0,
        "change": close.pct_change(),
    })
    df.index = pd.DatetimeIndex(df.index).strftime("%Y-%m-%d")

    first_date = df.dropna(subset=["close"]).index[0]
    start_idx  = int(np.where(cal_array == first_date)[0][0])
    cal_from_start = cal_array[start_idx:]
    df_aligned = df.reindex(cal_from_start)

    feat_dir = Path(QLIB_DATA_DIR) / "features" / symbol.lower()
    feat_dir.mkdir(parents=True)

    for field in ["open", "close", "high", "low", "volume", "factor", "change"]:
        values   = df_aligned[field].values.astype("float32")
        bin_data = np.concatenate([[np.float32(start_idx)], values])
        with open(feat_dir / f"{field}.day.bin", "wb") as fp:
            bin_data.astype("<f").tofile(fp)

    written.append(symbol)
    print(f"  {symbol}: {len(df_aligned)} days, start_idx={start_idx}")

print(f"\nWritten {len(written)} instruments: {written}")
print(f"qlib data store ready: {QLIB_DATA_DIR}")
