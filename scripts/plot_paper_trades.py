"""Plot a 4-panel summary of a finished paper-trading session's trades.jsonl.

Regenerates the figure used in docs/strategy_presentation.md
(paper_trades_8h.png) from data/paper_trading/lgbm_8h_20260730/trades.jsonl.

Panels:
  1. Holding-time histogram (seconds, derived from exit_ts - entry_ts)
  2. Trade entries per 15-minute bucket over the session
  3. pnl_proxy scatter over time, colored by sign
  4. Directional-accuracy bar chart by coin (from dir_hit)

NOTE ON SCHEMA: trades.jsonl written by experiments/paper_trade_lgbm.py uses
entry_ts/exit_ts/pnl_proxy/dir_hit — NOT the net_pnl_bps/gross_pnl_bps/
holding_sec/entry_time schema that experiments/analyze_paper.py expects (that
script targets the OU/zscore paper_trader.py sessions instead, e.g.
data/paper_trading/WIF_binance_cryptocom_ou). Running analyze_paper.py against
an lgbm_* session directory will KeyError. This script reads the actual
lgbm_8h_20260730 schema directly.

Usage:
    python scripts/plot_paper_trades.py data/paper_trading/lgbm_8h_20260730 --out paper_trades_8h.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_trades(session_dir: Path) -> pd.DataFrame:
    path = session_dir / "trades.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    df = pd.DataFrame(rows)
    df["entry_ts"] = pd.to_datetime(df["entry_ts"])
    df["exit_ts"] = pd.to_datetime(df["exit_ts"])
    df["holding_sec"] = (df["exit_ts"] - df["entry_ts"]).dt.total_seconds()
    return df


def plot(df: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Holding-time histogram
    ax = axes[0, 0]
    ax.hist(df["holding_sec"], bins=30, color="steelblue", edgecolor="black")
    ax.axvline(df["holding_sec"].median(), color="red", linestyle="--",
               label=f"median {df['holding_sec'].median():.1f}s")
    ax.axvline(df["holding_sec"].mean(), color="orange", linestyle="--",
               label=f"mean {df['holding_sec'].mean():.1f}s")
    ax.set_title("Holding time per trade")
    ax.set_xlabel("seconds")
    ax.set_ylabel("count")
    ax.legend()

    # 2. Entries per 15-min bucket
    ax = axes[0, 1]
    bucketed = df.set_index("entry_ts").resample("15min").size()
    ax.bar(bucketed.index, bucketed.values, width=0.01, color="slategray")
    ax.set_title("Trade entries per 15-min bucket")
    ax.set_xlabel("time")
    ax.set_ylabel("entries")
    ax.tick_params(axis="x", rotation=45)

    # 3. pnl_proxy scatter, colored by sign
    ax = axes[1, 0]
    colors = np.where(df["pnl_proxy"] >= 0, "seagreen", "firebrick")
    ax.scatter(df["entry_ts"], df["pnl_proxy"], c=colors, s=6, alpha=0.5)
    ax.axhline(df["pnl_proxy"].mean(), color="black", linestyle="--",
               label=f"mean {df['pnl_proxy'].mean():+.3f}")
    ax.set_title("pnl_proxy per trade")
    ax.set_xlabel("time")
    ax.set_ylabel("pnl_proxy (z-units)")
    ax.legend()
    ax.tick_params(axis="x", rotation=45)

    # 4. DirAcc by coin
    ax = axes[1, 1]
    dir_acc = df.groupby("coin")["dir_hit"].mean().sort_values()
    ax.barh(dir_acc.index, dir_acc.values, color="teal")
    ax.axvline(0.5, color="red", linestyle="--", label="coin flip")
    ax.set_title("Directional accuracy by coin")
    ax.set_xlabel("dir_hit rate")
    ax.legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"saved {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("session_dir", type=Path)
    ap.add_argument("--out", type=Path, default=Path("paper_trades_8h.png"))
    args = ap.parse_args()

    df = load_trades(args.session_dir)
    print(f"loaded {len(df)} trades from {args.session_dir}")
    plot(df, args.out)


if __name__ == "__main__":
    main()
