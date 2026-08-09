#!/usr/bin/env python3
"""Post-hoc |pred|>=tau sweep on a finished live LGBM paper session.

Uses signals for forecast metrics (DirAcc/R2/mean pnl_proxy) and closed trades
for book metrics when filtering by entry |pred|. Capacity-aware H=1 sim optional
via --sim-fill (rank by |pred|, max_open).

Does not retrain; does not touch a live trader.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from report_paper_session import align_forward, load_jsonl_rows, load_signals, resolve_horizon  # noqa: E402


def sharpe(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return float("nan")
    s = float(np.std(x, ddof=1))
    return float(np.mean(x) / s) if s > 0 else float("nan")


def diracc(y: np.ndarray, p: np.ndarray) -> float:
    m = (y != 0) & (p != 0) & np.isfinite(y) & np.isfinite(p)
    if not np.any(m):
        return float("nan")
    return float(np.mean(np.sign(p[m]) == np.sign(y[m])))


def mean_pnl(y: np.ndarray, p: np.ndarray) -> float:
    m = np.isfinite(y) & np.isfinite(p) & (p != 0)
    if not np.any(m):
        return float("nan")
    return float(np.mean(np.sign(p[m]) * y[m]))


def hourly_closed_sharpe(trades: pd.DataFrame) -> float:
    if trades.empty or "exit_ts" not in trades.columns:
        return float("nan")
    t = trades.copy()
    t["exit_ts"] = pd.to_datetime(t["exit_ts"], utc=True, errors="coerce")
    t = t.dropna(subset=["exit_ts", "pnl_proxy"])
    if t.empty:
        return float("nan")
    hour = t.groupby(t["exit_ts"].dt.floor("h"))["pnl_proxy"].sum()
    return sharpe(hour.to_numpy())


def score_aligned(df: pd.DataFrame, tau: float) -> dict:
    sub = df[df["pred"].abs() >= tau]
    y = sub["z_fwd"].to_numpy(float)
    p = sub["pred"].to_numpy(float)
    z = sub["zscore"].to_numpy(float)
    n = int(len(sub))
    return {
        "tau": float(tau),
        "n_signal_entries": n,
        "fire_rate": float(n / max(1, len(df))),
        "diracc": diracc(y, p),
        "r2": float(r2_score(y, p)) if n > 1 else float("nan"),
        "mean_pnl_proxy": mean_pnl(y, p),
        "total_pnl_proxy": float(mean_pnl(y, p) * n) if n else float("nan"),
        "sharpe_per_trade": sharpe(np.sign(p) * y) if n else float("nan"),
        "naive_diracc": diracc(y, z),
        "naive_r2": float(r2_score(y, z)) if n > 1 else float("nan"),
        "naive_mean_pnl_proxy": mean_pnl(y, z),
    }


def score_trades(trades: pd.DataFrame, tau: float) -> dict:
    if trades.empty:
        return {
            "tau": float(tau),
            "n_closed": 0,
            "diracc_closed": float("nan"),
            "mean_pnl_closed": float("nan"),
            "total_pnl_closed": float("nan"),
            "sharpe_per_trade_closed": float("nan"),
            "sharpe_A_closed_hourly": float("nan"),
            "note": "no trades.jsonl",
        }
    # Live trades already entered at session tau (usually 0.5); post-hoc tighter filter.
    pred_col = "pred" if "pred" in trades.columns else "entry_pred"
    if pred_col not in trades.columns:
        return {
            "tau": float(tau),
            "n_closed": 0,
            "diracc_closed": float("nan"),
            "mean_pnl_closed": float("nan"),
            "total_pnl_closed": float("nan"),
            "sharpe_per_trade_closed": float("nan"),
            "sharpe_A_closed_hourly": float("nan"),
            "note": f"missing pred column in trades",
        }
    sub = trades[trades[pred_col].abs() >= tau].copy()
    n = int(len(sub))
    if n == 0:
        return {
            "tau": float(tau),
            "n_closed": 0,
            "diracc_closed": 0,
            "mean_pnl_closed": float("nan"),
            "total_pnl_closed": 0.0,
            "sharpe_per_trade_closed": float("nan"),
            "sharpe_A_closed_hourly": float("nan"),
            "note": "posthoc filter on live closes (book was filled at lower tau)",
        }
    hit = sub["dir_hit"] if "dir_hit" in sub.columns else (
        np.sign(sub[pred_col]) == np.sign(sub.get("exit_z", sub.get("z_exit")))
    )
    if "dir_hit" not in sub.columns:
        exit_z = sub["exit_z"] if "exit_z" in sub.columns else sub["z_exit"]
        hit = (np.sign(sub[pred_col]) == np.sign(exit_z)).astype(float)
        pnl = np.sign(sub[pred_col].to_numpy(float)) * exit_z.to_numpy(float)
    else:
        pnl = sub["pnl_proxy"].to_numpy(float)
        hit = sub["dir_hit"].to_numpy(float)
    return {
        "tau": float(tau),
        "n_closed": n,
        "diracc_closed": float(np.mean(hit)),
        "mean_pnl_closed": float(np.mean(pnl)),
        "total_pnl_closed": float(np.sum(pnl)),
        "sharpe_per_trade_closed": sharpe(pnl),
        "sharpe_A_closed_hourly": hourly_closed_sharpe(sub),
        "note": "posthoc filter on live closes (book was filled at lower tau)",
    }


def sim_h1_fill(df: pd.DataFrame, tau: float, max_open: int) -> dict:
    """Capacity-aware H=1 replay: each snap fill top-|pred| among eligible."""
    need = {"coin", "pair", "snapshot_idx", "pred", "z_fwd"}
    if not need.issubset(df.columns):
        return {"tau": float(tau), "n_closed": 0, "note": "aligned signals missing cols"}

    panel = df.dropna(subset=["pred", "z_fwd"]).copy()
    panel["abs_pred"] = panel["pred"].abs()
    closed = []
    for snap, g in panel.groupby("snapshot_idx", sort=True):
        elig = g[g["abs_pred"] >= tau].sort_values("abs_pred", ascending=False)
        # one per (coin,pair) already unique in g typically
        take = elig.head(max_open)
        for _, row in take.iterrows():
            p = float(row["pred"])
            y = float(row["z_fwd"])
            closed.append(
                {
                    "snapshot_idx": int(snap),
                    "pred": p,
                    "pnl_proxy": float(np.sign(p) * y),
                    "dir_hit": float(np.sign(p) == np.sign(y)) if y != 0 and p != 0 else float("nan"),
                    "exit_ts": row.get("ts") or row.get("timestamp"),
                }
            )
    if not closed:
        return {
            "tau": float(tau),
            "n_closed": 0,
            "diracc": float("nan"),
            "mean_pnl_proxy": float("nan"),
            "total_pnl_proxy": 0.0,
            "sharpe_per_trade": float("nan"),
            "sharpe_A": float("nan"),
            "note": "sim H=1 |pred| rank fill",
        }
    tdf = pd.DataFrame(closed)
    # hour from snapshot if no ts
    if tdf["exit_ts"].isna().all() or tdf["exit_ts"].dtype == object and tdf["exit_ts"].isna().mean() > 0.5:
        # approximate: group by snapshot buckets of ~40 snaps (~1h at ~90s) if no clock
        sharpe_a = float("nan")
    else:
        tdf["exit_ts"] = pd.to_datetime(tdf["exit_ts"], utc=True, errors="coerce")
        sharpe_a = hourly_closed_sharpe(tdf.rename(columns={"pnl_proxy": "pnl_proxy"}))
    pnl = tdf["pnl_proxy"].to_numpy(float)
    return {
        "tau": float(tau),
        "n_closed": int(len(tdf)),
        "diracc": float(np.nanmean(tdf["dir_hit"].to_numpy(float))),
        "mean_pnl_proxy": float(np.mean(pnl)),
        "total_pnl_proxy": float(np.sum(pnl)),
        "sharpe_per_trade": sharpe(pnl),
        "sharpe_A": sharpe_a,
        "note": f"sim H=1 |pred| rank fill max_open={max_open}",
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("session_dir", type=Path)
    ap.add_argument("--taus", type=float, nargs="+", default=[0.5, 0.75, 1.0])
    ap.add_argument("--max-open", type=int, default=50)
    ap.add_argument("--sim-fill", action="store_true")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    out = args.session_dir.resolve()
    horizon = resolve_horizon(out, None)
    print(f"session={out} horizon={horizon}", flush=True)

    sig = load_signals(out)
    print(f"signals rows={len(sig):,}", flush=True)
    aligned = align_forward(sig, horizon=horizon)
    print(f"aligned rows={len(aligned):,}", flush=True)

    trades_rows = load_jsonl_rows(out, "trades")
    trades = pd.DataFrame(trades_rows) if trades_rows else pd.DataFrame()
    print(f"closed trades={len(trades):,}", flush=True)

    signal_rows = [score_aligned(aligned, t) for t in args.taus]
    trade_rows = [score_trades(trades, t) for t in args.taus]
    sim_rows = [sim_h1_fill(aligned, t, args.max_open) for t in args.taus] if args.sim_fill else []

    report = {
        "session": str(out),
        "horizon": horizon,
        "n_signals": int(len(sig)),
        "n_aligned": int(len(aligned)),
        "n_closed_live": int(len(trades)),
        "signal_filter": signal_rows,
        "live_trades_posthoc": trade_rows,
        "sim_h1_fill": sim_rows,
    }
    dest = args.out or (out / "tau_sweep_live.json")
    dest.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n=== Signal filter (|pred|>=tau -> z_fwd) ===")
    print(pd.DataFrame(signal_rows).to_string(index=False))
    print("\n=== Live closes posthoc (|entry pred|>=tau) ===")
    print(pd.DataFrame(trade_rows).to_string(index=False))
    if sim_rows:
        print("\n=== Sim H=1 capacity fill ===")
        print(pd.DataFrame(sim_rows).to_string(index=False))
    print(f"\nwrote {dest}", flush=True)


if __name__ == "__main__":
    main()
