"""Portfolio Sharpe for LGBM paper sessions, lit-aligned variants.

Literature baselines (issue #62 / Tadi & Kortchemski 2021, HF pairs papers):
  - Rf = 0
  - Sharpe on a *time-aggregated P&L / return series* (not per-trade), including
    unrealized MTM on open positions (Tadi distinguish realized vs unrealized)
  - Period return = period P&L / capital; capital often fixed book size
  - Annualize: S_ann = S_period * sqrt(periods_per_year)
    crypto 24/7 -> 24*365 hours; equity HF often uses 252*6.5 for hourly

Our settlement proxy marks a live bet as V = direction * z_t (same as terminal
pnl_proxy = direction * z_exit). Hourly (or per-snapshot) portfolio P&L is the
change in equity = realized_cum + sum(open marks).

Usage:
  python scripts/portfolio_sharpe_paper_session.py data/paper_trading/July31st_8_hr
  python scripts/portfolio_sharpe_paper_session.py data/paper_trading/lgbm_8h_20260730 --max-open 50
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def load_jsonl_rows(out: Path, stem: str) -> list[dict]:
    paths: list[Path] = []
    primary = out / f"{stem}.jsonl"
    if primary.exists():
        paths.append(primary)
    paths.extend(sorted(out.glob(f"{stem}_*.jsonl")))
    rows: list[dict] = []
    for path in paths:
        for line in path.read_text(encoding="utf-8-sig").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    return rows


def sharpe(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return float("nan")
    s = float(np.std(x, ddof=1))
    return float(np.mean(x) / s) if s > 0 else float("nan")


def build_z_panel(signals: pd.DataFrame) -> pd.DataFrame:
    """Latest z per (coin, pair, snapshot) with timestamp."""
    z = signals.dropna(subset=["zscore"]).copy()
    z["ts"] = pd.to_datetime(z["ts"], utc=True)
    z["snapshot_idx"] = z["snapshot_idx"].astype(int)
    z = z.sort_values(["coin", "pair", "snapshot_idx", "ts"])
    z = z.groupby(["coin", "pair", "snapshot_idx"], as_index=False).agg(
        ts=("ts", "last"), zscore=("zscore", "last"), spread_bps=("spread_bps", "last")
    )
    return z


def equity_at(
    t: pd.Timestamp,
    trades: pd.DataFrame,
    z_by_key: dict[tuple[str, str, int], float],
    snap_ts: pd.Series,
) -> tuple[float, float, float, int]:
    """Return (equity, realized_cum, open_mtm, n_open) at time t (inclusive)."""
    closed = trades[trades["exit_ts"] <= t]
    realized = float(closed["pnl_proxy"].sum()) if len(closed) else 0.0

    # Still open: entered and not yet exited
    open_mask = (trades["entry_ts"] <= t) & (trades["exit_ts"] > t)
    opens = trades.loc[open_mask]
    open_mtm = 0.0
    n_open = 0
    if len(opens):
        # mark at latest snapshot with snap_ts <= t
        eligible = snap_ts[snap_ts <= t]
        if len(eligible):
            mark_snap = int(eligible.index.max())  # index is snapshot_idx
            for row in opens.itertuples():
                z = z_by_key.get((row.coin, row.pair, mark_snap))
                if z is None:
                    # walk backward a few snaps if this pair missing at mark_snap
                    for s in range(mark_snap, max(mark_snap - 5, -1), -1):
                        z = z_by_key.get((row.coin, row.pair, s))
                        if z is not None:
                            break
                if z is not None and z == z:
                    open_mtm += float(row.direction) * float(z)
                    n_open += 1
    equity = realized + open_mtm
    return equity, realized, open_mtm, n_open


def period_pnl_series(
    trades: pd.DataFrame,
    z_panel: pd.DataFrame,
    freq: str,
) -> pd.DataFrame:
    z_by_key = {
        (r.coin, r.pair, int(r.snapshot_idx)): float(r.zscore)
        for r in z_panel.itertuples()
    }
    snap_ts = (
        z_panel.groupby("snapshot_idx")["ts"].max().sort_index()
    )  # snapshot_idx -> ts

    t0 = trades["entry_ts"].min().floor(freq)
    t1 = max(trades["exit_ts"].max(), z_panel["ts"].max()).ceil(freq)
    stamps = pd.date_range(t0, t1, freq=freq, tz="UTC")
    # evaluate at end of each bar (right edge): use stamp as bar end
    rows = []
    prev_eq = None
    for t in stamps:
        eq, real, mtm, n_open = equity_at(t, trades, z_by_key, snap_ts)
        pnl = float("nan") if prev_eq is None else eq - prev_eq
        rows.append(
            {
                "t": t,
                "equity": eq,
                "realized_cum": real,
                "open_mtm": mtm,
                "n_open": n_open,
                "pnl": pnl,
            }
        )
        prev_eq = eq
    return pd.DataFrame(rows)


def closed_only_hourly(trades: pd.DataFrame) -> pd.Series:
    return trades.set_index("exit_ts").resample("1h")["pnl_proxy"].sum().dropna()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("session_dir", type=Path)
    ap.add_argument("--max-open", type=int, default=50, help="Book capital in slot-units")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    out = args.session_dir.resolve()
    max_open = args.max_open

    trades = pd.DataFrame(load_jsonl_rows(out, "trades"))
    if trades.empty:
        raise SystemExit(f"no trades in {out}")
    trades["entry_ts"] = pd.to_datetime(trades["entry_ts"], utc=True)
    trades["exit_ts"] = pd.to_datetime(trades["exit_ts"], utc=True)
    trades["pnl_proxy"] = trades["pnl_proxy"].astype(float)
    trades["direction"] = trades["direction"].astype(int)

    sig_rows = load_jsonl_rows(out, "signals")
    if not sig_rows:
        raise SystemExit(f"no signals in {out} (needed for open MTM)")
    z_panel = build_z_panel(pd.DataFrame(sig_rows))

    # --- Variant A: closed-only hourly (prior headline) ---
    a = closed_only_hourly(trades)
    sharpe_a = sharpe(a.to_numpy())

    # --- Variant B/C: hourly equity with open MTM ---
    hourly = period_pnl_series(trades, z_panel, "1h")
    h_pnl = hourly["pnl"].dropna().to_numpy()
    # drop leading nan already; also drop bars with no activity (pnl==0 & n_open==0 & no closes)
    # keep all finite bars after first
    sharpe_b_pnl = sharpe(h_pnl)  # Tadi-style on P&L units (not % returns)
    r_cap = h_pnl / max_open
    sharpe_c_ret = sharpe(r_cap)  # capital = max_open slots

    # avg open as capital (time-varying risk); if flat, fall back to max_open
    n_open_bar = hourly.loc[hourly["pnl"].notna(), "n_open"].to_numpy(dtype=float)
    cap = np.where(n_open_bar > 0, n_open_bar, float(max_open))
    r_avg = h_pnl / cap
    sharpe_c2 = sharpe(r_avg)

    # --- Variant D: snapshot-bar equity (finer, still portfolio) ---
    # Use unique signal timestamps as bar ends (per snapshot)
    snap_ends = (
        z_panel.groupby("snapshot_idx", as_index=False)
        .agg(t=("ts", "max"))
        .sort_values("t")
    )
    prev_eq = None
    snap_pnls = []
    snap_nopen = []
    for row in snap_ends.itertuples():
        eq, _, _, n_open = equity_at(
            row.t,
            trades,
            {(r.coin, r.pair, int(r.snapshot_idx)): float(r.zscore) for r in z_panel.itertuples()},
            z_panel.groupby("snapshot_idx")["ts"].max().sort_index(),
        )
        if prev_eq is not None:
            snap_pnls.append(eq - prev_eq)
            snap_nopen.append(n_open)
        prev_eq = eq
    snap_pnls_a = np.asarray(snap_pnls, float)
    sharpe_d_pnl = sharpe(snap_pnls_a)
    sharpe_d_ret = sharpe(snap_pnls_a / max_open)

    # Annualization constants
    ann_hour_crypto = np.sqrt(24 * 365)
    ann_hour_equity_hf = np.sqrt(252 * 6.5)  # Fallahpour-style K=1638
    # snapshot ~110s -> periods/year
    if len(snap_ends) >= 2:
        dt_sec = (snap_ends["t"].iloc[-1] - snap_ends["t"].iloc[0]).total_seconds() / max(
            len(snap_ends) - 1, 1
        )
        snaps_per_year = (365 * 24 * 3600) / dt_sec if dt_sec > 0 else float("nan")
    else:
        snaps_per_year = float("nan")
    ann_snap = np.sqrt(snaps_per_year) if snaps_per_year == snaps_per_year else float("nan")

    # Final open inventory
    t_end = trades["exit_ts"].max()
    _, _, final_mtm, final_n_open = equity_at(
        t_end,
        trades,
        {(r.coin, r.pair, int(r.snapshot_idx)): float(r.zscore) for r in z_panel.itertuples()},
        z_panel.groupby("snapshot_idx")["ts"].max().sort_index(),
    )
    # positions still open after last close: entry ok, exit after t_end ΓÇö summary had n_open
    still_open = trades[trades["exit_ts"] > t_end]
    # Actually closed file only has closed; open bets not in trades.jsonl!
    # Open inventory at end is NOT in trades.jsonl ΓÇö only closed settles.
    # So MTM during session uses in-progress trades (entry < t < exit).
    # Terminal open book from paper trader is invisible here unless we have open snapshot.
    # Within-session straddles ARE captured.

    report = {
        "session": str(out),
        "n_closed": int(len(trades)),
        "max_open_capital": max_open,
        "note": (
            "Open MTM uses direction*z_t for bets with entry_ts<=t<exit_ts. "
            "Bets still open at session kill never appear in trades.jsonl, so "
            "terminal orphan inventory is not marked (same limitation as closed-only)."
        ),
        "variants": {
            "A_closed_only_hourly_pnl": {
                "sharpe": sharpe_a,
                "n_periods": int(len(a)),
                "mean_pnl": float(a.mean()),
                "std_pnl": float(a.std(ddof=1)),
                "ann_crypto_24x365": sharpe_a * ann_hour_crypto,
                "ann_equity_hf_1638": sharpe_a * ann_hour_equity_hf,
            },
            "B_hourly_equity_pnl_with_open_mtm": {
                "sharpe": sharpe_b_pnl,
                "n_periods": int(np.isfinite(h_pnl).sum()),
                "mean_pnl": float(np.nanmean(h_pnl)),
                "std_pnl": float(np.nanstd(h_pnl, ddof=1)),
                "ann_crypto_24x365": sharpe_b_pnl * ann_hour_crypto,
                "ann_equity_hf_1638": sharpe_b_pnl * ann_hour_equity_hf,
                "lit": "Tadi-style Rf=0 on period P&L; includes unrealized MTM delta",
            },
            "C_hourly_return_capital_max_open": {
                "sharpe": sharpe_c_ret,
                "n_periods": int(np.isfinite(r_cap).sum()),
                "mean_return": float(np.nanmean(r_cap)),
                "std_return": float(np.nanstd(r_cap, ddof=1)),
                "ann_crypto_24x365": sharpe_c_ret * ann_hour_crypto,
                "ann_equity_hf_1638": sharpe_c_ret * ann_hour_equity_hf,
                "lit": "Period P&L / max_open (fixed book); closest to capital-based Sharpe",
            },
            "C2_hourly_return_capital_n_open": {
                "sharpe": sharpe_c2,
                "ann_crypto_24x365": sharpe_c2 * ann_hour_crypto,
            },
            "D_snapshot_equity_pnl_with_open_mtm": {
                "sharpe": sharpe_d_pnl,
                "n_periods": int(len(snap_pnls_a)),
                "mean_dt_sec": float(dt_sec) if len(snap_ends) >= 2 else None,
                "ann_sqrt_snaps_per_year": sharpe_d_pnl * ann_snap if ann_snap == ann_snap else None,
            },
            "D2_snapshot_return_capital_max_open": {
                "sharpe": sharpe_d_ret,
                "ann_sqrt_snaps_per_year": sharpe_d_ret * ann_snap if ann_snap == ann_snap else None,
            },
        },
        "hourly_path": hourly.dropna(subset=["pnl"]).to_dict(orient="records"),
    }

    # JSON-safe timestamps
    for row in report["hourly_path"]:
        row["t"] = pd.Timestamp(row["t"]).isoformat()

    text = json.dumps(report, indent=2)
    print(text)
    dest = args.out or (out / "portfolio_sharpe_report.json")
    dest.write_text(text, encoding="utf-8")
    print(f"\nsaved {dest}", flush=True)

    print("\n=== HEADLINES ===", flush=True)
    print(f"A  closed-only hourly P&L Sharpe:     {sharpe_a:.4f}", flush=True)
    print(f"B  hourly equity+MTM P&L Sharpe:      {sharpe_b_pnl:.4f}", flush=True)
    print(f"C  hourly return / max_open Sharpe:   {sharpe_c_ret:.4f}  (ann~{sharpe_c_ret*ann_hour_crypto:.1f})", flush=True)
    print(f"D  snapshot equity+MTM P&L Sharpe:    {sharpe_d_pnl:.4f}", flush=True)
    print(f"D2 snapshot return / max_open Sharpe: {sharpe_d_ret:.4f}", flush=True)


if __name__ == "__main__":
    main()
