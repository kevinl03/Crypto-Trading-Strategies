"""Score a finished LGBM paper-trading session against a persistence baseline.

Reads the signals/trades written by `experiments/paper_trade_lgbm.py` and emits
regression + directional metrics, including the naive `z_t -> z_{t+HORIZON}`
baseline on identical rows so the model's true increment is visible.

Usage:
    python scripts/report_paper_session.py data/paper_trading/July31st_8_hr
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Default matches current cex_gbm_new / paper_trade_lgbm (t+1). Override via
# config.json "horizon" or --horizon.
DEFAULT_HORIZON = 1


def load_jsonl_rows(out: Path, stem: str) -> list[dict]:
    """Load all shards for a stem (signals / trades) in write order."""
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


def load_signals(out: Path) -> pd.DataFrame:
    return pd.DataFrame(load_jsonl_rows(out, "signals"))


def resolve_horizon(out: Path, cli_horizon: int | None) -> int:
    if cli_horizon is not None:
        return int(cli_horizon)
    cfg_path = out / "config.json"
    if cfg_path.exists():
        try:
            cfg = json.loads(cfg_path.read_text(encoding="utf-8-sig"))
            if cfg.get("horizon") is not None:
                return int(cfg["horizon"])
        except json.JSONDecodeError:
            pass
    return DEFAULT_HORIZON


def align_forward(sig: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Attach the realized z-score `horizon` snapshots ahead to each prediction."""
    z = sig.dropna(subset=["zscore"])[["coin", "pair", "snapshot_idx", "zscore"]].copy()
    z["snapshot_idx"] = z["snapshot_idx"].astype(int) - horizon
    z = z.rename(columns={"zscore": "z_fwd"})
    merged = sig.merge(z, on=["coin", "pair", "snapshot_idx"], how="inner")
    return merged.dropna(subset=["pred", "zscore", "z_fwd"])


def metrics(y_true: np.ndarray, y_pred: np.ndarray, label: str) -> dict:
    return {
        "set": label,
        "n": int(len(y_true)),
        "r2": float(r2_score(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(mean_squared_error(y_true, y_pred) ** 0.5),
        "dir_acc": float(np.mean(np.sign(y_pred) == np.sign(y_true))),
        "corr": float(np.corrcoef(y_pred, y_true)[0, 1]) if len(y_true) > 1 else float("nan"),
        "target_std": float(np.std(y_true)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("session_dir", type=Path)
    ap.add_argument("--entry-tau", type=float, default=0.5)
    ap.add_argument(
        "--horizon",
        type=int,
        default=None,
        help="Forward snapshots for realized target (default: config.json horizon, else 1)",
    )
    args = ap.parse_args()

    out = args.session_dir.resolve()
    horizon = resolve_horizon(out, args.horizon)
    sig = load_signals(out)
    df = align_forward(sig, horizon=horizon)

    rows = []
    for label, sub in [
        ("all_predictions", df),
        (f"entries_abs_pred>={args.entry_tau}", df[df["pred"].abs() >= args.entry_tau]),
    ]:
        if sub.empty:
            continue
        y = sub["z_fwd"].to_numpy(float)
        rows.append(metrics(y, sub["pred"].to_numpy(float), f"model | {label}"))
        rows.append(metrics(y, sub["zscore"].to_numpy(float), f"naive_persistence | {label}"))

    rep = pd.DataFrame(rows)
    rep.to_csv(out / "metrics_report.csv", index=False)

    summary = {}
    summary_path = out / "summary.json"
    if summary_path.exists():
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8-sig"))
        except json.JSONDecodeError:
            summary = {}
    print(json.dumps({"report_horizon": horizon, **summary}, indent=2))
    print()
    if rep.empty:
        print("no scorable rows yet (need signals + forward z at horizon)")
    else:
        print(rep.to_string(index=False))
    print(f"\nsaved {out / 'metrics_report.csv'} (horizon={horizon})")


if __name__ == "__main__":
    main()
