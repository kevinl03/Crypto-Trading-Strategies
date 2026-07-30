"""Score a finished LGBM paper-trading session against a persistence baseline.

Reads the signals/trades written by `experiments/paper_trade_lgbm.py` and emits
regression + directional metrics, including the naive `z_t -> z_{t+HORIZON}`
baseline on identical rows so the model's true increment is visible.

Usage:
    python scripts/report_paper_session.py data/paper_trading/lgbm_8h_20260730
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

HORIZON = 2


def load_signals(out: Path) -> pd.DataFrame:
    rows = [json.loads(l) for l in (out / "signals.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    return pd.DataFrame(rows)


def align_forward(sig: pd.DataFrame, horizon: int = HORIZON) -> pd.DataFrame:
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
        "corr": float(np.corrcoef(y_pred, y_true)[0, 1]),
        "target_std": float(np.std(y_true)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("session_dir", type=Path)
    ap.add_argument("--entry-tau", type=float, default=0.5)
    args = ap.parse_args()

    out = args.session_dir.resolve()
    sig = load_signals(out)
    df = align_forward(sig)

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

    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    print(json.dumps(summary, indent=2))
    print()
    print(rep.to_string(index=False))
    print(f"\nsaved {out / 'metrics_report.csv'}")


if __name__ == "__main__":
    main()
