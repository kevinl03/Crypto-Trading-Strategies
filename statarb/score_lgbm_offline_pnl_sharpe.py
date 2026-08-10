#!/usr/bin/env python3
"""Score production LGBM on Jul25-28 holdout with LSTM-comparable pnl/Sharpe metrics."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))

import lstm_zscore_lib as L
from experiments.paper_trade_lgbm import (  # noqa: E402
    N_LAGS,
    add_cross_exchange_features,
    build_spread_features,
    prepare_X,
)


def _to_wide_fast(df: pd.DataFrame, index_cols, col_col, value_cols, prefix) -> pd.DataFrame:
    """Faster lag+wide than paper_trade's lambda transform."""
    if df.empty:
        return pd.DataFrame()
    df = df.sort_values(index_cols + [col_col]).copy()
    grp = df.groupby(["window_id", "coin", col_col], sort=False)
    lag_cols = []
    for col in value_cols:
        for lag in range(1, N_LAGS + 1):
            name = f"{col}_lag{lag}"
            df[name] = grp[col].shift(lag)
            lag_cols.append(name)
    df[lag_cols] = df[lag_cols].astype("float32")
    keep = index_cols + [col_col] + lag_cols
    wide = df[keep].groupby(index_cols + [col_col], sort=False)[lag_cols].mean().unstack(col_col)
    wide.columns = [f"{prefix}{col}_{exch}" for col, exch in wide.columns]
    wide.columns.name = None
    return wide.reset_index()


def build_batch_matrix(raw: dict[str, pd.DataFrame]) -> pd.DataFrame:
    t0 = time.time()
    base = build_spread_features(raw["spread_matrix"])
    print(f"  spread features {base.shape} in {time.time()-t0:.1f}s")
    if base.empty:
        return base
    base["target"] = base.groupby(["window_id", "coin", "pair"], sort=False)["zscore"].shift(-1)
    merge_keys = ["window_id", "snapshot_idx", "coin"]
    df = base
    for label, data, cols, prefix in [
        ("ticker", raw["ticker"], ["mid", "spread_bps", "bid_volume", "ask_volume"], "tk_"),
        ("orderbook", raw["orderbook"], ["imbalance"], "ob_"),
        ("trades", raw["trades"], ["buy_sell_ratio", "total_volume"], "tr_"),
    ]:
        if data is None or data.empty or "window_id" not in data.columns:
            continue
        t1 = time.time()
        aux = _to_wide_fast(data, merge_keys, "exchange", cols, prefix)
        print(f"  {label} wide {aux.shape} in {time.time()-t1:.1f}s")
        if not aux.empty:
            df = df.merge(aux, on=merge_keys, how="left")
    df = add_cross_exchange_features(df)
    print(f"  matrix {df.shape} total {time.time()-t0:.1f}s")
    return df


def main() -> None:
    model_path = HERE / "outputs" / "statarb_lgbm.txt"
    out_dir = HERE / "outputs_lgbm_offline_jul25"
    lstm_metrics_path = HERE / "outputs_lstm_size_matched" / "metrics.json"
    out_dir.mkdir(parents=True, exist_ok=True)

    local_root = L.resolve_local_data_root()
    hf_token = L.resolve_hf_token()
    test_windows = [w for w in L.WINDOWS if w["role"] == "test"]
    print("Loading Jul25-28 test pool …")
    t0 = time.time()
    test_raw = L.pool_windows(
        test_windows, local_root=local_root, hf_token=hf_token, use_hf=True
    )
    print(f"load done in {time.time()-t0:.1f}s")

    print("Building LGBM feature matrix …")
    df = build_batch_matrix(test_raw)
    del test_raw
    df = df.dropna(subset=["target", "zscore"]).reset_index(drop=True)
    print("rows with target:", len(df))

    model = lgb.Booster(model_file=str(model_path))
    feature_names = model.feature_name()
    cat_maps = list(model.pandas_categorical or [])
    X = prepare_X(df, feature_names, cat_maps)
    print("predicting …")
    preds = model.predict(X)
    y = df["target"].to_numpy(dtype=np.float64)
    z_now = df["zscore"].to_numpy(dtype=np.float64)
    meta = df[["window_id", "snapshot_idx", "coin", "pair"]].copy()

    metrics = L.evaluate_model_and_naive(y, preds, z_now, tau=0.5, meta=meta)
    out = {
        "model": "lgbm_statarb_outputs",
        "model_path": str(model_path),
        "split": "jul25_28 holdout (snapshot_idx>=3584)",
        "n_lags": N_LAGS,
        "lgbm": metrics["lstm"],
        "naive_zt": metrics["naive_zt"],
        "notes": metrics["notes"],
    }
    (out_dir / "metrics.json").write_text(json.dumps(out, indent=2), encoding="utf-8")

    h = out["lgbm"]["headline_filtered"]
    ha = out["lgbm"]["all"]
    hn = out["naive_zt"]["headline_filtered"]

    lstm_block = None
    if lstm_metrics_path.exists():
        lstm_block = json.loads(lstm_metrics_path.read_text(encoding="utf-8"))["lstm"]

    def row(name, block):
        return (
            f"| {name} | {block['n']} | {block['diracc']:.4f} | {block['r2']:.4f} | "
            f"{block['mean_pnl_proxy']:.4f} | {block['sharpe_per_trade']:.4f} | "
            f"{block['sharpe_closed_hourly_A']:.4f} |"
        )

    lines = [
        "# Jul 25-28 holdout: LGBM vs size-matched LSTM",
        "",
        "Same window, target \(z_{t+1}\), filter `|pred|>0.5`, metric code (`evaluate_model_and_naive`).",
        "",
        "| Model | n | DirAcc | R2 | mean pnl_proxy | Sharpe/trade | Sharpe A |",
        "|---|---:|---:|---:|---:|---:|---:|",
        row("**LGBM filtered**", h),
        row("LGBM all", ha),
        row("Naive `|z_t|>0.5`", hn),
    ]
    if lstm_block is not None:
        lines.append(row("**LSTM size-matched filtered**", lstm_block["filtered"]))
        lines.append(row("LSTM size-matched all", lstm_block["all"]))
        lines.append("")
        lines.append(
            "Note: LSTM scored a stride/subsampled test panel (n=150k all); "
            "LGBM scored the full Jul25-28 rows with valid target. "
            "Both use the same calendar holdout and definitions."
        )

    md = "\n".join(lines) + "\n"
    (out_dir / "METRICS.md").write_text(md, encoding="utf-8")
    compare = {
        "window": "jul25_28",
        "lgbm_filtered": h,
        "lgbm_all": ha,
        "lgbm_naive_zt_filtered": hn,
        "lstm_size_matched_filtered": lstm_block["filtered"] if lstm_block else None,
        "lstm_size_matched_all": lstm_block["all"] if lstm_block else None,
    }
    (out_dir / "compare_jul25_lgbm_vs_lstm.json").write_text(
        json.dumps(compare, indent=2), encoding="utf-8"
    )
    print(md)
    print("wrote", out_dir)


if __name__ == "__main__":
    main()
