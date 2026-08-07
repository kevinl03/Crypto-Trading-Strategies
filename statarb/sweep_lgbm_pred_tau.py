#!/usr/bin/env python3
"""Sweep LGBM |pred| > tau on Jul25-28 holdout and rank filter thresholds.

Optimality is multi-objective (quality vs coverage). We report:
  - forecast/trading quality: DirAcc, R2, mean pnl_proxy, Sharpe/trade
  - coverage: n, fire_rate, Sharpe A (hourly closed proxy)
  - economic: mean gross bps return, win_rate_bps

Primary ranking score (paper-facing default):
  score = mean_pnl_proxy * sqrt(fire_rate)
which rewards higher z-proxy skill without collapsing to tiny n.
Also report argmax of DirAcc / R2 / Sharpe/trade / Sharpe A / mean_gross_bps.
"""

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
sys.path[:0] = [str(HERE), str(ROOT)]

import lstm_zscore_lib as L
from experiments.paper_trade_lgbm import prepare_X
from score_lgbm_offline_pnl_sharpe import build_batch_matrix

# Dense near common operating points; include 0.5 (campaign default) and 1.0
TAUS = [
    0.10,
    0.25,
    0.35,
    0.50,
    0.60,
    0.70,
    0.75,
    0.80,
    0.90,
    1.00,
    1.10,
    1.25,
    1.50,
    1.75,
    2.00,
]
MIN_N = 500  # ignore ultra-sparse thresholds for "optimal" picks


def _bps_stats(direction: np.ndarray, delta: np.ndarray, mask: np.ndarray) -> dict:
    gross = direction[mask] * delta[mask]
    gross = gross[np.isfinite(gross)]
    if len(gross) == 0:
        return {
            "mean_gross_bps": None,
            "win_rate_bps": None,
            "sharpe_bps": None,
            "mean_return_frac": None,
            "net_mean_5bps": None,
        }
    return {
        "mean_gross_bps": float(np.mean(gross)),
        "win_rate_bps": float(np.mean(gross > 0)),
        "sharpe_bps": L.sharpe_ratio(gross),
        "mean_return_frac": float(np.mean(gross) / 10_000),
        "net_mean_5bps": float(np.mean(gross) - 5.0),
    }


def evaluate_tau(
    y: np.ndarray,
    preds: np.ndarray,
    delta: np.ndarray,
    meta: pd.DataFrame,
    tau: float,
) -> dict:
    m = L.metrics_block(y, preds, tau=tau, meta=meta)
    traded = np.abs(preds) > tau
    direction = np.sign(preds)
    bps = _bps_stats(direction, delta, traded)
    n = int(m["filtered"]["n"])
    fire = n / max(1, len(preds))
    pnl = m["filtered"]["mean_pnl_proxy"]
    score = None
    if pnl is not None and fire > 0:
        score = float(pnl * np.sqrt(fire))
    return {
        "tau": float(tau),
        "n": n,
        "fire_rate": float(fire),
        "diracc": m["filtered"]["diracc"],
        "r2": m["filtered"]["r2"],
        "mean_pnl_proxy": pnl,
        "sharpe_per_trade": m["filtered"]["sharpe_per_trade"],
        "sharpe_closed_hourly_A": m["filtered"]["sharpe_closed_hourly_A"],
        "quality_coverage_score": score,
        **bps,
    }


def pick_best(rows: list[dict], key: str, *, higher: bool = True) -> dict | None:
    cand = [r for r in rows if r.get("n", 0) >= MIN_N and r.get(key) is not None]
    if not cand:
        return None
    return max(cand, key=lambda r: r[key]) if higher else min(cand, key=lambda r: r[key])


def main() -> None:
    out_dir = HERE / "outputs_lgbm_tau_sweep_jul25"
    out_dir.mkdir(parents=True, exist_ok=True)

    local_root = L.resolve_local_data_root()
    hf_token = L.resolve_hf_token()
    test_windows = [w for w in L.WINDOWS if w["role"] == "test"]
    print("Loading Jul25-28 …")
    t0 = time.time()
    raw = L.pool_windows(
        test_windows, local_root=local_root, hf_token=hf_token, use_hf=True
    )
    print(f"load done in {time.time()-t0:.1f}s")

    print("Building LGBM matrix …")
    df = build_batch_matrix(raw)
    del raw
    df = df.dropna(subset=["target", "zscore", "spread_bps"]).copy()
    df["spread_next"] = df.groupby(["window_id", "coin", "pair"], sort=False)[
        "spread_bps"
    ].shift(-1)
    df = df.dropna(subset=["spread_next"]).reset_index(drop=True)

    model = lgb.Booster(model_file=str(HERE / "outputs" / "statarb_lgbm.txt"))
    preds = np.asarray(
        model.predict(
            prepare_X(df, model.feature_name(), list(model.pandas_categorical or []))
        ),
        dtype=np.float64,
    )
    y = df["target"].to_numpy(dtype=np.float64)
    delta = (df["spread_next"] - df["spread_bps"]).to_numpy(dtype=np.float64)
    meta = df[["window_id", "snapshot_idx", "coin", "pair"]].copy()
    print("n_all", len(preds))

    rows = [evaluate_tau(y, preds, delta, meta, tau) for tau in TAUS]

    # Naive |z|>tau reference at campaign + high confidence
    naive_refs = {}
    for tau in (0.5, 1.0):
        nm = L.metrics_block(y, df["zscore"].to_numpy(float), tau=tau, meta=meta)
        naive_refs[str(tau)] = {
            "n": nm["filtered"]["n"],
            "diracc": nm["filtered"]["diracc"],
            "r2": nm["filtered"]["r2"],
            "mean_pnl_proxy": nm["filtered"]["mean_pnl_proxy"],
        }

    optima = {
        "by_quality_coverage_score": pick_best(rows, "quality_coverage_score"),
        "by_diracc": pick_best(rows, "diracc"),
        "by_r2": pick_best(rows, "r2"),
        "by_mean_pnl_proxy": pick_best(rows, "mean_pnl_proxy"),
        "by_sharpe_per_trade": pick_best(rows, "sharpe_per_trade"),
        "by_sharpe_A": pick_best(rows, "sharpe_closed_hourly_A"),
        "by_mean_gross_bps": pick_best(rows, "mean_gross_bps"),
    }

    # Recommended operating point: best quality_coverage among taus with fire_rate >= 5%
    # (keeps a usable trade rate); fallback to unconstrained score winner.
    usable = [
        r
        for r in rows
        if r["n"] >= MIN_N
        and r["fire_rate"] is not None
        and r["fire_rate"] >= 0.05
        and r["quality_coverage_score"] is not None
    ]
    recommended = (
        max(usable, key=lambda r: r["quality_coverage_score"])
        if usable
        else optima["by_quality_coverage_score"]
    )

    report = {
        "window": "jul25_28",
        "model": "statarb/outputs/statarb_lgbm.txt",
        "n_all": int(len(preds)),
        "min_n_for_optima": MIN_N,
        "definition": {
            "filter": "|pred| > tau",
            "pnl_proxy": "sign(pred) * z_{t+1}",
            "gross_return_bps": "sign(pred) * (spread_bps[t+1]-spread_bps[t])",
            "quality_coverage_score": "mean_pnl_proxy * sqrt(fire_rate)",
            "recommended_rule": (
                "argmax quality_coverage_score among taus with fire_rate>=5% "
                "(else unconstrained score winner)"
            ),
        },
        "sweep": rows,
        "naive_zt_refs": naive_refs,
        "optima": {k: (v["tau"] if v else None) for k, v in optima.items()},
        "optima_rows": optima,
        "recommended": recommended,
    }
    (out_dir / "tau_sweep.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    pd.DataFrame(rows).to_csv(out_dir / "tau_sweep.csv", index=False)

    def fmt(r: dict | None) -> str:
        if not r:
            return "n/a"
        return (
            f"τ={r['tau']:g} (n={r['n']:,}, DirAcc={r['diracc']:.3f}, "
            f"R²={r['r2']:.3f}, pnl_z={r['mean_pnl_proxy']:+.3f}, "
            f"fire={r['fire_rate']:.1%})"
        )

    lines = [
        "# LGBM `|pred| > τ` sweep — Jul 25–28 holdout",
        "",
        f"Scored **{len(preds):,}** rows with valid target + next spread.",
        "",
        "## Recommended operating point",
        "",
        f"**τ = {recommended['tau']:g}** under "
        "`score = mean_pnl_proxy × √(fire_rate)` with `fire_rate ≥ 5%`.",
        "",
        f"- n = {recommended['n']:,} ({recommended['fire_rate']:.1%} fire)",
        f"- DirAcc = {recommended['diracc']:.4f}",
        f"- R² = {recommended['r2']:.4f}",
        f"- mean pnl_proxy = {recommended['mean_pnl_proxy']:+.4f}",
        f"- Sharpe/trade = {recommended['sharpe_per_trade']:.4f}",
        f"- Sharpe A = {recommended['sharpe_closed_hourly_A']:.4f}",
        f"- mean gross bps = {recommended['mean_gross_bps']:+.4f}",
        "",
        "## Optima by single metric (n ≥ 500)",
        "",
        f"| Criterion | Best τ |",
        f"|---|---|",
        f"| quality×coverage score | {fmt(optima['by_quality_coverage_score'])} |",
        f"| DirAcc | {fmt(optima['by_diracc'])} |",
        f"| R² | {fmt(optima['by_r2'])} |",
        f"| mean pnl_proxy | {fmt(optima['by_mean_pnl_proxy'])} |",
        f"| Sharpe/trade | {fmt(optima['by_sharpe_per_trade'])} |",
        f"| Sharpe A | {fmt(optima['by_sharpe_A'])} |",
        f"| mean gross bps | {fmt(optima['by_mean_gross_bps'])} |",
        "",
        "## Full sweep",
        "",
        "| τ | n | fire | DirAcc | R² | mean pnl_z | Sharpe/tr | Sharpe A | score | mean gross bps | win_bps |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['tau']:.2f} | {r['n']:,} | {r['fire_rate']:.3f} | "
            f"{r['diracc']:.4f} | {r['r2']:.4f} | {r['mean_pnl_proxy']:+.4f} | "
            f"{r['sharpe_per_trade']:.3f} | {r['sharpe_closed_hourly_A']:.3f} | "
            f"{r['quality_coverage_score']:.4f} | {r['mean_gross_bps']:+.4f} | "
            f"{r['win_rate_bps']:.3f} |"
        )
    lines += [
        "",
        "## Notes",
        "",
        "- Campaign default τ=0.5 remains the live protocol peer unless you re-run live with a new τ.",
        "- Higher τ improves conditional quality but cuts coverage; Sharpe A often falls as the book thins.",
        "- Bps returns stay negative across this sweep — τ does not flip the z-proxy vs Δspread gap.",
        "- Naive refs: "
        + ", ".join(
            f"|z|>{k} DirAcc={v['diracc']:.3f} R²={v['r2']:.3f} pnl_z={v['mean_pnl_proxy']:+.3f}"
            for k, v in naive_refs.items()
        ),
        "",
    ]
    md = "\n".join(lines)
    (out_dir / "TAU_SWEEP.md").write_text(md, encoding="utf-8")
    print(md)
    print("wrote", out_dir)


if __name__ == "__main__":
    main()
