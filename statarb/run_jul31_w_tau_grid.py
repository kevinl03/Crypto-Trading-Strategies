#!/usr/bin/env python3
"""W × τ grid on the Jul 31 ~8h live collector window.

Trains spread-only LightGBM (same family as run_zscore_window_sweep.py) at each W
on the Jul25 protocol train/val split, then scores on the Jul31 collector
spread_matrix and reports metrics for each τ.

Why retrain: live signals were produced by a single W=300 model; changing W
changes the z-label, so preds must be regenerated.

Usage (from statarb/):
  python run_jul31_w_tau_grid.py
  python run_jul31_w_tau_grid.py --w 300,560,720 --taus 0.5,0.75,1.0
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

# Reuse loaders / feature builders / trainer from the W sweep.
import run_zscore_window_sweep as WSW

HERE = Path(__file__).resolve().parent
DEFAULT_JUL31_RUN = Path(
    r"C:/Users/Kev/repos/stochastic-spread-modeling/data/statarb/20260801_025316"
)
OUT = HERE / "outputs_jul31_w_tau"

DEFAULT_W = [300, 400, 560, 720]
DEFAULT_TAUS = [0.5, 0.75, 1.0]


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


def load_jul31_spread(run_dir: Path) -> pd.DataFrame:
    """Parse collector spread_matrix JSONL day shards into the sweep schema."""
    d = run_dir / "spread_matrix"
    paths = sorted(d.glob("*.jsonl"))
    if not paths:
        raise FileNotFoundError(f"no spread_matrix jsonl under {d}")
    records: list[tuple] = []
    for path in paths:
        print(f"  jul31 spread {path.name} …", flush=True)
        with path.open(encoding="utf-8-sig") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                snap = rec.get("snapshot_idx")
                coin = rec.get("coin")
                payload = rec.get("payload", rec)
                if isinstance(payload, str):
                    try:
                        payload = json.loads(payload)
                    except json.JSONDecodeError:
                        continue
                pairs = None
                if isinstance(payload, dict):
                    pairs = payload.get("pairwise_spreads") or payload.get("spreads")
                if not pairs or snap is None or coin is None:
                    continue
                for pair in pairs:
                    ex1 = pair.get("ex1") or pair.get("exchange_a")
                    ex2 = pair.get("ex2") or pair.get("exchange_b")
                    if ex1 not in WSW.TOP_EXCHANGES or ex2 not in WSW.TOP_EXCHANGES:
                        continue
                    try:
                        bps = float(pair["spread_bps"])
                    except (KeyError, TypeError, ValueError):
                        continue
                    records.append((int(snap), str(coin), str(ex1), str(ex2), bps))
    out = pd.DataFrame(
        records,
        columns=["snapshot_idx", "coin", "exchange_a", "exchange_b", "spread_bps"],
    )
    out["window_id"] = "jul31_live"
    out["spread_bps"] = out["spread_bps"].astype("float32")
    out["snapshot_idx"] = out["snapshot_idx"].astype("int32")
    print(f"  jul31 spread rows: {len(out):,}", flush=True)
    return out


def eval_tau(y: np.ndarray, preds: np.ndarray, z_lag1: np.ndarray, tau: float, meta: pd.DataFrame) -> dict:
    m = np.abs(preds) >= tau
    n = int(m.sum())
    fire = float(n / max(1, len(preds)))
    if n == 0:
        return {
            "tau": float(tau),
            "n": 0,
            "fire_rate": fire,
            "diracc": float("nan"),
            "r2": float("nan"),
            "mean_pnl_proxy": float("nan"),
            "total_pnl_proxy": 0.0,
            "sharpe_per_trade": float("nan"),
            "sharpe_A_hourly": float("nan"),
            "naive_diracc": float("nan"),
            "naive_r2": float("nan"),
            "naive_mean_pnl_proxy": float("nan"),
        }
    y_m, p_m, z_m = y[m], preds[m], z_lag1[m]
    pnl = np.sign(p_m) * y_m
    # Hourly-ish Sharpe from snapshot buckets (~40 snaps ≈ 1h at ~90s)
    sharpe_a = float("nan")
    if "snapshot_idx" in meta.columns:
        sub = meta.loc[m].copy()
        sub["pnl"] = pnl
        # ~1h buckets if snap cadence ~90s → 40 snaps/hour
        sub["hour_bucket"] = (sub["snapshot_idx"].astype(int) // 40).astype(int)
        hour_pnl = sub.groupby("hour_bucket")["pnl"].sum().to_numpy()
        sharpe_a = sharpe(hour_pnl)
    return {
        "tau": float(tau),
        "n": n,
        "fire_rate": fire,
        "diracc": diracc(y_m, p_m),
        "r2": float(r2_score(y_m, p_m)) if n > 1 else float("nan"),
        "mean_pnl_proxy": float(np.mean(pnl)),
        "total_pnl_proxy": float(np.sum(pnl)),
        "sharpe_per_trade": sharpe(pnl),
        "sharpe_A_hourly": sharpe_a,
        "naive_diracc": diracc(y_m, z_m),
        "naive_r2": float(r2_score(y_m, z_m)) if n > 1 else float("nan"),
        "naive_mean_pnl_proxy": mean_pnl(y_m, z_m),
    }


def pick_best(rows: list[dict]) -> dict:
    """Rank by filtered DirAcc, then mean pnl, then R2; require n>=200."""
    cand = [r for r in rows if r.get("n", 0) >= 200 and np.isfinite(r.get("diracc", np.nan))]
    if not cand:
        cand = [r for r in rows if r.get("n", 0) > 0]
    return max(
        cand,
        key=lambda r: (
            r.get("diracc", float("-inf")),
            r.get("mean_pnl_proxy", float("-inf")),
            r.get("r2", float("-inf")),
        ),
    )


def run_grid(
    *,
    w_grid: list[int],
    taus: list[float],
    jul31_run: Path,
    min_periods_fixed: int | None,
    out_dir: Path,
) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)
    assert WSW.LOCAL_DATA_ROOT.exists(), f"missing train data root: {WSW.LOCAL_DATA_ROOT}"
    assert jul31_run.exists(), f"missing Jul31 run_dir: {jul31_run}"

    print("=== Load train/val spreads (once) ===", flush=True)
    t0 = time.time()
    sm_train = WSW.load_all_spreads()
    print(f"  train pool wall: {time.time() - t0:.1f}s", flush=True)

    print("=== Load Jul31 live spreads ===", flush=True)
    t1 = time.time()
    sm_jul31 = load_jul31_spread(jul31_run)
    print(f"  jul31 load wall: {time.time() - t1:.1f}s", flush=True)

    rows: list[dict] = []
    for w in w_grid:
        mp = WSW.min_periods_for(w, fixed=min_periods_fixed)
        print(f"\n=== Train W={w} MIN_PERIODS={mp} ===", flush=True)
        t2 = time.time()
        feat_tr = WSW.build_spread_features(sm_train, zscore_window=w, min_periods=mp)
        print(f"  train features wall: {time.time() - t2:.1f}s", flush=True)

        def split(ids):
            return feat_tr[feat_tr["window_id"].isin(ids if isinstance(ids, list) else [ids])]

        X_tr, y_tr = WSW.feature_frame(split(WSW.TRAIN_IDS))
        X_va, y_va = WSW.feature_frame(split(WSW.VAL_ID))
        X_va = X_va.reindex(columns=X_tr.columns)
        cat = ["coin"] if "coin" in X_tr.columns else []
        print(f"  rows train/val: {len(y_tr):,} / {len(y_va):,}", flush=True)

        t3 = time.time()
        model = WSW.train_reg(X_tr, y_tr, X_va, y_va, cat)
        print(f"  train wall: {time.time() - t3:.1f}s  best_iter={model.best_iteration}", flush=True)

        model_path = out_dir / f"lgbm_spread_W{w}.txt"
        model.save_model(str(model_path))

        # Score Jul31 under the same W definition
        t4 = time.time()
        feat_te = WSW.build_spread_features(sm_jul31, zscore_window=w, min_periods=mp)
        te = feat_te.dropna(subset=["target", "zscore_lag1"]).copy()
        X_te, y_te = WSW.feature_frame(te)
        X_te = X_te.reindex(columns=X_tr.columns)
        preds = np.asarray(model.predict(X_te, num_iteration=model.best_iteration), dtype=float)
        z_lag1 = X_te["zscore_lag1"].astype(float).to_numpy()
        meta = te[["snapshot_idx", "coin", "pair"]].reset_index(drop=True)
        # feature_frame drops rows — align lengths via recompute mask
        assert len(preds) == len(y_te) == len(meta)
        print(
            f"  jul31 score wall: {time.time() - t4:.1f}s  n={len(y_te):,}",
            flush=True,
        )

        for tau in taus:
            r = eval_tau(y_te, preds, z_lag1, tau, meta)
            r.update(
                {
                    "W": int(w),
                    "min_periods": int(mp),
                    "best_iter": int(model.best_iteration or 0),
                    "n_all": int(len(y_te)),
                    "model": "lgbm_spread_only",
                }
            )
            rows.append(r)
            print(
                f"  W={w} τ={tau:g}: n={r['n']:,} DirAcc={r['diracc']:.3%} "
                f"R²={r['r2']:.3f} mean={r['mean_pnl_proxy']:+.3f} "
                f"total={r['total_pnl_proxy']:+.0f} vs naive DirAcc={r['naive_diracc']:.3%}",
                flush=True,
            )

        del feat_tr, feat_te, te, X_tr, X_va, X_te, y_tr, y_va, y_te, model, preds
        gc.collect()

    df = pd.DataFrame(rows)
    csv_path = out_dir / "jul31_w_tau_grid.csv"
    df.to_csv(csv_path, index=False)

    best = pick_best(rows)
    # Also pick best by mean pnl and by R2 for reporting
    by_mean = max(rows, key=lambda r: (r.get("n", 0) >= 200, r.get("mean_pnl_proxy") or -1e9))
    by_r2 = max(rows, key=lambda r: (r.get("n", 0) >= 200, r.get("r2") or -1e9))
    by_total = max(rows, key=lambda r: (r.get("total_pnl_proxy") or -1e9))

    report = {
        "protocol": {
            "train": WSW.TRAIN_IDS,
            "val": WSW.VAL_ID,
            "test": "jul31_live collector spread_matrix",
            "jul31_run_dir": str(jul31_run),
            "model": "spread-only LightGBM (same as W sweep; not 68-feat production)",
            "filter": "|pred| >= tau",
            "note": (
                "Each W redefines the z-label. Compare within-W lift vs naive; "
                "cross-W absolute R2/DirAcc are not identical targets."
            ),
        },
        "grid": rows,
        "best_by_diracc": best,
        "best_by_mean_pnl": by_mean,
        "best_by_r2": by_r2,
        "best_by_total_pnl": by_total,
    }
    (out_dir / "jul31_w_tau_grid.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    lines = [
        "# Jul 31 W × τ grid (spread-only LGBM)",
        "",
        f"Test window: `{jul31_run}`",
        f"W grid: {w_grid}  ·  τ grid: {taus}",
        f"MIN_PERIODS: {'fixed='+str(min_periods_fixed) if min_periods_fixed is not None else 'scale=0.3*W'}",
        "",
        "## Best picks (n ≥ 200)",
        "",
        f"- **By DirAcc:** W={best['W']} τ={best['tau']:g}  "
        f"(DirAcc={best['diracc']:.3%}, R²={best['r2']:.3f}, mean={best['mean_pnl_proxy']:+.3f}, n={best['n']:,})",
        f"- By mean pnl: W={by_mean['W']} τ={by_mean['tau']:g}  "
        f"(mean={by_mean['mean_pnl_proxy']:+.3f}, DirAcc={by_mean['diracc']:.3%}, n={by_mean['n']:,})",
        f"- By R²: W={by_r2['W']} τ={by_r2['tau']:g}  "
        f"(R²={by_r2['r2']:.3f}, DirAcc={by_r2['diracc']:.3%}, n={by_r2['n']:,})",
        f"- By total pnl: W={by_total['W']} τ={by_total['tau']:g}  "
        f"(total={by_total['total_pnl_proxy']:+.0f}, n={by_total['n']:,})",
        "",
        "## Full grid",
        "",
        "| W | τ | n | DirAcc | R² | mean pnl | total pnl | Sharpe/tr | vs naive DirAcc |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['W']} | {r['tau']:g} | {r['n']:,} | {r['diracc']:.3%} | {r['r2']:.3f} | "
            f"{r['mean_pnl_proxy']:+.3f} | {r['total_pnl_proxy']:+.0f} | "
            f"{r['sharpe_per_trade']:.3f} | {r['naive_diracc']:.3%} |"
        )
    lines += [
        "",
        "## Caveat",
        "",
        "This is **spread-only** LGBM (W-sweep family), not the 68-feature production booster.",
        "Use it to pick a W/τ pairing for the next live / 72h campaign; confirm with the full model before paper claims.",
        "",
    ]
    md = "\n".join(lines)
    (out_dir / "RESULTS.md").write_text(md, encoding="utf-8")
    print("\n" + md, flush=True)
    print(f"wrote {csv_path}", flush=True)
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--w", type=str, default=",".join(str(x) for x in DEFAULT_W))
    ap.add_argument("--taus", type=str, default=",".join(str(x) for x in DEFAULT_TAUS))
    ap.add_argument("--jul31-run", type=Path, default=DEFAULT_JUL31_RUN)
    ap.add_argument(
        "--min-periods",
        type=int,
        default=20,
        help="Fixed MIN_PERIODS for all W (default 20, matches W-sweep mp20 table)",
    )
    ap.add_argument("--out-dir", type=Path, default=OUT)
    args = ap.parse_args()

    w_grid = [int(x.strip()) for x in args.w.split(",") if x.strip()]
    taus = [float(x.strip()) for x in args.taus.split(",") if x.strip()]
    run_grid(
        w_grid=w_grid,
        taus=taus,
        jul31_run=args.jul31_run.resolve(),
        min_periods_fixed=args.min_periods,
        out_dir=args.out_dir.resolve(),
    )


if __name__ == "__main__":
    main()
