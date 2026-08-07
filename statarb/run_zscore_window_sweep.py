"""Sensitivity sweep over ZSCORE_WINDOW (W) for the Jul25 paper protocol.

Keeps cex_gbm_new.ipynb untouched. Spread-only LightGBM (same feature family as
run_improve_experiments) with:

  train  = all windows before 2026-07-25 except jul19_pre
  val    = jul19_pre  (early stopping only)
  test   = Jul 25-28  (snapshot_idx >= 3584 on jul22-28 run)

Usage (from statarb/):
  python run_zscore_window_sweep.py
  python run_zscore_window_sweep.py --w 60,120,300
  python run_zscore_window_sweep.py --min-periods 20 --out-dir outputs_w_sweep_mp20
"""
from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

LOCAL_DATA_ROOT = Path(r"C:/Users/Kev/repos/stochastic-spread-modeling/data/cex_unified")
OUT = Path("./outputs_w_sweep")  # overridden by --out-dir

TOP_EXCHANGES = ["binance", "bybit", "okx", "coinbase", "kraken", "mexc"]
HORIZON = 1
N_LAGS = 3
JUL25_CUT = 3584  # snapshot_idx cut on jul22-28 parquet

DEFAULT_W_GRID = [20, 60, 120, 180, 240, 300, 320, 400, 560, 720, 960, 1280]

LGBM_REG = {
    "objective": "regression",
    "metric": ["rmse"],
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_child_samples": 50,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.9,
    "bagging_freq": 5,
    "lambda_l1": 0.01,
    "lambda_l2": 0.1,
    "verbosity": -1,
    "n_jobs": -1,
    "seed": 42,
}
NUM_BOOST = 1500
PATIENCE = 50

WINDOWS = {
    "jun13": "",
    "jun22": "test",
    "jul13": "validation",
    "jul19_pre": "validation_jul19-22/pre_outage",
    "jul19_post": "validation_jul19-22/post_outage",
    "jul22_28": "validation_jul22-28",
}

TRAIN_IDS = ["jun13", "jun22", "jul13", "jul19_post", "jul22_24"]
VAL_ID = "jul19_pre"
TEST_ID = "jul25_28"


def min_periods_for(w: int, fixed: int | None = None) -> int:
    """Warmup before a z-score is emitted.

    fixed: use this constant for all W (must be <= W).
    default: scale ~0.3·W (paper: W=300 → 90).
    """
    if fixed is not None:
        if fixed > w:
            raise ValueError(f"min_periods={fixed} > W={w}")
        return fixed
    return max(5, int(round(0.3 * w)))


def load_spread(window_id: str) -> pd.DataFrame:
    rel = WINDOWS[window_id]
    path = LOCAL_DATA_ROOT / rel / "spread_matrix.parquet" if rel else LOCAL_DATA_ROOT / "spread_matrix.parquet"
    print(f"  spread [{window_id}] {path} …", flush=True)
    df = pd.read_parquet(path)
    if "error" in df.columns:
        df = df[df["error"].isna()].drop(columns=["error"])
    records = []
    for row in df.itertuples():
        try:
            p = json.loads(row.payload) if isinstance(row.payload, str) else row.payload
            for pair in p["pairwise_spreads"]:
                if pair["ex1"] in TOP_EXCHANGES and pair["ex2"] in TOP_EXCHANGES:
                    records.append(
                        (row.snapshot_idx, row.coin, pair["ex1"], pair["ex2"], float(pair["spread_bps"]))
                    )
        except Exception:
            continue
    del df
    gc.collect()
    out = pd.DataFrame(records, columns=["snapshot_idx", "coin", "exchange_a", "exchange_b", "spread_bps"])
    out["window_id"] = window_id
    out["spread_bps"] = out["spread_bps"].astype("float32")
    out["snapshot_idx"] = out["snapshot_idx"].astype("int32")
    print(f"    -> {out.shape}", flush=True)
    return out


def load_all_spreads() -> pd.DataFrame:
    """Load all windows; split jul22_28 into jul22_24 (train) / jul25_28 (test)."""
    frames = []
    for wid in ["jun13", "jun22", "jul13", "jul19_pre", "jul19_post"]:
        frames.append(load_spread(wid))

    jul = load_spread("jul22_28")
    pre = jul[jul["snapshot_idx"] < JUL25_CUT].copy()
    post = jul[jul["snapshot_idx"] >= JUL25_CUT].copy()
    pre["window_id"] = "jul22_24"
    post["window_id"] = "jul25_28"
    print(f"  jul22_28 split @ {JUL25_CUT}: jul22_24={pre.shape} jul25_28={post.shape}", flush=True)
    frames.extend([pre, post])
    del jul
    gc.collect()

    sm = pd.concat(frames, ignore_index=True)
    print(f"  pooled spread rows: {len(sm):,}", flush=True)
    return sm


def build_spread_features(
    sm: pd.DataFrame,
    *,
    zscore_window: int,
    min_periods: int,
    horizon: int = HORIZON,
    n_lags: int = N_LAGS,
) -> pd.DataFrame:
    sm = sm.copy()
    sm["pair"] = sm["exchange_a"] + "__" + sm["exchange_b"]
    sm = sm.sort_values(["window_id", "coin", "pair", "snapshot_idx"]).reset_index(drop=True)
    keys = ["window_id", "coin", "pair"]
    grp = sm.groupby(keys)["spread_bps"]
    roll_mean = grp.transform(lambda x: x.rolling(zscore_window, min_periods=min_periods).mean())
    roll_std = grp.transform(lambda x: x.rolling(zscore_window, min_periods=min_periods).std())
    sm["zscore"] = (sm["spread_bps"] - roll_mean) / roll_std.replace(0, np.nan)
    sm["target"] = sm.groupby(keys)["zscore"].transform(lambda x: x.shift(-horizon))
    for lag in range(1, n_lags + 1):
        sm[f"spread_bps_lag{lag}"] = grp.transform(lambda x, l=lag: x.shift(l))
        sm[f"zscore_lag{lag}"] = sm.groupby(keys)["zscore"].transform(lambda x, l=lag: x.shift(l))
    return sm


def feature_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    df = df.dropna(subset=["target", "zscore_lag1"]).copy()
    id_cols = {
        "snapshot_idx", "exchange_a", "exchange_b", "spread_bps", "zscore", "target",
        "window_id", "pair", "split",
    }
    feat_cols = [c for c in df.columns if c not in id_cols]
    X = df[feat_cols].copy()
    if "coin" in X.columns:
        X["coin"] = X["coin"].astype("category")
    y = df["target"].to_numpy()
    return X, y


def eval_reg(y, preds, label: str) -> dict:
    mae = mean_absolute_error(y, preds)
    rmse = float(np.sqrt(mean_squared_error(y, preds)))
    r2 = float(r2_score(y, preds))
    dir_acc = float(np.mean(np.sign(preds) == np.sign(y)))
    mask = np.abs(preds) > 0.5
    n_tau = int(mask.sum())
    if n_tau > 0:
        dir_acc_tau = float(np.mean(np.sign(preds[mask]) == np.sign(y[mask])))
        r2_tau = float(r2_score(y[mask], preds[mask]))
    else:
        dir_acc_tau = float("nan")
        r2_tau = float("nan")
    pnl = float(np.mean(np.sign(preds) * y))
    print(
        f"  {label:28s} n={len(y):,} R2={r2:.4f} DirAcc={dir_acc:.3%} "
        f"R2@|p|>0.5={r2_tau:.4f} DirAcc@|p|>0.5={dir_acc_tau:.3%} "
        f"PnL={pnl:.4f} (n_tau={n_tau:,})",
        flush=True,
    )
    return {
        "label": label,
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "dir_acc": dir_acc,
        "r2_tau0.5": r2_tau,
        "dir_acc_tau0.5": dir_acc_tau,
        "pnl_proxy": pnl,
        "n": int(len(y)),
        "n_tau0.5": n_tau,
    }


def train_reg(X_tr, y_tr, X_va, y_va, cat_cols: list[str]):
    dtr = lgb.Dataset(X_tr, label=y_tr, categorical_feature=cat_cols or "auto")
    dva = lgb.Dataset(X_va, label=y_va, reference=dtr, categorical_feature=cat_cols or "auto")
    return lgb.train(
        LGBM_REG,
        dtr,
        num_boost_round=NUM_BOOST,
        valid_sets=[dva],
        valid_names=["val"],
        callbacks=[lgb.early_stopping(PATIENCE, verbose=False), lgb.log_evaluation(0)],
    )


def run_sweep(w_grid: list[int], *, min_periods_fixed: int | None = None) -> pd.DataFrame:
    global OUT
    OUT.mkdir(parents=True, exist_ok=True)
    assert LOCAL_DATA_ROOT.exists(), f"missing data root: {LOCAL_DATA_ROOT}"
    mp_mode = f"fixed={min_periods_fixed}" if min_periods_fixed is not None else "scale=0.3*W"
    print(f"=== Loading spreads (once)  MIN_PERIODS mode: {mp_mode}  out={OUT} ===", flush=True)
    t0 = time.time()
    sm = load_all_spreads()
    print(f"  load wall: {time.time() - t0:.1f}s", flush=True)

    rows: list[dict] = []
    for w in w_grid:
        mp = min_periods_for(w, fixed=min_periods_fixed)
        print(f"\n=== W={w}  MIN_PERIODS={mp}  H={HORIZON}  N_LAGS={N_LAGS} ===", flush=True)
        t1 = time.time()
        feat = build_spread_features(sm, zscore_window=w, min_periods=mp)
        print(f"  features wall: {time.time() - t1:.1f}s", flush=True)

        def split(ids):
            return feat[feat["window_id"].isin(ids if isinstance(ids, list) else [ids])]

        X_tr, y_tr = feature_frame(split(TRAIN_IDS))
        X_va, y_va = feature_frame(split(VAL_ID))
        X_te, y_te = feature_frame(split(TEST_ID))
        X_va = X_va.reindex(columns=X_tr.columns)
        X_te = X_te.reindex(columns=X_tr.columns)
        cat = ["coin"] if "coin" in X_tr.columns else []
        print(f"  rows train/val/test: {len(y_tr):,} / {len(y_va):,} / {len(y_te):,}", flush=True)

        t2 = time.time()
        model = train_reg(X_tr, y_tr, X_va, y_va, cat)
        pred = model.predict(X_te, num_iteration=model.best_iteration)
        print(f"  train wall: {time.time() - t2:.1f}s  best_iter={model.best_iteration}", flush=True)

        r = eval_reg(y_te, pred, f"lgbm W={w}")
        r.update(
            {
                "zscore_window": w,
                "min_periods": mp,
                "horizon": HORIZON,
                "n_lags": N_LAGS,
                "model": "lgbm_reg",
                "best_iter": int(model.best_iteration or 0),
                "n_train": int(len(y_tr)),
                "n_val": int(len(y_va)),
            }
        )
        rows.append(r)

        if "zscore_lag1" in X_te.columns:
            rn = eval_reg(y_te, X_te["zscore_lag1"].astype(float).to_numpy(), f"naive W={w}")
            rn.update(
                {
                    "zscore_window": w,
                    "min_periods": mp,
                    "horizon": HORIZON,
                    "n_lags": N_LAGS,
                    "model": "naive_zlag1",
                    "best_iter": 0,
                    "n_train": int(len(y_tr)),
                    "n_val": int(len(y_va)),
                }
            )
            rows.append(rn)

        del feat, X_tr, X_va, X_te, y_tr, y_va, y_te, model
        gc.collect()

    out = pd.DataFrame(rows)
    csv_path = OUT / "w_sweep.csv"
    out.to_csv(csv_path, index=False)
    print(f"\nWrote {csv_path}", flush=True)

    lgbm = out[out["model"] == "lgbm_reg"].sort_values("r2", ascending=False)
    summary_lines = [
        "ZSCORE_WINDOW (W) sensitivity — Jul25 protocol (spread-only LGBM)",
        f"train={TRAIN_IDS}  val={VAL_ID}  test={TEST_ID}  cut={JUL25_CUT}",
        f"HORIZON={HORIZON}  N_LAGS={N_LAGS}  MIN_PERIODS={mp_mode}",
        f"W grid: {w_grid}",
        "",
        "Ranked by test R2 (lgbm):",
    ]
    for _, row in lgbm.iterrows():
        summary_lines.append(
            f"  W={int(row['zscore_window']):3d}  R2={row['r2']:.4f}  DirAcc={row['dir_acc']:.3%}  "
            f"R2@|p|>0.5={row['r2_tau0.5']:.4f}  DirAcc@|p|>0.5={row['dir_acc_tau0.5']:.3%}  "
            f"PnL={row['pnl_proxy']:.4f}  best_iter={int(row['best_iter'])}  n={int(row['n']):,}"
        )
    best = lgbm.iloc[0]
    summary_lines.extend(
        [
            "",
            f"BEST by R2: W={int(best['zscore_window'])} "
            f"(R2={best['r2']:.4f}, DirAcc={best['dir_acc']:.3%})",
            "",
            "Note: each W redefines the z-label; compare relative lift vs naive on the same W.",
        ]
    )
    summary = "\n".join(summary_lines) + "\n"
    (OUT / "SUMMARY.txt").write_text(summary, encoding="utf-8")
    print("\n" + summary, flush=True)
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ZSCORE_WINDOW sensitivity sweep")
    p.add_argument(
        "--w",
        type=str,
        default=",".join(str(x) for x in DEFAULT_W_GRID),
        help="Comma-separated W values",
    )
    p.add_argument(
        "--min-periods",
        type=int,
        default=None,
        help="Fixed MIN_PERIODS for all W (default: scale as round(0.3*W))",
    )
    p.add_argument(
        "--out-dir",
        type=str,
        default="outputs_w_sweep",
        help="Output directory relative to cwd (default: outputs_w_sweep)",
    )
    return p.parse_args()


def main() -> None:
    global OUT
    args = parse_args()
    OUT = Path(args.out_dir)
    w_grid = [int(x.strip()) for x in args.w.split(",") if x.strip()]
    if not w_grid:
        raise SystemExit("empty --w grid")
    for w in w_grid:
        if w < 5:
            raise SystemExit(f"W too small: {w}")
        if args.min_periods is not None and args.min_periods > w:
            raise SystemExit(f"min_periods={args.min_periods} > W={w}")
    run_sweep(w_grid, min_periods_fixed=args.min_periods)


if __name__ == "__main__":
    main()
