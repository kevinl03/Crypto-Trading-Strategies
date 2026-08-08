"""Retrain LightGBM on spread-change (bps) targets net of fees.

Paper protocol (same as cex_gbm_new / LOGO): H=1, W=300, N_LAGS=3,
Jul 25-28 test cut, published LGBM_PARAMS.

Targets compared:
  1) zscore_fwd          — published objective (baseline)
  2) bps_gross           — ΔS_bps = spread_{t+H} - spread_t
  3) bps_net_flat16      — ΔS_bps - 16  (4 legs × 4 bps)
  4) bps_net_pair_fee    — ΔS_bps - pair round-trip taker fee (scripts/fees.py)

Entry rules on the held-out test set:
  - z model: |pred| >= 0.5, direction = sign(pred)
  - bps models: pred >= entry_min (0 for net targets; 16 or pair fee for gross)

Usage:
  python statarb/run_bps_net_retrain.py
  python statarb/run_bps_net_retrain.py --force-rebuild
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from scripts.fees import TRADING_FEES_TAKER  # noqa: E402

CELL_DUMP = HERE / "_nb_cell_dump"
OUT = HERE / "outputs_bps_net"
CACHE = OUT / "cache"
OUT.mkdir(parents=True, exist_ok=True)
CACHE.mkdir(parents=True, exist_ok=True)

HORIZON = 1
ZSCORE_WINDOW = 300
N_LAGS = 3
MIN_PERIODS = 90
FLAT_FEE_BPS = 16.0

LGBM_PARAMS = {
    "objective": "regression",
    "metric": ["rmse", "mae"],
    "learning_rate": 0.1,
    "num_leaves": 255,
    "min_child_samples": 200,
    "feature_fraction": 0.6,
    "bagging_fraction": 0.7,
    "bagging_freq": 5,
    "lambda_l1": 0.1,
    "lambda_l2": 0.1,
    "verbosity": -1,
    "n_jobs": -1,
    "seed": 42,
}
NUM_BOOST_ROUND = 2500
EARLY_STOPPING = 250

ID_COLS = {
    "snapshot_idx",
    "exchange_a",
    "exchange_b",
    "spread_bps",
    "zscore",
    "target",
    "target_z",
    "target_bps_gross",
    "target_bps_net_flat16",
    "target_bps_net_pair",
    "pair_fee_bps",
    "p1",
    "p2",
    "window_id",
}


def pair_round_trip_fee_bps(pair: str, default_leg_bps: float = 4.0) -> float:
    if not isinstance(pair, str) or "__" not in pair:
        return 4.0 * default_leg_bps
    a, b = pair.split("__", 1)
    fa = TRADING_FEES_TAKER.get(a)
    fb = TRADING_FEES_TAKER.get(b)
    if fa is None or fb is None:
        return 4.0 * default_leg_bps
    return float(2.0 * (fa + fb) * 10_000)


def _exec_cell(ns: dict, cell_id: int) -> None:
    path = CELL_DUMP / f"cell_{cell_id:03d}.py"
    code = path.read_text(encoding="utf-8")
    exec(compile(code, str(path), "exec"), ns, ns)


def _patch_spread_builder(ns: dict) -> None:
    """Replace build_spread_features to also emit bps gross / net targets."""

    def build_spread_features(sm: pd.DataFrame) -> pd.DataFrame:
        if sm.empty:
            return pd.DataFrame()
        sm = sm.copy()
        sm["pair"] = sm["exchange_a"] + "__" + sm["exchange_b"]
        sm = sm.sort_values(["window_id", "coin", "pair", "snapshot_idx"]).reset_index(drop=True)
        grp_keys = ["window_id", "coin", "pair"]
        grp = sm.groupby(grp_keys)["spread_bps"]
        roll_mean = grp.transform(lambda x: x.rolling(ZSCORE_WINDOW, min_periods=MIN_PERIODS).mean())
        roll_std = grp.transform(lambda x: x.rolling(ZSCORE_WINDOW, min_periods=MIN_PERIODS).std())
        sm["zscore"] = (sm["spread_bps"] - roll_mean) / roll_std.replace(0, np.nan)
        del roll_mean, roll_std

        sm["target_z"] = sm.groupby(grp_keys)["zscore"].transform(lambda x: x.shift(-HORIZON))
        sm["target"] = sm["target_z"]  # keep notebook name for prune corr
        sm["target_bps_gross"] = grp.transform(lambda x: x.shift(-HORIZON) - x)
        sm["pair_fee_bps"] = sm["pair"].map(pair_round_trip_fee_bps).astype("float32")
        sm["target_bps_net_flat16"] = sm["target_bps_gross"] - FLAT_FEE_BPS
        sm["target_bps_net_pair"] = sm["target_bps_gross"] - sm["pair_fee_bps"]

        for lag in range(1, N_LAGS + 1):
            sm[f"spread_bps_lag{lag}"] = grp.transform(lambda x, l=lag: x.shift(l))
            sm[f"zscore_lag{lag}"] = sm.groupby(grp_keys)["zscore"].transform(lambda x, l=lag: x.shift(l))

        sm = sm.drop(columns=["exchange_a", "exchange_b", "p1", "p2"], errors="ignore")
        return sm

    ns["build_spread_features"] = build_spread_features
    ns["HORIZON"] = HORIZON
    ns["ZSCORE_WINDOW"] = ZSCORE_WINDOW
    ns["N_LAGS"] = N_LAGS
    ns["MIN_PERIODS"] = MIN_PERIODS


def build_frames(force: bool = False) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    train_path = CACHE / "df_train_bps.parquet"
    test_path = CACHE / "df_test_bps.parquet"
    meta_path = CACHE / "meta.json"
    if not force and train_path.exists() and test_path.exists():
        print(f"Loading cached bps frames from {CACHE}")
        return (
            pd.read_parquet(train_path),
            pd.read_parquet(test_path),
            json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {},
        )

    if not CELL_DUMP.exists():
        raise SystemExit(f"Missing {CELL_DUMP}; dump notebook cells first")

    print("=" * 60)
    print("Building feature matrices with bps targets (slow once) …")
    print("=" * 60)
    t0 = time.time()
    ns: dict = {"__name__": "__bps_build__"}
    _exec_cell(ns, 3)
    _exec_cell(ns, 4)
    ns["OUTPUT_DIR"] = OUT
    ns["HORIZON"] = HORIZON
    ns["ZSCORE_WINDOW"] = ZSCORE_WINDOW
    ns["N_LAGS"] = N_LAGS
    ns["MIN_PERIODS"] = MIN_PERIODS

    # UTF-8 stdout for notebook box-drawing chars on Windows
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    _exec_cell(ns, 6)
    for cid in (8, 10, 12, 14, 16):
        _exec_cell(ns, cid)
    _patch_spread_builder(ns)

    merge_src = (CELL_DUMP / "cell_018.py").read_text(encoding="utf-8")
    merge_fn = merge_src.split('print("Building train')[0]
    exec(compile(merge_fn, "cell_018_fn", "exec"), ns, ns)

    print("\nBuilding train …")
    df_train = ns["build_feature_matrix"]({k: v for k, v in ns["train_raw"].items()})
    gc.collect()
    print(f"train: {df_train.shape}")

    print("\nBuilding test …")
    df_test = ns["build_feature_matrix"]({k: v for k, v in ns["test_raw"].items()})
    gc.collect()
    print(f"test: {df_test.shape}")

    # Prune like paper (on numeric feature cols only — not targets)
    import scipy.stats  # noqa: F401

    feature_cols = [
        c
        for c in df_train.columns
        if c
        not in {
            "snapshot_idx",
            "exchange_a",
            "exchange_b",
            "spread_bps",
            "zscore",
            "target",
            "target_z",
            "target_bps_gross",
            "target_bps_net_flat16",
            "target_bps_net_pair",
            "pair_fee_bps",
            "p1",
            "p2",
            "coin",
            "pair",
            "window_id",
        }
    ]
    report = []
    for col in feature_cols:
        s = df_train[col].dropna()
        if len(s) < 100:
            continue
        null_pct = float(df_train[col].isna().mean())
        corr = df_train[[col, "target"]].dropna().corr().iloc[0, 1]
        report.append(
            {
                "feature": col,
                "null_pct": round(null_pct, 3),
                "corr_target": round(float(corr), 4) if pd.notna(corr) else 0.0,
            }
        )
    report_df = pd.DataFrame(report)
    drop_null = report_df[report_df["null_pct"] > 0.4]["feature"].tolist()
    drop_weak = report_df[report_df["corr_target"].abs() < 0.005]["feature"].tolist()
    all_drops = list(set(drop_null + drop_weak))
    print(f"Pruning {len(all_drops)} feature columns")
    for df in (df_train, df_test):
        df.drop(columns=[c for c in all_drops if c in df.columns], inplace=True)

    def signed_log1p(s):
        return np.sign(s) * np.log1p(np.abs(s))

    log_patterns = ["tk_bid_volume", "tk_ask_volume", "tr_total_volume", "tr_buy_sell_ratio"]
    log_cols = [c for c in df_train.columns if any(p in c for p in log_patterns)]
    for df in (df_train, df_test):
        for col in log_cols:
            if col in df.columns:
                df[col] = signed_log1p(df[col])

    from scipy.stats.mstats import winsorize

    winsor_cols = [
        c for c in df_train.columns if any(p in c for p in ("spread_bps_lag", "tk_spread_bps"))
    ]
    for df in (df_train, df_test):
        for col in winsor_cols:
            if col in df.columns:
                med = df[col].median()
                df[col] = winsorize(df[col].fillna(med), limits=[0.01, 0.01])

    cross_src = (CELL_DUMP / "cell_027.py").read_text(encoding="utf-8")
    cross_fn = cross_src.split("df_train = add_cross_exchange_features")[0]
    exec(compile(cross_fn, "cell_027_fn", "exec"), ns, ns)
    df_train = ns["add_cross_exchange_features"](df_train)
    df_test = ns["add_cross_exchange_features"](df_test)

    # Ensure fee/net columns survived (recompute if cross copy dropped them — shouldn't)
    for df in (df_train, df_test):
        if "pair_fee_bps" not in df.columns and "pair" in df.columns:
            df["pair_fee_bps"] = df["pair"].map(pair_round_trip_fee_bps).astype("float32")
        if "target_bps_gross" in df.columns:
            df["target_bps_net_flat16"] = df["target_bps_gross"] - FLAT_FEE_BPS
            if "pair_fee_bps" in df.columns:
                df["target_bps_net_pair"] = df["target_bps_gross"] - df["pair_fee_bps"]

    df_train.to_parquet(train_path, index=False)
    df_test.to_parquet(test_path, index=False)
    meta = {
        "n_train": int(len(df_train)),
        "n_test": int(len(df_test)),
        "n_cols": int(df_train.shape[1]),
        "build_seconds": round(time.time() - t0, 1),
        "flat_fee_bps": FLAT_FEE_BPS,
        "dropped": all_drops,
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Cached in {meta['build_seconds']}s → {train_path}")
    return df_train, df_test, meta


def prepare_xy(df: pd.DataFrame, target_col: str, reference_cols=None):
    df = df.dropna(subset=[target_col]).copy()
    cat_cols = [c for c in ["coin", "pair"] if c in df.columns]
    for c in cat_cols:
        df[c] = df[c].astype("category")
    feat_cols = [c for c in df.columns if c not in ID_COLS]
    X = df[feat_cols].copy()
    y = df[target_col].to_numpy(float)
    meta = {
        "pair_fee_bps": df["pair_fee_bps"].to_numpy(float) if "pair_fee_bps" in df.columns else None,
        "bps_gross": df["target_bps_gross"].to_numpy(float) if "target_bps_gross" in df.columns else None,
    }
    if reference_cols is not None:
        X = X.reindex(columns=reference_cols)
        for c in cat_cols:
            if c in X.columns:
                X[c] = X[c].astype("category")
    return X, y, feat_cols, cat_cols, meta


def train_model(X_tr, y_tr, X_te, y_te, cat_cols):
    cats = [c for c in cat_cols if c in X_tr.columns]
    dtrain = lgb.Dataset(X_tr, label=y_tr, categorical_feature=cats, free_raw_data=False)
    dtest = lgb.Dataset(X_te, label=y_te, categorical_feature=cats, reference=dtrain, free_raw_data=False)
    model = lgb.train(
        LGBM_PARAMS,
        dtrain,
        num_boost_round=NUM_BOOST_ROUND,
        valid_sets=[dtrain, dtest],
        valid_names=["train", "test"],
        callbacks=[lgb.early_stopping(EARLY_STOPPING, verbose=False), lgb.log_evaluation(0)],
    )
    n = int(model.best_iteration or model.current_iteration())
    preds = model.predict(X_te, num_iteration=n)
    return model, preds, n


def eval_regression(y, preds) -> dict:
    return {
        "mae": float(mean_absolute_error(y, preds)),
        "rmse": float(np.sqrt(mean_squared_error(y, preds))),
        "r2": float(r2_score(y, preds)),
        "dir_acc": float(np.mean(np.sign(preds) == np.sign(y))),
    }


def simulate_trades(
    preds: np.ndarray,
    bps_gross: np.ndarray,
    pair_fees: np.ndarray | None,
    *,
    mode: str,
    entry_min: float,
    use_abs: bool,
) -> dict:
    """Apply entry rule and score realized net bps."""
    if use_abs:
        mask = np.abs(preds) >= entry_min
        direction = np.sign(preds)
    else:
        mask = preds >= entry_min
        direction = np.ones(len(preds))  # long the predicted positive ΔS
        # For bps models predicting ΔS itself, direction of trade on the spread:
        # if pred > 0 expect spread to widen → direction +1 on spread_delta
        direction = np.where(preds >= 0, 1.0, -1.0)
        mask = np.abs(preds) >= entry_min if use_abs else (preds >= entry_min)

    # For gross/net bps models: prediction is E[ΔS] or E[ΔS - fee].
    # Realized gross for a signed bet: if we always "take the predicted sign",
    # pnl_gross = sign(pred) * realized_ΔS, but realized_ΔS IS target_bps_gross
    # and the label already is ΔS (not direction*ΔS). So if we enter only when
    # pred >= entry_min (expecting positive ΔS), realized gross = bps_gross.
    if mode in ("bps_gross", "bps_net_flat16", "bps_net_pair"):
        if use_abs:
            # directional: take sign(pred) * realized ΔS
            realized_gross = direction * bps_gross
            trade_mask = np.abs(preds) >= entry_min
        else:
            # long-only on predicted positive edge
            realized_gross = bps_gross
            trade_mask = preds >= entry_min
    else:
        # z-model: direction = sign(pred), settlement in bps = direction * ΔS
        realized_gross = np.sign(preds) * bps_gross
        trade_mask = np.abs(preds) >= entry_min

    trade_mask = trade_mask & np.isfinite(realized_gross) & np.isfinite(preds)
    n = int(trade_mask.sum())
    if n == 0:
        return {
            "n_trades": 0,
            "pct_rows": 0.0,
            "mean_gross_bps": None,
            "mean_net_flat16": None,
            "mean_net_pair_fee": None,
            "win_rate_gross": None,
            "win_rate_net_flat16": None,
            "entry_min": entry_min,
            "mode": mode,
        }

    g = realized_gross[trade_mask]
    fees = pair_fees[trade_mask] if pair_fees is not None else np.full(n, FLAT_FEE_BPS)
    net16 = g - FLAT_FEE_BPS
    net_pair = g - fees
    return {
        "n_trades": n,
        "pct_rows": float(n / len(preds)),
        "mean_gross_bps": float(np.mean(g)),
        "median_gross_bps": float(np.median(g)),
        "mean_net_flat16": float(np.mean(net16)),
        "mean_net_pair_fee": float(np.mean(net_pair)),
        "win_rate_gross": float(np.mean(g > 0)),
        "win_rate_net_flat16": float(np.mean(net16 > 0)),
        "win_rate_net_pair_fee": float(np.mean(net_pair > 0)),
        "entry_min": entry_min,
        "mode": mode,
    }


def run(force_rebuild: bool = False) -> pd.DataFrame:
    df_train, df_test, meta = build_frames(force=force_rebuild)

    # Sanity on targets
    for col in ("target_z", "target_bps_gross", "target_bps_net_flat16", "target_bps_net_pair"):
        if col not in df_train.columns:
            raise SystemExit(f"Missing {col} in train frame — rebuild with --force-rebuild")
        print(
            f"  {col}: train mean={df_train[col].mean():.4f}  "
            f"std={df_train[col].std():.4f}  null%={df_train[col].isna().mean():.2%}"
        )

    configs = [
        {
            "name": "zscore_fwd",
            "target": "target_z",
            "entry_min": 0.5,
            "use_abs": True,
            "mode": "z",
        },
        {
            "name": "bps_gross",
            "target": "target_bps_gross",
            "entry_min": FLAT_FEE_BPS,  # only trade when E[ΔS] >= 16
            "use_abs": False,
            "mode": "bps_gross",
        },
        {
            "name": "bps_gross_entry0",
            "target": "target_bps_gross",
            "entry_min": 0.0,
            "use_abs": False,
            "mode": "bps_gross",
            "reuse_model": "bps_gross",
        },
        {
            "name": "bps_net_flat16",
            "target": "target_bps_net_flat16",
            "entry_min": 0.0,  # pred is already net of 16
            "use_abs": False,
            "mode": "bps_net_flat16",
        },
        {
            "name": "bps_net_pair_fee",
            "target": "target_bps_net_pair",
            "entry_min": 0.0,
            "use_abs": False,
            "mode": "bps_net_pair",
        },
    ]

    models: dict[str, lgb.Booster] = {}
    preds_cache: dict[str, np.ndarray] = {}
    rows = []

    for cfg in configs:
        name = cfg["name"]
        reuse = cfg.get("reuse_model")
        print(f"\n=== {name} (target={cfg['target']}) ===", flush=True)
        t0 = time.time()

        if reuse and reuse in preds_cache:
            preds = preds_cache[reuse]
            # Need y for regression metrics on this target
            _, y_te, _, _, meta_te = prepare_xy(df_test, cfg["target"])
            # Align: prepare_xy drops NaN on target — must match reuse model's test index
            X_tr, y_tr, feat_cols, cat_cols, _ = prepare_xy(df_train, configs[1]["target"])
            X_te, y_te_g, _, _, meta_te = prepare_xy(df_test, configs[1]["target"], reference_cols=feat_cols)
            # For reused bps_gross model, y for entry0 is same gross target
            y_te = y_te_g
            best_iter = models[reuse].best_iteration
            reg = eval_regression(y_te, preds)
        else:
            X_tr, y_tr, feat_cols, cat_cols, _ = prepare_xy(df_train, cfg["target"])
            X_te, y_te, _, _, meta_te = prepare_xy(df_test, cfg["target"], reference_cols=feat_cols)
            print(f"  train={X_tr.shape} test={X_te.shape}", flush=True)
            model, preds, best_iter = train_model(X_tr, y_tr, X_te, y_te, cat_cols)
            models[name] = model
            preds_cache[name] = preds
            model.save_model(str(OUT / f"lgbm_{name}.txt"))
            reg = eval_regression(y_te, preds)
            print(f"  best_iter={best_iter}  R²={reg['r2']:.4f}  DirAcc={reg['dir_acc']:.3%}  ({time.time()-t0:.1f}s)")

        bps_gross = meta_te["bps_gross"]
        pair_fees = meta_te["pair_fee_bps"]
        # Drop rows where bps_gross is nan (should already be aligned)
        ok = np.isfinite(bps_gross) & np.isfinite(preds)
        trade = simulate_trades(
            preds[ok],
            bps_gross[ok],
            pair_fees[ok] if pair_fees is not None else None,
            mode=cfg["mode"],
            entry_min=cfg["entry_min"],
            use_abs=cfg["use_abs"],
        )
        print(
            f"  trades={trade['n_trades']}  mean_gross={trade['mean_gross_bps']}  "
            f"mean_net16={trade['mean_net_flat16']}  mean_net_pair={trade['mean_net_pair_fee']}"
        )
        rows.append(
            {
                "model": name,
                "target": cfg["target"],
                "best_iter": int(best_iter) if best_iter is not None else None,
                "entry_min": cfg["entry_min"],
                "use_abs_entry": cfg["use_abs"],
                **{f"reg_{k}": v for k, v in reg.items()},
                **{f"trade_{k}": v for k, v in trade.items()},
                "wall_s": round(time.time() - t0, 1),
            }
        )

    out = pd.DataFrame(rows)
    out.to_csv(OUT / "bps_net_retrain_results.csv", index=False)

    # Markdown summary
    lines = [
        "# Bps-net-of-cost retrain results",
        "",
        "Retrained LightGBM on the Jul25–28 protocol with spread-change targets.",
        f"Flat fee assumption: **{FLAT_FEE_BPS} bps** round-trip. Pair fees from `scripts/fees.py`.",
        "",
        "| model | R² | DirAcc | n trades | mean gross | mean net@16 | mean net@pair | win net@16 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, r in out.iterrows():
        lines.append(
            f"| {r['model']} | {r['reg_r2']:.4f} | {r['reg_dir_acc']:.3f} | "
            f"{int(r['trade_n_trades'])} | "
            f"{r['trade_mean_gross_bps'] if pd.notna(r['trade_mean_gross_bps']) else '—'} | "
            f"{r['trade_mean_net_flat16'] if pd.notna(r['trade_mean_net_flat16']) else '—'} | "
            f"{r['trade_mean_net_pair_fee'] if pd.notna(r['trade_mean_net_pair_fee']) else '—'} | "
            f"{r['trade_win_rate_net_flat16'] if pd.notna(r['trade_win_rate_net_flat16']) else '—'} |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
    ]
    # Auto verdict
    best = out.dropna(subset=["trade_mean_net_flat16"])
    if len(best) and best["trade_mean_net_flat16"].max() > 0:
        top = best.loc[best["trade_mean_net_flat16"].idxmax()]
        lines.append(
            f"**Positive mean net@16** achieved by `{top['model']}` "
            f"(mean net={top['trade_mean_net_flat16']:.3f} bps, n={int(top['trade_n_trades'])})."
        )
    else:
        lines.append(
            "No trained variant achieves **positive mean net@16** on the Jul25–28 test set. "
            "Changing the target alone is not enough at this horizon / fee level; "
            "need longer holds, maker pairs, or lower execution cost."
        )
    (OUT / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    (ROOT / "docs" / "bps_net_retrain_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n" + "\n".join(lines))
    return out


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--force-rebuild", action="store_true")
    args = ap.parse_args()
    run(force_rebuild=args.force_rebuild)


if __name__ == "__main__":
    main()
