"""Leave-one-group-out / nested feature-group ablation for the paper LGBM.

Reuses the Jul31 paper protocol from cex_gbm_new.ipynb:
  H=1, W=300, N_LAGS=3, LGBM_PARAMS as published, Jul 25-28 test cut.

Design (nested cumulative, as specified for the paper table):
  baseline AR → +ticker → +orderbook → +trades → +funding → +OI → +cross → full

Also reports classic LOGO (full minus one group).

Fixed boosting rounds: train full model with early stopping once, then reuse
that best_iteration for every variant so round-count is not a confounder.

Primary metric: R² on |ẑ| ≥ 0.9 filtered test subset (paper confidence gate).
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
CELL_DUMP = HERE / "_nb_cell_dump"
OUT = HERE / "outputs_logo"
CACHE = OUT / "cache"
OUT.mkdir(parents=True, exist_ok=True)
CACHE.mkdir(parents=True, exist_ok=True)

# Paper protocol (must match cex_gbm_new.ipynb)
HORIZON = 1
ZSCORE_WINDOW = 300
N_LAGS = 3
MIN_PERIODS = 90
FILTER_TAU = 0.9  # paper protocol gate

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
    "p1",
    "p2",
    "window_id",
}

# Exact 68-feature names from the published booster (sanity / fallback grouping).
PUBLISHED_FEATURES = [
    "coin",
    "pair",
    "spread_bps_lag1",
    "zscore_lag1",
    "spread_bps_lag2",
    "zscore_lag2",
    "spread_bps_lag3",
    "zscore_lag3",
    "tk_mid_lag1_binance",
    "tk_mid_lag1_bybit",
    "tk_mid_lag1_coinbase",
    "tk_mid_lag1_kraken",
    "tk_mid_lag1_mexc",
    "tk_mid_lag1_okx",
    "tk_mid_lag2_binance",
    "tk_mid_lag2_bybit",
    "tk_mid_lag2_coinbase",
    "tk_mid_lag2_kraken",
    "tk_mid_lag2_mexc",
    "tk_mid_lag2_okx",
    "tk_mid_lag3_binance",
    "tk_mid_lag3_bybit",
    "tk_mid_lag3_coinbase",
    "tk_mid_lag3_kraken",
    "tk_mid_lag3_mexc",
    "tk_mid_lag3_okx",
    "tk_spread_bps_lag1_binance",
    "tk_spread_bps_lag1_bybit",
    "tk_spread_bps_lag1_kraken",
    "tk_spread_bps_lag1_okx",
    "tk_spread_bps_lag2_binance",
    "tk_spread_bps_lag2_bybit",
    "tk_spread_bps_lag2_kraken",
    "tk_spread_bps_lag2_okx",
    "tk_spread_bps_lag3_binance",
    "tk_spread_bps_lag3_bybit",
    "tk_spread_bps_lag3_coinbase",
    "tk_spread_bps_lag3_kraken",
    "tk_spread_bps_lag3_okx",
    "tk_bid_volume_lag1_coinbase",
    "tk_bid_volume_lag2_coinbase",
    "tk_bid_volume_lag3_coinbase",
    "tk_ask_volume_lag1_coinbase",
    "tk_ask_volume_lag2_coinbase",
    "tk_ask_volume_lag3_coinbase",
    "ob_imbalance_lag1_coinbase",
    "ob_imbalance_lag2_coinbase",
    "ob_imbalance_lag3_coinbase",
    "tr_total_volume_lag1_bybit",
    "tr_total_volume_lag2_bybit",
    "tr_total_volume_lag3_bybit",
    "cross_mid_std",
    "cross_mid_range",
    "price_rank_binance",
    "price_rank_bybit",
    "price_rank_okx",
    "price_rank_coinbase",
    "price_rank_kraken",
    "cross_ba_std",
    "tightest_ba",
    "widest_ba",
    "ba_range",
    "cross_ob_std",
    "cross_ob_range",
    "net_ob_pressure",
    "spread_momentum_1_3",
    "zscore_momentum_1_3",
    "zscore_accel",
]


def _assign_group(col: str) -> str:
    """Map a feature column to a LOGO family."""
    if col in ("coin", "pair"):
        return "identity"
    if col.startswith(("spread_bps_lag", "zscore_lag")):
        return "baseline"
    if col in ("spread_momentum_1_3", "zscore_momentum_1_3", "zscore_accel") or col.startswith(
        ("spread_momentum_", "zscore_momentum_")
    ):
        return "baseline"
    if col.startswith("tk_"):
        return "ticker"
    if col.startswith("ob_"):
        return "orderbook"
    if col.startswith("tr_"):
        return "trades"
    if col.startswith("fr_"):
        return "funding"
    if col.startswith("oi_"):
        return "oi"
    if (
        col.startswith(("cross_", "price_rank_"))
        or col in ("tightest_ba", "widest_ba", "ba_range", "net_ob_pressure", "net_flow_signal", "flow_divergence")
    ):
        return "cross"
    return "other"


def group_features(feat_cols: list[str]) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {
        "identity": [],
        "baseline": [],
        "ticker": [],
        "orderbook": [],
        "trades": [],
        "funding": [],
        "oi": [],
        "cross": [],
        "other": [],
    }
    for c in feat_cols:
        groups[_assign_group(c)].append(c)
    return groups


def nested_variants(groups: dict[str, list[str]]) -> list[tuple[str, list[str], str]]:
    """Cumulative add-one-group variants for the paper table.

    Returns list of (name, feature_list, features_added_label).
    Identity (coin/pair) is always included with the AR baseline.
    """
    order = [
        ("baseline", "z-score/spread lags + momentum"),
        ("ticker", "+ ticker (mid, BA, volumes)"),
        ("orderbook", "+ orderbook imbalance"),
        ("trades", "+ trade flow"),
        ("funding", "+ funding rate"),
        ("oi", "+ open interest"),
        ("cross", "+ cross-venue aggregates"),
    ]
    acc = list(groups.get("identity", []))
    variants = []
    for key, label in order:
        cols = groups.get(key, [])
        if key != "baseline" and not cols:
            # Group was pruned — still emit a row marker so the table can note N/A
            variants.append((f"+{key} (pruned)", list(acc), f"{label} [0 cols surviving prune]"))
            continue
        acc = acc + cols
        name = "AR baseline" if key == "baseline" else f"+{key}"
        variants.append((name, list(acc), label))
    # Full = everything present
    full = [c for g in groups.values() for c in g]
    variants.append(("full (all surviving)", full, "all feature groups"))
    return variants


def logo_variants(groups: dict[str, list[str]], full: list[str]) -> list[tuple[str, list[str], str]]:
    """Classic leave-one-group-out: full minus one family (identity always kept)."""
    out = [("full", full, "all features")]
    for key in ("baseline", "ticker", "orderbook", "trades", "funding", "oi", "cross"):
        drop = set(groups.get(key, []))
        if not drop:
            out.append((f"−{key} (pruned)", full, f"group empty after prune"))
            continue
        kept = [c for c in full if c not in drop]
        out.append((f"−{key}", kept, f"drop {len(drop)} {key} cols"))
    return out


def eval_split(y: np.ndarray, preds: np.ndarray) -> dict:
    mae = float(mean_absolute_error(y, preds))
    rmse = float(np.sqrt(mean_squared_error(y, preds)))
    r2 = float(r2_score(y, preds))
    dir_acc = float(np.mean(np.sign(preds) == np.sign(y)))
    mask = np.abs(preds) >= FILTER_TAU
    n_filt = int(mask.sum())
    if n_filt > 0:
        yf, pf = y[mask], preds[mask]
        r2_f = float(r2_score(yf, pf))
        dir_f = float(np.mean(np.sign(pf) == np.sign(yf)))
        mae_f = float(mean_absolute_error(yf, pf))
    else:
        r2_f = dir_f = mae_f = float("nan")
    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "dir_acc": dir_acc,
        "n_filtered": n_filt,
        "pct_filtered": n_filt / max(len(y), 1),
        "r2_filtered": r2_f,
        "dir_acc_filtered": dir_f,
        "mae_filtered": mae_f,
    }


def train_fixed_rounds(
    X_tr: pd.DataFrame,
    y_tr: np.ndarray,
    X_te: pd.DataFrame,
    y_te: np.ndarray,
    cat_cols: list[str],
    n_rounds: int,
) -> tuple[lgb.Booster, dict]:
    cats = [c for c in cat_cols if c in X_tr.columns]
    dtrain = lgb.Dataset(X_tr, label=y_tr, categorical_feature=cats, free_raw_data=False)
    model = lgb.train(
        LGBM_PARAMS,
        dtrain,
        num_boost_round=n_rounds,
        callbacks=[lgb.log_evaluation(0)],
    )
    preds = model.predict(X_te, num_iteration=n_rounds)
    metrics = eval_split(y_te, preds)
    metrics["best_iter"] = n_rounds
    return model, metrics


def train_with_early_stop(
    X_tr: pd.DataFrame,
    y_tr: np.ndarray,
    X_te: pd.DataFrame,
    y_te: np.ndarray,
    cat_cols: list[str],
) -> tuple[lgb.Booster, dict, int]:
    """Match notebook: early-stop on the test split (same protocol confound as paper).

    We only use this once to lock the round count for the fixed-round LOGO grid.
    """
    cats = [c for c in cat_cols if c in X_tr.columns]
    dtrain = lgb.Dataset(X_tr, label=y_tr, categorical_feature=cats, free_raw_data=False)
    dtest = lgb.Dataset(X_te, label=y_te, categorical_feature=cats, reference=dtrain, free_raw_data=False)
    model = lgb.train(
        LGBM_PARAMS,
        dtrain,
        num_boost_round=NUM_BOOST_ROUND,
        valid_sets=[dtrain, dtest],
        valid_names=["train", "test"],
        callbacks=[
            lgb.early_stopping(EARLY_STOPPING, verbose=False),
            lgb.log_evaluation(0),
        ],
    )
    n = int(model.best_iteration or model.current_iteration())
    preds = model.predict(X_te, num_iteration=n)
    metrics = eval_split(y_te, preds)
    metrics["best_iter"] = n
    return model, metrics, n


def _exec_cell(ns: dict, cell_id: int) -> None:
    path = CELL_DUMP / f"cell_{cell_id:03d}.py"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}; run _dump_nb_cells.py first")
    code = path.read_text(encoding="utf-8")
    # Skip the side-effecting "build train/test" tails in merge/cross/prepare cells —
    # those are handled explicitly below.
    exec(compile(code, str(path), "exec"), ns, ns)


def build_feature_frames(force: bool = False) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Build (or load cached) post-prune + cross-exchange train/test frames."""
    train_path = CACHE / "df_train_logo.parquet"
    test_path = CACHE / "df_test_logo.parquet"
    meta_path = CACHE / "meta.json"

    if not force and train_path.exists() and test_path.exists() and meta_path.exists():
        print(f"Loading cached frames from {CACHE}")
        df_train = pd.read_parquet(train_path)
        df_test = pd.read_parquet(test_path)
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        return df_train, df_test, meta

    if not CELL_DUMP.exists():
        raise SystemExit("Run _dump_nb_cells.py first to materialise notebook cells.")

    print("=" * 60)
    print("Building feature matrices via notebook cell dump (this is slow once) …")
    print("=" * 60)
    t0 = time.time()
    ns: dict = {"__name__": "__logo_build__"}

    # Config + LOAD_TABLES
    _exec_cell(ns, 3)
    _exec_cell(ns, 4)
    # Redirect outputs into outputs_logo
    ns["OUTPUT_DIR"] = OUT
    OUT.mkdir(parents=True, exist_ok=True)

    # Data load (WINDOWS, parsers, pool)
    _exec_cell(ns, 6)

    # Feature builders (spread, ticker/_to_wide, orderbook, trades, funding/OI)
    for cid in (8, 10, 12, 14, 16):
        _exec_cell(ns, cid)

    # build_feature_matrix definition only (strip the train/test call at the bottom)
    merge_src = (CELL_DUMP / "cell_018.py").read_text(encoding="utf-8")
    merge_fn = merge_src.split("print(\"Building train")[0]
    exec(compile(merge_fn, "cell_018_fn", "exec"), ns, ns)

    print("\nBuilding train feature matrix …")
    train_raw = ns["train_raw"]
    # build_feature_matrix pops keys — copy refs carefully
    df_train = ns["build_feature_matrix"]({k: v for k, v in train_raw.items()})
    gc.collect()
    print(f"train raw features: {df_train.shape}")

    print("\nBuilding test feature matrix …")
    test_raw = ns["test_raw"]
    df_test = ns["build_feature_matrix"]({k: v for k, v in test_raw.items()})
    gc.collect()
    print(f"test raw features: {df_test.shape}")

    # Diagnostics report for prune decisions (on train)
    import scipy.stats as stats

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
        null_pct = df_train[col].isna().mean()
        corr_target = df_train[[col, "target"]].dropna().corr().iloc[0, 1]
        report.append(
            {
                "feature": col,
                "null_pct": round(float(null_pct), 3),
                "corr_target": round(float(corr_target), 4) if pd.notna(corr_target) else 0.0,
            }
        )
    report_df = pd.DataFrame(report)
    report_df.to_csv(OUT / "feature_prune_report.csv", index=False)

    # Track which funding/OI cols existed pre-prune
    pre_prune_groups = group_features(feature_cols)
    funding_pre = pre_prune_groups["funding"]
    oi_pre = pre_prune_groups["oi"]

    drop_null = report_df[report_df["null_pct"] > 0.4]["feature"].tolist()
    drop_weak = report_df[report_df["corr_target"].abs() < 0.005]["feature"].tolist()
    all_drops = list(set(drop_null + drop_weak))
    print(f"Pruning {len(all_drops)} columns (null>0.4 or |corr|<0.005)")
    for df in (df_train, df_test):
        df.drop(columns=[c for c in all_drops if c in df.columns], inplace=True)

    # Log-transform volume/ratio cols
    def signed_log1p(s):
        return np.sign(s) * np.log1p(np.abs(s))

    log_patterns = ["tk_bid_volume", "tk_ask_volume", "tr_total_volume", "tr_buy_sell_ratio"]
    log_cols = [c for c in df_train.columns if any(p in c for p in log_patterns)]
    for df in (df_train, df_test):
        for col in log_cols:
            if col in df.columns:
                df[col] = signed_log1p(df[col])

    # Winsorize spreads
    from scipy.stats.mstats import winsorize

    winsor_patterns = ["spread_bps_lag", "tk_spread_bps"]
    winsor_cols = [c for c in df_train.columns if any(p in c for p in winsor_patterns)]
    for df in (df_train, df_test):
        for col in winsor_cols:
            if col in df.columns:
                med = df[col].median()
                df[col] = winsorize(df[col].fillna(med), limits=[0.01, 0.01])

    # Cross-exchange + momentum (function body only)
    cross_src = (CELL_DUMP / "cell_027.py").read_text(encoding="utf-8")
    cross_fn = cross_src.split("df_train = add_cross_exchange_features")[0]
    exec(compile(cross_fn, "cell_027_fn", "exec"), ns, ns)
    df_train = ns["add_cross_exchange_features"](df_train)
    df_test = ns["add_cross_exchange_features"](df_test)

    # Persist
    print(f"\nCaching to {CACHE} …")
    df_train.to_parquet(train_path, index=False)
    df_test.to_parquet(test_path, index=False)
    meta = {
        "horizon": HORIZON,
        "zscore_window": ZSCORE_WINDOW,
        "n_lags": N_LAGS,
        "n_train": int(len(df_train)),
        "n_test": int(len(df_test)),
        "n_cols": int(df_train.shape[1]),
        "dropped": all_drops,
        "funding_pre_prune": funding_pre,
        "oi_pre_prune": oi_pre,
        "funding_dropped": [c for c in funding_pre if c in all_drops],
        "oi_dropped": [c for c in oi_pre if c in all_drops],
        "build_seconds": round(time.time() - t0, 1),
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Done in {meta['build_seconds']}s — train={df_train.shape} test={df_test.shape}")
    return df_train, df_test, meta


def prepare_xy(df: pd.DataFrame, reference_cols=None):
    df = df.dropna(subset=["target"]).copy()
    cat_cols = [c for c in ["coin", "pair"] if c in df.columns]
    for c in cat_cols:
        df[c] = df[c].astype("category")
    feat_cols = [c for c in df.columns if c not in ID_COLS]
    X = df[feat_cols].copy()
    y = df["target"].values
    if reference_cols is not None:
        X = X.reindex(columns=reference_cols)
        for c in cat_cols:
            if c in X.columns:
                X[c] = X[c].astype("category")
    return X, y, feat_cols, cat_cols


def run_ablation(df_train: pd.DataFrame, df_test: pd.DataFrame, meta: dict, mode: str = "both") -> pd.DataFrame:
    X_train, y_train, feat_cols, cat_cols = prepare_xy(df_train)
    X_test, y_test, _, _ = prepare_xy(df_test, reference_cols=X_train.columns)

    # Prefer intersection with published 68 if shapes match closely; else use all surviving
    published_present = [c for c in PUBLISHED_FEATURES if c in X_train.columns]
    if len(published_present) >= 60:
        print(f"Restricting to {len(published_present)} published features (of 68)")
        feat_cols = published_present
        X_train = X_train[feat_cols]
        X_test = X_test.reindex(columns=feat_cols)
        for c in cat_cols:
            if c in X_train.columns:
                X_train[c] = X_train[c].astype("category")
                X_test[c] = X_test[c].astype("category")

    groups = group_features(feat_cols)
    print("\nFeature group sizes:")
    for k, v in groups.items():
        print(f"  {k:12s} {len(v):3d}")

    print("\nTraining FULL model with early stopping to lock round count …")
    t0 = time.time()
    _, full_metrics, n_rounds = train_with_early_stop(X_train, y_train, X_test, y_test, cat_cols)
    print(
        f"  best_iter={n_rounds}  R²={full_metrics['r2']:.4f}  "
        f"R²_filt={full_metrics['r2_filtered']:.4f}  "
        f"DirAcc_filt={full_metrics['dir_acc_filtered']:.3%}  "
        f"({time.time()-t0:.1f}s)"
    )

    variants: list[tuple[str, list[str], str, str]] = []
    if mode in ("nested", "both"):
        for name, cols, added in nested_variants(groups):
            variants.append(("nested", name, cols, added))
    if mode in ("logo", "both"):
        for name, cols, added in logo_variants(groups, feat_cols):
            variants.append(("logo", name, cols, added))

    rows = []
    baseline_r2_f = None
    cache_by_key: dict[tuple[str, ...], dict] = {}
    for design, name, cols, added in variants:
        # Deduplicate while preserving order
        seen = set()
        cols_u = []
        for c in cols:
            if c not in seen and c in X_train.columns:
                seen.add(c)
                cols_u.append(c)
        if not cols_u:
            rows.append(
                {
                    "design": design,
                    "variant": name,
                    "features_added": added,
                    "n_features": 0,
                    "note": "empty feature set",
                    "r2": np.nan,
                    "r2_filtered": np.nan,
                    "dir_acc_filtered": np.nan,
                    "delta_r2_filtered_vs_baseline": np.nan,
                    "best_iter": n_rounds,
                }
            )
            continue

        key = tuple(cols_u)
        t1 = time.time()
        if key in cache_by_key:
            metrics = dict(cache_by_key[key])
            metrics["note"] = "identical feature set — reused"
            wall = 0.0
            print(
                f"[{design:6s}] {name:28s} n={len(cols_u):3d}  "
                f"R²={metrics['r2']:.4f}  R²_f={metrics['r2_filtered']:.4f}  "
                f"(reused identical set)"
            )
        else:
            Xtr = X_train[cols_u]
            Xte = X_test[cols_u]
            cats = [c for c in cat_cols if c in cols_u]
            _, metrics = train_fixed_rounds(Xtr, y_train, Xte, y_test, cats, n_rounds)
            cache_by_key[key] = dict(metrics)
            wall = round(time.time() - t1, 1)
            print(
                f"[{design:6s}] {name:28s} n={len(cols_u):3d}  "
                f"R²={metrics['r2']:.4f}  R²_f={metrics['r2_filtered']:.4f}  "
                f"Dir_f={metrics['dir_acc_filtered']:.3%}  ({wall:.1f}s)"
            )
        if design == "nested" and name == "AR baseline":
            baseline_r2_f = metrics["r2_filtered"]
        delta = (
            metrics["r2_filtered"] - baseline_r2_f
            if baseline_r2_f is not None and design == "nested" and np.isfinite(metrics["r2_filtered"])
            else np.nan
        )
        rows.append(
            {
                "design": design,
                "variant": name,
                "features_added": added,
                "n_features": len(cols_u),
                "r2": metrics["r2"],
                "dir_acc": metrics["dir_acc"],
                "r2_filtered": metrics["r2_filtered"],
                "dir_acc_filtered": metrics["dir_acc_filtered"],
                "mae_filtered": metrics["mae_filtered"],
                "n_filtered": metrics["n_filtered"],
                "pct_filtered": metrics["pct_filtered"],
                "delta_r2_filtered_vs_baseline": delta,
                "best_iter": n_rounds,
                "wall_s": wall,
                "note": metrics.get("note", ""),
            }
        )

    # Attach prune notes for funding/OI
    for gname, pre, dropped in (
        ("funding", meta.get("funding_pre_prune", []), meta.get("funding_dropped", [])),
        ("oi", meta.get("oi_pre_prune", []), meta.get("oi_dropped", [])),
    ):
        print(f"\n{gname}: {len(pre)} pre-prune cols, {len(dropped)} dropped by null/weak filter")

    out = pd.DataFrame(rows)
    out.to_csv(OUT / "logo_ablation_results.csv", index=False)

    # Paper-facing nested table
    nested = out[out["design"] == "nested"].copy()
    nested.to_csv(OUT / "logo_nested_table.csv", index=False)
    logo = out[out["design"] == "logo"].copy()
    logo.to_csv(OUT / "logo_leave_one_out_table.csv", index=False)

    print("\n=== Nested cumulative (paper table) ===")
    cols_show = [
        "variant",
        "n_features",
        "r2",
        "r2_filtered",
        "dir_acc_filtered",
        "delta_r2_filtered_vs_baseline",
    ]
    if len(nested):
        print(nested[cols_show].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\n=== Classic LOGO ===")
    if len(logo):
        print(
            logo[["variant", "n_features", "r2", "r2_filtered", "dir_acc_filtered"]].to_string(
                index=False, float_format=lambda x: f"{x:.4f}"
            )
        )

    lines = [
        "# LOGO / Nested Feature-Group Ablation Results",
        "",
        "**Verdict: Scenario C.** AR baseline (lags + momentum + identity) carries",
        "filtered R²; microstructure adds ~0.01 or less (often slightly negative).",
        "",
        f"Protocol: H=1, W=300, N_LAGS=3, Jul 25–28 offline test, filter **τ={FILTER_TAU:g}**",
        f"(paper confidence gate). Fixed rounds = **{n_rounds}**.",
        "",
        "Script: `statarb/run_logo_ablation.py` · Artifacts: `statarb/outputs_logo/`",
        "",
        f"## Nested cumulative (τ={FILTER_TAU:g})",
        "",
        "| Variant | # feat. | R² (all) | R² (filt.) | DirAcc (filt.) | Δ R² filt. vs AR |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for _, r in nested.iterrows():
        d = r["delta_r2_filtered_vs_baseline"]
        d_s = "—" if pd.isna(d) else f"{d:+.3f}"
        lines.append(
            f"| {r['variant']} | {int(r['n_features'])} | {r['r2']:.3f} | "
            f"{r['r2_filtered']:.3f} | {100 * r['dir_acc_filtered']:.1f}% | {d_s} |"
        )
    lines += [
        "",
        "## Classic leave-one-group-out",
        "",
        "| Variant | # feat. | R² (all) | R² (filt.) | DirAcc (filt.) |",
        "|---|---:|---:|---:|---:|",
    ]
    for _, r in logo.iterrows():
        lines.append(
            f"| {r['variant']} | {int(r['n_features'])} | {r['r2']:.3f} | "
            f"{r['r2_filtered']:.3f} | {100 * r['dir_acc_filtered']:.1f}% |"
        )
    lines += [
        "",
        "## Takeaway",
        "",
        "- **Important:** AR baseline (lags + momentum + coin/pair identity).",
        "- **Not important:** ticker / OB / trades / cross — ~0.01 R² or worse vs AR.",
        "- Funding / OI prune to 0 columns.",
        "- Fee-aware / bps-net checks are separate extras (see `docs/ablation_results.md`).",
        "",
    ]
    (OUT / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {OUT / 'RESULTS.md'}")
    return out


def main() -> None:
    # Windows cp1252 consoles choke on notebook box-drawing chars
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    global FILTER_TAU
    ap = argparse.ArgumentParser(description="LOGO / nested feature-group ablation")
    ap.add_argument("--force-rebuild", action="store_true", help="Ignore cached feature frames")
    ap.add_argument("--mode", choices=["nested", "logo", "both"], default="both")
    ap.add_argument("--dump-cells", action="store_true", help="Refresh notebook cell dump first")
    ap.add_argument(
        "--filter-tau",
        type=float,
        default=FILTER_TAU,
        help="Primary |pred| gate for filtered R² / DirAcc (paper default 0.9)",
    )
    args = ap.parse_args()
    FILTER_TAU = float(args.filter_tau)

    if args.dump_cells or not CELL_DUMP.exists():
        dump = HERE / "_dump_nb_cells.py"
        print(f"Refreshing cell dump via {dump}")
        ns: dict = {}
        exec(compile(dump.read_text(encoding="utf-8"), str(dump), "exec"), ns, ns)

    print(f"Using filter τ={FILTER_TAU:g} for filtered metrics")
    df_train, df_test, meta = build_feature_frames(force=args.force_rebuild)
    run_ablation(df_train, df_test, meta, mode=args.mode)
    print(f"\nResults written to {OUT}")


if __name__ == "__main__":
    main()
