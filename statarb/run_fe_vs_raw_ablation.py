"""Ablation: pre-training feature engineering vs raw market features.

Compares predicting z_{t+1} (H=1) with:
  - raw live microstructure (ticker / OB / trades / cross) + coin/pair
  - engineered spread trajectory (z/spread lags + momentum/accel) + coin/pair
  - full set

coin/pair are held fixed on both sides (identity controls, not the FE question).

Uses cached LOGO frames + fixed rounds from early-stopped full model.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

OUT = Path(__file__).resolve().parent / "outputs_logo"
CACHE = OUT / "cache"
FILTER_TAU = 0.9

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

ID_SKIP = {
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

ENGINEERED = {
    "spread_bps_lag1",
    "spread_bps_lag2",
    "spread_bps_lag3",
    "zscore_lag1",
    "zscore_lag2",
    "zscore_lag3",
    "spread_momentum_1_3",
    "zscore_momentum_1_3",
    "zscore_accel",
}
IDENTITY = {"coin", "pair"}


def classify(cols: list[str]) -> dict[str, list[str]]:
    eng, raw, ident, other = [], [], [], []
    for c in cols:
        if c in IDENTITY:
            ident.append(c)
        elif c in ENGINEERED:
            eng.append(c)
        elif c.startswith(("tk_", "ob_", "tr_", "cross_", "price_rank_")) or c in {
            "tightest_ba",
            "widest_ba",
            "ba_range",
            "net_ob_pressure",
        }:
            raw.append(c)
        else:
            other.append(c)
    return {"identity": ident, "engineered": eng, "raw": raw, "other": other}


def eval_split(y, preds):
    r2 = float(r2_score(y, preds))
    dir_acc = float(np.mean(np.sign(preds) == np.sign(y)))
    mask = np.abs(preds) >= FILTER_TAU
    n_filt = int(mask.sum())
    if n_filt:
        yf, pf = y[mask], preds[mask]
        r2_f = float(r2_score(yf, pf))
        dir_f = float(np.mean(np.sign(pf) == np.sign(yf)))
        mae_f = float(mean_absolute_error(yf, pf))
    else:
        r2_f = dir_f = mae_f = float("nan")
    return {
        "r2": r2,
        "dir_acc": dir_acc,
        "mae": float(mean_absolute_error(y, preds)),
        "rmse": float(np.sqrt(mean_squared_error(y, preds))),
        "n_filtered": n_filt,
        "r2_filtered": r2_f,
        "dir_acc_filtered": dir_f,
        "mae_filtered": mae_f,
    }


def prepare(df: pd.DataFrame):
    d = df.dropna(subset=["target"]).copy()
    for c in ("coin", "pair"):
        if c in d.columns:
            d[c] = d[c].astype("category")
    feat = [c for c in d.columns if c not in ID_SKIP]
    return d, feat


def train_early(Xtr, ytr, Xte, yte, cats):
    dtrain = lgb.Dataset(Xtr, label=ytr, categorical_feature=cats, free_raw_data=False)
    dtest = lgb.Dataset(Xte, label=yte, categorical_feature=cats, reference=dtrain, free_raw_data=False)
    model = lgb.train(
        LGBM_PARAMS,
        dtrain,
        num_boost_round=NUM_BOOST_ROUND,
        valid_sets=[dtrain, dtest],
        valid_names=["train", "test"],
        callbacks=[lgb.early_stopping(EARLY_STOPPING, verbose=False), lgb.log_evaluation(0)],
    )
    n = int(model.best_iteration or model.current_iteration())
    preds = model.predict(Xte, num_iteration=n)
    return model, eval_split(yte, preds), n


def train_fixed(Xtr, ytr, Xte, yte, cats, n_rounds):
    dtrain = lgb.Dataset(Xtr, label=ytr, categorical_feature=cats, free_raw_data=False)
    model = lgb.train(LGBM_PARAMS, dtrain, num_boost_round=n_rounds, callbacks=[lgb.log_evaluation(0)])
    preds = model.predict(Xte, num_iteration=n_rounds)
    return model, eval_split(yte, preds)


def main():
    tr = pd.read_parquet(CACHE / "df_train_logo.parquet")
    te = pd.read_parquet(CACHE / "df_test_logo.parquet")
    dtr, feat_cols = prepare(tr)
    dte, _ = prepare(te)
    groups = classify(feat_cols)
    print("group sizes:", {k: len(v) for k, v in groups.items()})
    if groups["other"]:
        print("WARNING other cols:", groups["other"])

    ytr = dtr["target"].to_numpy()
    yte = dte["target"].to_numpy()

    variants = [
        (
            "raw + identity",
            groups["raw"] + groups["identity"],
            "ticker/OB/trades/cross + coin/pair; no z/spread trajectory FE",
        ),
        (
            "engineered + identity",
            groups["engineered"] + groups["identity"],
            "z/spread lags + momentum/accel + coin/pair",
        ),
        (
            "engineered only",
            list(groups["engineered"]),
            "trajectory FE without coin/pair",
        ),
        (
            "raw only",
            list(groups["raw"]),
            "raw microstructure without coin/pair",
        ),
        (
            "full",
            feat_cols,
            "all surviving features",
        ),
    ]

    # Lock rounds on full model
    print("Early-stop full model to lock rounds …")
    cats_full = [c for c in ("coin", "pair") if c in feat_cols]
    Xtr_f = dtr[feat_cols]
    Xte_f = dte.reindex(columns=feat_cols)
    for c in cats_full:
        Xte_f[c] = Xte_f[c].astype("category")
    _, full_m, n_rounds = train_early(Xtr_f, ytr, Xte_f, yte, cats_full)
    print(f"  n_rounds={n_rounds}  R2_f={full_m['r2_filtered']:.4f}  Dir_f={full_m['dir_acc_filtered']:.3%}")

    rows = []
    for name, cols, note in variants:
        cols_u = [c for c in cols if c in dtr.columns]
        cats = [c for c in ("coin", "pair") if c in cols_u]
        Xtr = dtr[cols_u].copy()
        Xte = dte.reindex(columns=cols_u).copy()
        for c in cats:
            Xtr[c] = Xtr[c].astype("category")
            Xte[c] = Xte[c].astype("category")
        t0 = time.time()
        _, m = train_fixed(Xtr, ytr, Xte, yte, cats, n_rounds)
        wall = time.time() - t0
        print(
            f"{name:28s} n={len(cols_u):3d}  R2={m['r2']:.4f}  "
            f"R2_f={m['r2_filtered']:.4f}  Dir_f={m['dir_acc_filtered']:.3%}  ({wall:.1f}s)"
        )
        rows.append(
            {
                "variant": name,
                "n_features": len(cols_u),
                "note": note,
                "best_iter": n_rounds,
                **m,
            }
        )

    out = pd.DataFrame(rows)
    out_path = OUT / "fe_vs_raw_ablation.csv"
    out.to_csv(out_path, index=False)

    # Markdown summary
    md = [
        "# Feature-engineering vs raw market features",
        "",
        f"Protocol: H=1, Jul 25–28 offline test, filter τ={FILTER_TAU:g}, fixed rounds = **{n_rounds}**.",
        "",
        "Identity (`coin`, `pair`) is an entity control, not the FE question:",
        "- `coin`: which asset",
        "- `pair`: which two venues the cross-exchange mean-reversion lives on",
        "",
        "| Variant | # | R² | R²_f | DirAcc_f |",
        "|---|---:|---:|---:|---:|",
    ]
    for r in rows:
        md.append(
            f"| {r['variant']} | {r['n_features']} | {r['r2']:.3f} | "
            f"{r['r2_filtered']:.3f} | {100*r['dir_acc_filtered']:.1f}% |"
        )
    (OUT / "FE_VS_RAW_RESULTS.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("wrote", out_path)
    print("wrote", OUT / "FE_VS_RAW_RESULTS.md")


if __name__ == "__main__":
    main()
