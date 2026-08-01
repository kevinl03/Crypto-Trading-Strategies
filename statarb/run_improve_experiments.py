"""Model-improvement experiments from the analysis plan.

1) Protocol fix — chronological split, early-stop on val (not test),
   report jul19_pre / jul19_post / jul22_28 separately.
2) Objective + HORIZON sweep — zscore_fwd vs zscore_delta, DirAcc, threshold PnL proxy.
3) Gated LSR + liquidations ablation on jul22-28 only.

OHLCV remains deferred (incomplete coverage) — see outputs_improve/DEFER_OHLCV.md.
"""
from __future__ import annotations

import json
import gc
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)

LOCAL_DATA_ROOT = Path(r"C:/Users/Kev/repos/stochastic-spread-modeling/data/cex_unified")
OUT = Path("./outputs_improve")
OUT.mkdir(parents=True, exist_ok=True)

TOP_EXCHANGES = ["binance", "bybit", "okx", "coinbase", "kraken", "mexc"]
ZSCORE_WINDOW = 120
N_LAGS = 2
MIN_PERIODS = 30

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
LGBM_CLS = {
    **LGBM_REG,
    "objective": "binary",
    "metric": ["binary_logloss", "auc"],
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


def _coin_from_symbol(symbol: str | None) -> str | None:
    if not symbol or not isinstance(symbol, str):
        return None
    base = symbol.split("/")[0]
    return base or None


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
    print(f"    -> {out.shape}")
    return out


def load_aux_signal(window_id: str, table: str) -> pd.DataFrame:
    """Load long_short_ratio or liquidations (jul22-28 only in unified cache)."""
    rel = WINDOWS[window_id]
    path = LOCAL_DATA_ROOT / rel / f"{table}.parquet"
    if not path.exists():
        return pd.DataFrame()
    print(f"  {table} [{window_id}] …", flush=True)
    df = pd.read_parquet(path)
    if "error" in df.columns:
        df = df[df["error"].isna()].drop(columns=["error"])
    records = []
    for row in df.itertuples():
        try:
            p = json.loads(row.payload) if isinstance(row.payload, str) else row.payload
            raw_coin = getattr(row, "coin", None)
            if raw_coin is None or (isinstance(raw_coin, float) and np.isnan(raw_coin)) or raw_coin == "":
                coin = _coin_from_symbol(getattr(row, "symbol", None) or p.get("symbol"))
            else:
                coin = str(raw_coin)
            exch = getattr(row, "exchange", None) or p.get("exchange")
            if exch is None or (isinstance(exch, float) and np.isnan(exch)):
                continue
            exch = str(exch)
            if not coin:
                continue
            if table == "long_short_ratio":
                records.append((row.snapshot_idx, coin, exch, float(p.get("long_short_ratio") or 0)))
            else:
                side = 1.0 if str(p.get("side", "")).lower() in ("buy", "long") else -1.0
                qv = float(p.get("quote_value") or 0)
                records.append((row.snapshot_idx, coin, exch, side * qv, abs(qv)))
        except Exception:
            continue
    del df
    gc.collect()
    if table == "long_short_ratio":
        out = pd.DataFrame(records, columns=["snapshot_idx", "coin", "exchange", "long_short_ratio"])
    else:
        out = pd.DataFrame(records, columns=["snapshot_idx", "coin", "exchange", "liq_signed_quote", "liq_abs_quote"])
        # aggregate multiple liquidations per snapshot/coin/exchange
        out = (
            out.groupby(["snapshot_idx", "coin", "exchange"], as_index=False)
            .agg(liq_signed_quote=("liq_signed_quote", "sum"), liq_abs_quote=("liq_abs_quote", "sum"))
        )
    out["window_id"] = window_id
    print(f"    -> {out.shape}")
    return out


def build_spread_features(sm: pd.DataFrame, horizon: int, target_kind: str) -> pd.DataFrame:
    sm = sm.copy()
    sm["pair"] = sm["exchange_a"] + "__" + sm["exchange_b"]
    sm = sm.sort_values(["window_id", "coin", "pair", "snapshot_idx"]).reset_index(drop=True)
    keys = ["window_id", "coin", "pair"]
    grp = sm.groupby(keys)["spread_bps"]
    roll_mean = grp.transform(lambda x: x.rolling(ZSCORE_WINDOW, min_periods=MIN_PERIODS).mean())
    roll_std = grp.transform(lambda x: x.rolling(ZSCORE_WINDOW, min_periods=MIN_PERIODS).std())
    sm["zscore"] = (sm["spread_bps"] - roll_mean) / roll_std.replace(0, np.nan)
    z_fwd = sm.groupby(keys)["zscore"].transform(lambda x: x.shift(-horizon))
    if target_kind == "zscore_fwd":
        sm["target"] = z_fwd
    elif target_kind == "zscore_delta":
        sm["target"] = z_fwd - sm["zscore"]
    else:
        raise ValueError(target_kind)
    for lag in range(1, N_LAGS + 1):
        sm[f"spread_bps_lag{lag}"] = grp.transform(lambda x, l=lag: x.shift(l))
        sm[f"zscore_lag{lag}"] = sm.groupby(keys)["zscore"].transform(lambda x, l=lag: x.shift(l))
    return sm


def merge_aux(base: pd.DataFrame, aux: pd.DataFrame, value_cols: list[str], prefix: str) -> pd.DataFrame:
    if aux.empty:
        return base
    wide = aux.pivot_table(
        index=["window_id", "snapshot_idx", "coin"],
        columns="exchange",
        values=value_cols,
        aggfunc="last",
    )
    wide.columns = [f"{prefix}_{c[0]}_{c[1]}" if isinstance(c, tuple) else f"{prefix}_{c}" for c in wide.columns]
    wide = wide.reset_index()
    return base.merge(wide, on=["window_id", "snapshot_idx", "coin"], how="left")


def feature_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    df = df.dropna(subset=["target", "zscore_lag1"]).copy()
    id_cols = {
        "snapshot_idx", "exchange_a", "exchange_b", "spread_bps", "zscore", "target",
        "p1", "p2", "window_id", "pair", "split",
    }
    # keep coin as categorical; drop pair to reduce cardinality for sweeps
    feat_cols = [c for c in df.columns if c not in id_cols]
    X = df[feat_cols].copy()
    if "coin" in X.columns:
        X["coin"] = X["coin"].astype("category")
    y = df["target"].values
    return X, y


def eval_reg(y, preds, label: str) -> dict:
    mae = mean_absolute_error(y, preds)
    rmse = float(np.sqrt(mean_squared_error(y, preds)))
    r2 = r2_score(y, preds)
    dir_acc = float(np.mean(np.sign(preds) == np.sign(y)))
    # thresholded directional hit when |pred| > 0.5
    mask = np.abs(preds) > 0.5
    dir_acc_tau = float(np.mean(np.sign(preds[mask]) == np.sign(y[mask]))) if mask.any() else float("nan")
    # crude PnL proxy: trade sign(pred) * realized delta (or level change proxy = y for delta target)
    pnl = float(np.mean(np.sign(preds) * y))
    print(
        f"{label:40s} R²={r2:.4f} DirAcc={dir_acc:.3%} "
        f"DirAcc@|p|>0.5={dir_acc_tau:.3%} PnLproxy={pnl:.4f}"
    )
    return {
        "label": label,
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "dir_acc": dir_acc,
        "dir_acc_tau0.5": dir_acc_tau,
        "pnl_proxy": pnl,
        "n": int(len(y)),
    }


def train_reg(X_tr, y_tr, X_va, y_va, cat_cols: list[str]):
    dtr = lgb.Dataset(X_tr, label=y_tr, categorical_feature=cat_cols or "auto")
    dva = lgb.Dataset(X_va, label=y_va, reference=dtr, categorical_feature=cat_cols or "auto")
    model = lgb.train(
        LGBM_REG,
        dtr,
        num_boost_round=NUM_BOOST,
        valid_sets=[dva],
        valid_names=["val"],
        callbacks=[lgb.early_stopping(PATIENCE, verbose=False), lgb.log_evaluation(0)],
    )
    return model


def train_cls(X_tr, y_tr, X_va, y_va, cat_cols: list[str]):
    # classify sign of target (drop zeros)
    def prep(X, y):
        m = y != 0
        return X.loc[m].reset_index(drop=True), (y[m] > 0).astype(int)

    Xtr, ytr = prep(X_tr, y_tr)
    Xva, yva = prep(X_va, y_va)
    dtr = lgb.Dataset(Xtr, label=ytr, categorical_feature=cat_cols or "auto")
    dva = lgb.Dataset(Xva, label=yva, reference=dtr, categorical_feature=cat_cols or "auto")
    model = lgb.train(
        LGBM_CLS,
        dtr,
        num_boost_round=NUM_BOOST,
        valid_sets=[dva],
        valid_names=["val"],
        callbacks=[lgb.early_stopping(PATIENCE, verbose=False), lgb.log_evaluation(0)],
    )
    return model, Xva, yva


# ── 1) Protocol fix ───────────────────────────────────────────────────────────

def run_protocol() -> pd.DataFrame:
    print("\n=== PROTOCOL: chronological split ===")
    # Train before Jul19; val=jul19_pre (early stop); tests=jul19_post + jul22_28 forward
    train_ids = ["jun13", "jun22", "jul13"]
    val_id = "jul19_pre"
    test_ids = ["jul19_post", "jul22_28"]

    frames = {wid: load_spread(wid) for wid in train_ids + [val_id] + test_ids}
    sm_all = pd.concat(frames.values(), ignore_index=True)
    feat = build_spread_features(sm_all, horizon=2, target_kind="zscore_fwd")

    def split(ids):
        return feat[feat["window_id"].isin(ids if isinstance(ids, list) else [ids])]

    X_tr, y_tr = feature_frame(split(train_ids))
    X_va, y_va = feature_frame(split(val_id))
    # align columns
    X_va = X_va.reindex(columns=X_tr.columns)
    cat = ["coin"] if "coin" in X_tr.columns else []

    model = train_reg(X_tr, y_tr, X_va, y_va, cat)
    print(f"best_iteration (val={val_id}): {model.best_iteration}")

    rows = []
    rows.append(eval_reg(y_tr, model.predict(X_tr, num_iteration=model.best_iteration), "train"))
    rows.append(eval_reg(y_va, model.predict(X_va, num_iteration=model.best_iteration), "val_jul19_pre"))
    for tid in test_ids:
        Xt, yt = feature_frame(split(tid))
        Xt = Xt.reindex(columns=X_tr.columns)
        rows.append(eval_reg(yt, model.predict(Xt, num_iteration=model.best_iteration), f"test_{tid}"))
        # naive persistence DirAcc baseline on same split
        if "zscore_lag1" in Xt.columns:
            naive = Xt["zscore_lag1"].astype(float).values
            rows.append(eval_reg(yt, naive, f"naive_{tid}"))

    # contrast: old inverted protocol (jul22 in train, early-stop on pooled jul19)
    print("\n--- contrast: old inverted protocol (jul22_28 in TRAIN, early-stop on jul19_post) ---")
    X_tr_old, y_tr_old = feature_frame(split(train_ids + ["jul22_28"]))
    X_va_old, y_va_old = feature_frame(split("jul19_post"))
    X_va_old = X_va_old.reindex(columns=X_tr_old.columns)
    model_old = train_reg(X_tr_old, y_tr_old, X_va_old, y_va_old, cat)
    Xt_pre, yt_pre = feature_frame(split("jul19_pre"))
    Xt_pre = Xt_pre.reindex(columns=X_tr_old.columns)
    rows.append(
        eval_reg(
            yt_pre,
            model_old.predict(Xt_pre, num_iteration=model_old.best_iteration),
            "OLDPROTO_test_jul19_pre",
        )
    )
    rows.append(
        eval_reg(
            y_va_old,
            model_old.predict(X_va_old, num_iteration=model_old.best_iteration),
            "OLDPROTO_earlystop_jul19_post",
        )
    )

    out = pd.DataFrame(rows)
    out.to_csv(OUT / "protocol_eval.csv", index=False)
    print(f"saved {OUT / 'protocol_eval.csv'}")
    return out


# ── 1b) Validation choice / ordering sensitivity ──────────────────────────────

def run_val_rotation() -> pd.DataFrame:
    """Does the choice (and ordering) of the early-stopping validation set change accuracy?

    Jul22-28 is held fixed as an untouched forward yardstick in every config, so
    differences in forward metrics are attributable to the validation choice alone.
    """
    print("\n=== VALIDATION ROTATION / ORDER SENSITIVITY ===")
    all_ids = ["jun13", "jun22", "jul13", "jul19_pre", "jul19_post", "jul22_28"]
    frames = {wid: load_spread(wid) for wid in all_ids}
    feat = build_spread_features(pd.concat(frames.values(), ignore_index=True), horizon=2, target_kind="zscore_fwd")

    def win(ids):
        return feat[feat["window_id"].isin(ids if isinstance(ids, list) else [ids])]

    # jul13 chronological tail (for a val slice carved out of train)
    jul13 = win("jul13")
    j13_snaps = np.sort(jul13["snapshot_idx"].unique())
    j13_cut = j13_snaps[int(len(j13_snaps) * 0.80)]

    # pooled jul19 for chronological-vs-random comparisons
    jul19 = win(["jul19_pre", "jul19_post"])
    j19_snaps = np.sort(jul19["snapshot_idx"].unique())

    rng = np.random.default_rng(42)

    configs: list[dict] = [
        {
            "name": "A_val=jul19_pre_test=jul19_post",
            "train": win(["jun13", "jun22", "jul13"]),
            "val": win("jul19_pre"),
            "test": win("jul19_post"),
            "note": "current protocol",
        },
        {
            "name": "B_val=jul19_post_test=jul19_pre",
            "train": win(["jun13", "jun22", "jul13"]),
            "val": win("jul19_post"),
            "test": win("jul19_pre"),
            "note": "swapped val/test order",
        },
        {
            "name": "C_val=jul13tail_test=jul19_all",
            "train": pd.concat([win(["jun13", "jun22"]), jul13[jul13["snapshot_idx"] <= j13_cut]]),
            "val": jul13[jul13["snapshot_idx"] > j13_cut],
            "test": jul19,
            "note": "val carved from train tail; jul19 fully held out",
        },
        {
            "name": "D_val=jul19_random20_test=jul19_rest",
            "train": win(["jun13", "jun22", "jul13"]),
            "val": None,  # filled below (random rows)
            "test": None,
            "note": "NON-chronological random val slice inside jul19",
        },
        {
            "name": "E_val=jul19_first20_test=jul19_rest",
            "train": win(["jun13", "jun22", "jul13"]),
            "val": jul19[jul19["snapshot_idx"] <= j19_snaps[int(len(j19_snaps) * 0.20)]],
            "test": jul19[jul19["snapshot_idx"] > j19_snaps[int(len(j19_snaps) * 0.20)]],
            "note": "chronological first-20% val inside jul19",
        },
    ]

    # config D random split by rows
    mask = rng.random(len(jul19)) < 0.20
    configs[3]["val"] = jul19[mask]
    configs[3]["test"] = jul19[~mask]

    forward_df = win("jul22_28")
    rows = []

    for cfg in configs:
        print(f"\n-- {cfg['name']}  ({cfg['note']}) --")
        X_tr, y_tr = feature_frame(cfg["train"])
        X_va, y_va = feature_frame(cfg["val"])
        X_te, y_te = feature_frame(cfg["test"])
        X_fw, y_fw = feature_frame(forward_df)
        X_va = X_va.reindex(columns=X_tr.columns)
        X_te = X_te.reindex(columns=X_tr.columns)
        X_fw = X_fw.reindex(columns=X_tr.columns)
        cat = ["coin"] if "coin" in X_tr.columns else []

        model = train_reg(X_tr, y_tr, X_va, y_va, cat)
        print(
            f"train={len(y_tr):,} val={len(y_va):,} test={len(y_te):,} "
            f"forward={len(y_fw):,} best_iter={model.best_iteration}"
        )

        for split_name, X, y in [("val", X_va, y_va), ("test", X_te, y_te), ("forward_jul22_28", X_fw, y_fw)]:
            r = eval_reg(y, model.predict(X, num_iteration=model.best_iteration), f"{cfg['name']} {split_name}")
            r.update({"config": cfg["name"], "split": split_name, "best_iter": model.best_iteration, "note": cfg["note"]})
            rows.append(r)

    # ── literal row-order check: shuffling val rows must not change anything ──
    print("\n-- row-order invariance check (shuffle val rows) --")
    X_tr, y_tr = feature_frame(win(["jun13", "jun22", "jul13"]))
    X_va, y_va = feature_frame(win("jul19_pre"))
    X_va = X_va.reindex(columns=X_tr.columns)
    cat = ["coin"] if "coin" in X_tr.columns else []
    m1 = train_reg(X_tr, y_tr, X_va, y_va, cat)
    perm = np.random.default_rng(7).permutation(len(y_va))
    m2 = train_reg(X_tr, y_tr, X_va.iloc[perm].reset_index(drop=True), y_va[perm], cat)
    X_fw, y_fw = feature_frame(forward_df)
    X_fw = X_fw.reindex(columns=X_tr.columns)
    r1 = eval_reg(y_fw, m1.predict(X_fw, num_iteration=m1.best_iteration), "ordered_val forward")
    r2 = eval_reg(y_fw, m2.predict(X_fw, num_iteration=m2.best_iteration), "shuffled_val forward")
    print(
        f"best_iter ordered={m1.best_iteration} shuffled={m2.best_iteration}  "
        f"forward R² delta={r2['r2'] - r1['r2']:+.6f}"
    )
    for r, nm in [(r1, "rowcheck_ordered_val"), (r2, "rowcheck_shuffled_val")]:
        r.update({"config": nm, "split": "forward_jul22_28", "best_iter": m1.best_iteration if nm.endswith("ordered_val") else m2.best_iteration, "note": "row-order invariance"})
        rows.append(r)

    out = pd.DataFrame(rows)
    out.to_csv(OUT / "val_rotation.csv", index=False)
    print(f"\nsaved {OUT / 'val_rotation.csv'}")

    fwd = out[out["split"] == "forward_jul22_28"]
    print("\n=== forward Jul22-28 (fixed yardstick) by config ===")
    print(fwd[["config", "r2", "dir_acc", "best_iter"]].to_string(index=False))
    print(f"\nforward R² spread across configs: {fwd['r2'].max() - fwd['r2'].min():.5f}")
    return out


# ── 2) Horizon + objective sweep ──────────────────────────────────────────────

def run_horizon_sweep() -> pd.DataFrame:
    print("\n=== HORIZON + TARGET + DirAcc objective sweep ===")
    # Use chronological protocol throughout
    train_ids = ["jun13", "jun22", "jul13"]
    val_id = "jul19_pre"
    test_id = "jul19_post"
    frames = {wid: load_spread(wid) for wid in train_ids + [val_id, test_id]}
    sm = pd.concat(frames.values(), ignore_index=True)

    rows = []
    for horizon in [2, 10, 20, 60, 120]:
        for target_kind in ["zscore_fwd", "zscore_delta"]:
            print(f"\n-- H={horizon} target={target_kind} --")
            feat = build_spread_features(sm, horizon=horizon, target_kind=target_kind)

            def split(ids):
                return feat[feat["window_id"].isin(ids if isinstance(ids, list) else [ids])]

            X_tr, y_tr = feature_frame(split(train_ids))
            X_va, y_va = feature_frame(split(val_id))
            X_te, y_te = feature_frame(split(test_id))
            X_va = X_va.reindex(columns=X_tr.columns)
            X_te = X_te.reindex(columns=X_tr.columns)
            cat = ["coin"] if "coin" in X_tr.columns else []

            # regression
            model = train_reg(X_tr, y_tr, X_va, y_va, cat)
            pred = model.predict(X_te, num_iteration=model.best_iteration)
            r = eval_reg(y_te, pred, f"reg H={horizon} {target_kind}")
            r.update({"horizon": horizon, "target": target_kind, "model": "lgbm_reg", "best_iter": model.best_iteration})
            rows.append(r)

            # naive sign persistence (on zscore_lag1 vs target sign)
            if "zscore_lag1" in X_te.columns:
                naive = X_te["zscore_lag1"].astype(float).values
                rn = eval_reg(y_te, naive, f"naive H={horizon} {target_kind}")
                rn.update({"horizon": horizon, "target": target_kind, "model": "naive_zlag1", "best_iter": 0})
                rows.append(rn)

            # binary classifier on sign(target)
            try:
                clf, Xva_c, yva_c = train_cls(X_tr, y_tr, X_va, y_va, cat)
                m = y_te != 0
                Xte_c = X_te.loc[m].reset_index(drop=True).reindex(columns=X_tr.columns)
                yte_c = (y_te[m] > 0).astype(int)
                proba = clf.predict(Xte_c, num_iteration=clf.best_iteration)
                pred_cls = (proba >= 0.5).astype(int)
                acc = accuracy_score(yte_c, pred_cls)
                # DirAcc vs true sign: map 0/1 back
                dir_acc = float(np.mean((pred_cls == 1) == (yte_c == 1)))
                try:
                    auc = roc_auc_score(yte_c, proba)
                except Exception:
                    auc = float("nan")
                # naive: sign(zscore_lag1)
                naive_cls = (Xte_c["zscore_lag1"].astype(float).values > 0).astype(int)
                naive_acc = accuracy_score(yte_c, naive_cls)
                print(
                    f"{'cls H=' + str(horizon) + ' ' + target_kind:40s} "
                    f"Acc={acc:.3%} AUC={auc:.3f} naiveAcc={naive_acc:.3%} best_iter={clf.best_iteration}"
                )
                rows.append(
                    {
                        "label": f"cls H={horizon} {target_kind}",
                        "mae": np.nan,
                        "rmse": np.nan,
                        "r2": np.nan,
                        "dir_acc": dir_acc,
                        "dir_acc_tau0.5": np.nan,
                        "pnl_proxy": float(np.mean(np.where(pred_cls == 1, 1, -1) * y_te[m])),
                        "n": int(m.sum()),
                        "horizon": horizon,
                        "target": target_kind,
                        "model": "lgbm_cls",
                        "best_iter": clf.best_iteration,
                        "auc": auc,
                        "naive_dir_acc": naive_acc,
                    }
                )
            except Exception as e:
                print(f"classifier skipped H={horizon} {target_kind}: {e}")

    out = pd.DataFrame(rows)
    out.to_csv(OUT / "horizon_sweep.csv", index=False)
    print(f"saved {OUT / 'horizon_sweep.csv'}")
    return out


# ── 3) LSR + liquidations gated ablation (jul22-28 only) ──────────────────────

def run_lsr_ablation() -> pd.DataFrame:
    print("\n=== LSR + liquidations ablation (jul22-28 only) ===")
    sm = load_spread("jul22_28")
    lsr = load_aux_signal("jul22_28", "long_short_ratio")
    liq = load_aux_signal("jul22_28", "liquidations")

    # Chronological split inside the window by snapshot_idx tertiles
    snaps = np.sort(sm["snapshot_idx"].unique())
    n = len(snaps)
    train_max = snaps[int(n * 0.60)]
    val_max = snaps[int(n * 0.80)]
    print(f"snapshot split: train <= {train_max}, val <= {val_max}, test > {val_max} (n_snaps={n})")

    rows = []
    for target_kind in ["zscore_fwd", "zscore_delta"]:
        for use_aux in [False, True]:
            tag = "with_LSR_liqs" if use_aux else "spread_only"
            print(f"\n-- {target_kind} / {tag} --")
            feat = build_spread_features(sm, horizon=2, target_kind=target_kind)
            if use_aux:
                feat = merge_aux(feat, lsr, ["long_short_ratio"], "lsr")
                feat = merge_aux(feat, liq, ["liq_signed_quote", "liq_abs_quote"], "liq")

            feat = feat.copy()
            feat["split"] = np.where(
                feat["snapshot_idx"] <= train_max,
                "train",
                np.where(feat["snapshot_idx"] <= val_max, "val", "test"),
            )
            X_tr, y_tr = feature_frame(feat[feat["split"] == "train"])
            X_va, y_va = feature_frame(feat[feat["split"] == "val"])
            X_te, y_te = feature_frame(feat[feat["split"] == "test"])
            X_va = X_va.reindex(columns=X_tr.columns)
            X_te = X_te.reindex(columns=X_tr.columns)
            cat = ["coin"] if "coin" in X_tr.columns else []
            n_aux = sum(1 for c in X_tr.columns if c.startswith("lsr_") or c.startswith("liq_"))
            print(f"features={X_tr.shape[1]} aux_cols={n_aux} train={len(y_tr)} val={len(y_va)} test={len(y_te)}")

            model = train_reg(X_tr, y_tr, X_va, y_va, cat)
            pred = model.predict(X_te, num_iteration=model.best_iteration)
            r = eval_reg(y_te, pred, f"{target_kind} {tag}")
            r.update(
                {
                    "target": target_kind,
                    "features": tag,
                    "n_features": int(X_tr.shape[1]),
                    "n_aux": n_aux,
                    "best_iter": model.best_iteration,
                    "model": "lgbm_reg",
                }
            )
            rows.append(r)

            if "zscore_lag1" in X_te.columns:
                rn = eval_reg(y_te, X_te["zscore_lag1"].astype(float).values, f"naive {target_kind} {tag}")
                rn.update(
                    {
                        "target": target_kind,
                        "features": tag,
                        "n_features": 1,
                        "n_aux": 0,
                        "best_iter": 0,
                        "model": "naive_zlag1",
                    }
                )
                rows.append(rn)

            # classifier
            try:
                clf, _, _ = train_cls(X_tr, y_tr, X_va, y_va, cat)
                m = y_te != 0
                Xte_c = X_te.loc[m].reset_index(drop=True).reindex(columns=X_tr.columns)
                yte_c = (y_te[m] > 0).astype(int)
                proba = clf.predict(Xte_c, num_iteration=clf.best_iteration)
                pred_cls = (proba >= 0.5).astype(int)
                acc = accuracy_score(yte_c, pred_cls)
                naive_acc = accuracy_score(yte_c, (Xte_c["zscore_lag1"].astype(float).values > 0).astype(int))
                print(f"cls {target_kind} {tag}: Acc={acc:.3%} naive={naive_acc:.3%}")
                rows.append(
                    {
                        "label": f"cls {target_kind} {tag}",
                        "mae": np.nan,
                        "rmse": np.nan,
                        "r2": np.nan,
                        "dir_acc": acc,
                        "dir_acc_tau0.5": np.nan,
                        "pnl_proxy": float(np.mean(np.where(pred_cls == 1, 1, -1) * y_te[m])),
                        "n": int(m.sum()),
                        "target": target_kind,
                        "features": tag,
                        "n_features": int(X_tr.shape[1]),
                        "n_aux": n_aux,
                        "best_iter": clf.best_iteration,
                        "model": "lgbm_cls",
                        "naive_dir_acc": naive_acc,
                    }
                )
            except Exception as e:
                print("cls skipped:", e)

    out = pd.DataFrame(rows)
    out.to_csv(OUT / "lsr_ablation.csv", index=False)
    print(f"saved {OUT / 'lsr_ablation.csv'}")
    return out


def write_defer_note() -> None:
    (OUT / "DEFER_OHLCV.md").write_text(
        """# OHLCV deferred (plan item)

Do **not** enable OHLCV in the GBM pipeline yet.

## Why
- Local 1m OHLCV only covers Jun 13–16 (9 venues) and partially Jun 22–24 (kraken+gateio).
- Jul 13, Jul 19–22, and Jul 22–28 have **no** OHLCV in `cex_unified`.
- Turning the toggle on would inject structured NaNs on most train/test rows and add nothing to the holdouts we care about.

## Also deferred
- Collecting more hours of the *same* signal set (diminishing returns; persistence already explains ~91% of R²).
- Longer LightGBM boosting / hyperparameter retunes (already flat within 0.0005 R²).

## Revisit when
- OHLCV is backfilled for Jul19+ on the same venues as the mid-based spread features, **and**
- paper/backtest both use the same price definition (mids vs 1m closes).
""",
        encoding="utf-8",
    )


def main() -> None:
    assert LOCAL_DATA_ROOT.exists(), LOCAL_DATA_ROOT
    write_defer_note()
    import sys
    only = sys.argv[1] if len(sys.argv) > 1 else "all"
    protocol = horizon = lsr = None
    if only in ("all", "protocol"):
        protocol = run_protocol()
    if only in ("all", "valrot"):
        run_val_rotation()
    if only in ("all", "horizon"):
        horizon = run_horizon_sweep()
    if only in ("all", "lsr"):
        lsr = run_lsr_ablation()

    parts = []
    if (OUT / "protocol_eval.csv").exists():
        parts.append("## Protocol (chronological)\n" + pd.read_csv(OUT / "protocol_eval.csv").to_string(index=False))
    if (OUT / "val_rotation.csv").exists():
        parts.append("## Validation rotation\n" + pd.read_csv(OUT / "val_rotation.csv").to_string(index=False))
    if (OUT / "horizon_sweep.csv").exists():
        parts.append("## Horizon sweep (test = jul19_post)\n" + pd.read_csv(OUT / "horizon_sweep.csv").to_string(index=False))
    if (OUT / "lsr_ablation.csv").exists():
        parts.append("## LSR / liquidations (jul22-28 only)\n" + pd.read_csv(OUT / "lsr_ablation.csv").to_string(index=False))
    (OUT / "SUMMARY.txt").write_text("\n\n".join(parts), encoding="utf-8")
    print(f"\nAll done. See {OUT}/")


if __name__ == "__main__":
    main()
