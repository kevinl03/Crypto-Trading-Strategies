"""Fair persistence baselines for the zscore_fwd target.

Compares:
  1. Naive raw zscore_lag1 (the broken baseline already in the notebook)
  2. OLS of target ~ zscore_lag1  (properly scaled persistence)
  3. OLS of target ~ zscore_lag1 + zscore_lag2
  4. LightGBM on zscore lags only (same hyperparams / early stopping as full model)
  5. LightGBM on zscore + spread_bps lags
  6. Published full-model metrics from outputs_ob_fix (for reference)

Only needs spread_matrix parquets — much faster than the full feature pipeline.
"""
from __future__ import annotations

import json
import gc
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

LOCAL_DATA_ROOT = Path(r"C:/Users/Kev/repos/stochastic-spread-modeling/data/cex_unified")
OUTPUT_DIR = Path("./outputs_persistence")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

HORIZON = 2
ZSCORE_WINDOW = 120
N_LAGS = 2
MIN_PERIODS = 30
TOP_EXCHANGES = ["binance", "bybit", "okx", "coinbase", "kraken", "mexc"]

LGBM_PARAMS = {
    "objective": "regression",
    "metric": ["rmse", "mae"],
    "learning_rate": 0.05,
    "num_leaves": 127,
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
NUM_BOOST_ROUND = 2000
EARLY_STOPPING = 50

WINDOWS = [
    {"id": "jun13", "role": "train", "local_dir": ""},
    {"id": "jun22", "role": "train", "local_dir": "test"},
    {"id": "jul13", "role": "train", "local_dir": "validation"},
    {"id": "jul22_28", "role": "train", "local_dir": "validation_jul22-28"},
    {"id": "jul19_pre", "role": "test", "local_dir": "validation_jul19-22/pre_outage"},
    {"id": "jul19_post", "role": "test", "local_dir": "validation_jul19-22/post_outage"},
]


def load_spread(window: dict) -> pd.DataFrame:
    rel = (window.get("local_dir") or "").strip("/\\")
    path = LOCAL_DATA_ROOT / rel / "spread_matrix.parquet" if rel else LOCAL_DATA_ROOT / "spread_matrix.parquet"
    print(f"  [{window['id']}] {path.name} …", end=" ", flush=True)
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
                        (
                            row.snapshot_idx,
                            row.coin,
                            pair["ex1"],
                            pair["ex2"],
                            float(pair["spread_bps"]),
                        )
                    )
        except Exception:
            continue
    del df
    gc.collect()
    out = pd.DataFrame(
        records,
        columns=["snapshot_idx", "coin", "exchange_a", "exchange_b", "spread_bps"],
    )
    out["window_id"] = window["id"]
    out["spread_bps"] = out["spread_bps"].astype("float32")
    out["snapshot_idx"] = out["snapshot_idx"].astype("int32")
    print(out.shape)
    return out


def build_spread_features(sm: pd.DataFrame) -> pd.DataFrame:
    sm = sm.copy()
    sm["pair"] = sm["exchange_a"] + "__" + sm["exchange_b"]
    sm = sm.sort_values(["window_id", "coin", "pair", "snapshot_idx"]).reset_index(drop=True)
    grp_keys = ["window_id", "coin", "pair"]
    grp = sm.groupby(grp_keys)["spread_bps"]
    roll_mean = grp.transform(lambda x: x.rolling(ZSCORE_WINDOW, min_periods=MIN_PERIODS).mean())
    roll_std = grp.transform(lambda x: x.rolling(ZSCORE_WINDOW, min_periods=MIN_PERIODS).std())
    sm["zscore"] = (sm["spread_bps"] - roll_mean) / roll_std.replace(0, np.nan)
    sm["target"] = sm.groupby(grp_keys)["zscore"].transform(lambda x: x.shift(-HORIZON))
    for lag in range(1, N_LAGS + 1):
        sm[f"spread_bps_lag{lag}"] = grp.transform(lambda x, l=lag: x.shift(l))
        sm[f"zscore_lag{lag}"] = sm.groupby(grp_keys)["zscore"].transform(lambda x, l=lag: x.shift(l))
    return sm


def eval_preds(y, preds, label: str) -> dict:
    mae = mean_absolute_error(y, preds)
    rmse = float(np.sqrt(mean_squared_error(y, preds)))
    r2 = r2_score(y, preds)
    dir_acc = float(np.mean(np.sign(preds) == np.sign(y)))
    print(f"{label:40s}  MAE={mae:.4f}  RMSE={rmse:.4f}  R²={r2:.4f}  DirAcc={dir_acc:.3%}")
    return {"label": label, "mae": mae, "rmse": rmse, "r2": r2, "dir_acc": dir_acc}


def main() -> None:
    frames = {"train": [], "test": []}
    for w in WINDOWS:
        frames[w["role"]].append(load_spread(w))

    print("\nBuilding features …")
    df_train = build_spread_features(pd.concat(frames["train"], ignore_index=True))
    df_test = build_spread_features(pd.concat(frames["test"], ignore_index=True))
    df_train = df_train.dropna(subset=["target", "zscore_lag1", "zscore_lag2"]).copy()
    df_test = df_test.dropna(subset=["target", "zscore_lag1", "zscore_lag2"]).copy()
    print(f"Train rows: {len(df_train):,}  |  Test rows: {len(df_test):,}")

    y_tr = df_train["target"].values
    y_te = df_test["target"].values
    results = []

    # 1) Naive raw lag (broken baseline)
    results.append(eval_preds(y_te, df_test["zscore_lag1"].values, "test naive raw zscore_lag1"))
    results.append(eval_preds(y_tr, df_train["zscore_lag1"].values, "train naive raw zscore_lag1"))

    # 2) OLS ~ lag1
    ols1 = LinearRegression().fit(df_train[["zscore_lag1"]], y_tr)
    print(f"OLS lag1 coef={ols1.coef_[0]:.4f}  intercept={ols1.intercept_:.4f}")
    results.append(eval_preds(y_te, ols1.predict(df_test[["zscore_lag1"]]), "test OLS ~ zscore_lag1"))
    results.append(eval_preds(y_tr, ols1.predict(df_train[["zscore_lag1"]]), "train OLS ~ zscore_lag1"))

    # 3) OLS ~ lag1 + lag2
    ols2 = LinearRegression().fit(df_train[["zscore_lag1", "zscore_lag2"]], y_tr)
    print(f"OLS lag1+2 coefs={ols2.coef_}  intercept={ols2.intercept_:.4f}")
    results.append(
        eval_preds(y_te, ols2.predict(df_test[["zscore_lag1", "zscore_lag2"]]), "test OLS ~ zscore_lag1+2")
    )
    results.append(
        eval_preds(y_tr, ols2.predict(df_train[["zscore_lag1", "zscore_lag2"]]), "train OLS ~ zscore_lag1+2")
    )

    # 4) LGBM on zscore lags only
    for name, cols in [
        ("lgbm zscore_lags", ["zscore_lag1", "zscore_lag2"]),
        ("lgbm zscore+spread_lags", ["zscore_lag1", "zscore_lag2", "spread_bps_lag1", "spread_bps_lag2"]),
    ]:
        Xtr = df_train[cols]
        Xte = df_test[cols]
        dtrain = lgb.Dataset(Xtr, label=y_tr)
        dtest = lgb.Dataset(Xte, label=y_te, reference=dtrain)
        model = lgb.train(
            LGBM_PARAMS,
            dtrain,
            num_boost_round=NUM_BOOST_ROUND,
            valid_sets=[dtrain, dtest],
            valid_names=["train", "test"],
            callbacks=[lgb.early_stopping(EARLY_STOPPING, verbose=False), lgb.log_evaluation(0)],
        )
        print(f"{name}: best_iteration={model.best_iteration}")
        results.append(
            eval_preds(y_te, model.predict(Xte, num_iteration=model.best_iteration), f"test {name}")
        )
        results.append(
            eval_preds(y_tr, model.predict(Xtr, num_iteration=model.best_iteration), f"train {name}")
        )

    # 5) Reference full model from prior run
    full = Path("./outputs_ob_fix/eval_results.csv")
    if full.exists():
        ref = pd.read_csv(full)
        for _, row in ref.iterrows():
            if row.get("model", "lgbm") == "lgbm":
                results.append(
                    {
                        "label": f"{row['label']} FULL model (ref)",
                        "mae": row["mae"],
                        "rmse": row["rmse"],
                        "r2": row["r2"],
                        "dir_acc": row["dir_acc"],
                    }
                )
                print(
                    f"{row['label'] + ' FULL model (ref)':40s}  "
                    f"MAE={row['mae']:.4f}  RMSE={row['rmse']:.4f}  "
                    f"R²={row['r2']:.4f}  DirAcc={row['dir_acc']:.3%}"
                )

    out = pd.DataFrame(results)
    out.to_csv(OUTPUT_DIR / "eval_results.csv", index=False)
    print(f"\nSaved {OUTPUT_DIR / 'eval_results.csv'}")
    print("\n=== TEST summary ===")
    test = out[out["label"].str.startswith("test") | out["label"].str.startswith("test ")]
    # also catch 'test FULL'
    test = out[out["label"].str.contains(r"^test\b", regex=True)]
    print(test.sort_values("r2", ascending=False).to_string(index=False))


if __name__ == "__main__":
    main()
