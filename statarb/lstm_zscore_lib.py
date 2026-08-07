"""
LSTM z-score forecasting pipeline for cross-exchange CEX spreads.

Used by cex_lstm_zscore.ipynb. Protocol matches cex_gbm_new.ipynb Jul-25 split
and target (z_{t+1}), but uses LSTM-native sequence features (not LGBM columns).
"""

from __future__ import annotations

import gc
import json
import math
import random
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

# ── Protocol / defaults ───────────────────────────────────────────────────────

HF_REPO = "SFU-fintech-AI/statarb-crypto-research"
HF_REVISION = "c5c695d3cec28db8801fe6de173b3c21f3803436"
HF_PARQUET_REVISION = "main"

HORIZON = 1
ZSCORE_WINDOW = 300
MIN_PERIODS = 90
SEQ_LEN = 64
ENTRY_TAU = 0.5
SEED = 42

TOP_EXCHANGES = ["binance", "bybit", "okx", "coinbase", "kraken", "mexc"]

LOAD_TABLES = {
    "spread_matrix": True,
    "ticker": True,
    "orderbook": True,
    "trades": True,
    "funding_rate": False,  # v1: off until coverage check flips True
    "open_interest": False,
    "ohlcv": False,
}

SUBSETS = [
    "spread_matrix",
    "ticker",
    "orderbook",
    "trades",
    "funding_rate",
    "open_interest",
    "ohlcv",
]

CHANNEL_NAMES = [
    "spread_bps",
    "zscore",
    "log_mid_diff",
    "ba_bps_a",
    "ba_bps_b",
    "bid_vol_a",
    "ask_vol_a",
    "bid_vol_b",
    "ask_vol_b",
    "imb_a",
    "imb_b",
    "bsr_a",
    "vol_a",
    "bsr_b",
    "vol_b",
    "cross_mid_std",
    "cross_ba_std",
    "net_ob_pressure",
]

VOLUME_CHANNELS = ["bid_vol_a", "ask_vol_a", "bid_vol_b", "ask_vol_b", "vol_a", "vol_b"]
WINSOR_CHANNELS = ["spread_bps", "ba_bps_a", "ba_bps_b"]

WINDOWS = [
    {
        "id": "jun13",
        "role": "train",
        "label": "Jun 13-16",
        "hf_prefix": "",
        "hf_split": "train",
        "hf_ohlcv": "ohlcv",
        "ohlcv_schema": "flat",
        "local_dir": "",
        "load_mode": "dataset_config",
    },
    {
        "id": "jun22",
        "role": "train",
        "label": "Jun 22-24",
        "hf_prefix": "test_",
        "hf_split": "train",
        "hf_ohlcv": "test_ohlcv_live",
        "ohlcv_schema": "flat",
        "local_dir": "test",
        "load_mode": "dataset_config",
    },
    {
        "id": "jul13",
        "role": "train",
        "label": "Jul 13-15 (partial)",
        "hf_prefix": "validation_",
        "hf_split": "train",
        "hf_ohlcv": None,
        "ohlcv_schema": "flat",
        "local_dir": "validation",
        "load_mode": "dataset_config",
    },
    {
        "id": "jul19_pre",
        "role": "train",
        "label": "Jul 19-22 pre-outage",
        "hf_prefix": "jul19_22_pre_outage_",
        "hf_split": "train",
        "hf_ohlcv": "jul19_22_pre_outage_ohlcv_snapshot",
        "ohlcv_schema": "snapshot",
        "local_dir": "validation_jul19-22/pre_outage",
        "load_mode": "dataset_config",
    },
    {
        "id": "jul19_post",
        "role": "train",
        "label": "Jul 19-22 post-outage",
        "hf_prefix": "jul19_22_post_outage_",
        "hf_split": "train",
        "hf_ohlcv": "jul19_22_post_outage_ohlcv_snapshot",
        "ohlcv_schema": "snapshot",
        "local_dir": "validation_jul19-22/post_outage",
        "load_mode": "dataset_config",
    },
    {
        "id": "jul22_24",
        "role": "train",
        "label": "Jul 22-24 (from jul22-28 run)",
        "hf_prefix": "",
        "hf_split": "train",
        "hf_ohlcv": None,
        "ohlcv_schema": "flat",
        "local_dir": "validation_jul22-28",
        "load_mode": "parquet_dir",
        "hf_parquet_dir": "validation_jul22-28",
        "hf_parquet_revision": "main",
        "snapshot_idx_max": 3584,
    },
    {
        "id": "jul25_28",
        "role": "test",
        "label": "Jul 25-28",
        "hf_prefix": "",
        "hf_split": "train",
        "hf_ohlcv": None,
        "ohlcv_schema": "flat",
        "local_dir": "validation_jul22-28",
        "load_mode": "parquet_dir",
        "hf_parquet_dir": "validation_jul22-28",
        "hf_parquet_revision": "main",
        "snapshot_idx_min": 3584,
    },
]

_LOCAL_CANDIDATES = [
    Path(r"C:/Users/Kev/repos/stochastic-spread-modeling/data/cex_unified"),
    Path("../data/cex_unified"),
    Path("../../stochastic-spread-modeling/data/cex_unified"),
    Path("./cex_unified"),
    Path("../data/cex_unified"),
]


@dataclass
class TrainConfig:
    seq_len: int = SEQ_LEN
    hidden_size: int = 128
    num_layers: int = 2
    dropout: float = 0.2
    emb_dim: int = 8
    lr: float = 1e-3
    weight_decay: float = 1e-5
    batch_size: int = 512
    max_epochs: int = 30
    patience: int = 6
    val_fraction: float = 0.15
    seed: int = SEED
    entry_tau: float = ENTRY_TAU
    num_workers: int = 0
    # Cap sequence counts for CPU-friendly runs (chronological subsample)
    max_train_seqs: int = 250_000
    max_test_seqs: int = 150_000


@dataclass
class SequenceBundle:
    X: np.ndarray
    y: np.ndarray
    coin_id: np.ndarray
    pair_id: np.ndarray
    z_now: np.ndarray
    meta: pd.DataFrame
    channel_names: list[str] = field(default_factory=lambda: list(CHANNEL_NAMES))


def resolve_local_data_root() -> Path | None:
    for p in _LOCAL_CANDIDATES:
        try:
            rp = p.resolve()
        except OSError:
            continue
        if rp.exists():
            return rp
    return None


def resolve_hf_token() -> str | None:
    try:
        import google.colab  # noqa: F401

        from google.colab import userdata

        tok = userdata.get("HF_TOKEN")
        return tok
    except Exception:
        pass
    try:
        from huggingface_hub import get_token

        return get_token()
    except Exception:
        return None


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def local_parquet_path(window: dict, local_name: str, local_root: Path | None) -> Path | None:
    if local_root is None:
        return None
    rel_dir = (window.get("local_dir") or "").strip("/\\")
    path = (
        local_root / rel_dir / f"{local_name}.parquet"
        if rel_dir
        else local_root / f"{local_name}.parquet"
    )
    return path if path.exists() else None


def hf_name_for(window: dict, local_name: str) -> str:
    return f"{window['hf_prefix']}{local_name}"


def _load_one_window(
    window: dict,
    *,
    local_root: Path | None,
    hf_token: str | None,
    use_hf: bool = True,
    load_tables: dict[str, bool] | None = None,
) -> dict[str, pd.DataFrame]:
    load_tables = load_tables or LOAD_TABLES
    result: dict[str, pd.DataFrame] = {}
    wid = window["id"]

    for local_name in SUBSETS:
        if not load_tables.get(local_name, True):
            print(f"    [skip] {local_name}")
            result[local_name] = pd.DataFrame()
            continue

        local_path = local_parquet_path(window, local_name, local_root)
        load_mode = window.get("load_mode", "dataset_config")
        print(f"    loading {local_name} …", end=" ")
        try:
            if local_path is not None:
                print(f"[local {local_path.name}] ", end="")
                df = pd.read_parquet(local_path)
            elif not use_hf:
                print("[no local and USE_HF=False]")
                result[local_name] = pd.DataFrame()
                continue
            elif load_mode == "parquet_dir":
                from huggingface_hub import hf_hub_download

                rel = f"{window['hf_parquet_dir'].rstrip('/')}/{local_name}.parquet"
                rev = window.get("hf_parquet_revision", HF_PARQUET_REVISION)
                print(f"[hf parquet {rel}] ", end="")
                path = hf_hub_download(
                    repo_id=HF_REPO,
                    repo_type="dataset",
                    filename=rel,
                    revision=rev,
                    token=hf_token,
                )
                df = pd.read_parquet(path)
            else:
                from datasets import load_dataset

                hf = hf_name_for(window, local_name)
                print(f"[hf config {hf}] ", end="")
                ds = load_dataset(
                    HF_REPO,
                    hf,
                    split="train",
                    token=hf_token,
                    revision=HF_REVISION,
                )
                drop = [c for c in ["run_id", "ts", "market", "symbol", "error"] if c in ds.column_names]
                if drop:
                    ds = ds.remove_columns(drop)
                df = ds.to_pandas()
                del ds
                gc.collect()
        except Exception as e:
            print(f"[WARN: {e}]")
            result[local_name] = pd.DataFrame()
            continue

        if "error" in df.columns:
            df = df[df["error"].isna()].drop(columns=["error"], errors="ignore")

        if "snapshot_idx" in df.columns:
            if window.get("snapshot_idx_min") is not None:
                df = df[df["snapshot_idx"] >= int(window["snapshot_idx_min"])]
            if window.get("snapshot_idx_max") is not None:
                df = df[df["snapshot_idx"] < int(window["snapshot_idx_max"])]
            if df.empty:
                print("[empty after snapshot cut] ", end="")
                result[local_name] = pd.DataFrame()
                continue

        drop = [c for c in ["run_id", "ts", "market", "symbol"] if c in df.columns]
        if drop:
            df = df.drop(columns=drop)
        gc.collect()

        if local_name == "ohlcv":
            result[local_name] = pd.DataFrame()
            print("[ohlcv skipped in LSTM v1]")
            continue

        records: list[tuple] = []
        for row in df.itertuples():
            try:
                p = json.loads(row.payload) if isinstance(row.payload, str) else row.payload
                snap = row.snapshot_idx
                coin = row.coin
                exch = getattr(row, "exchange", None)

                if local_name == "spread_matrix":
                    for pair in p["pairwise_spreads"]:
                        if pair["ex1"] in TOP_EXCHANGES and pair["ex2"] in TOP_EXCHANGES:
                            records.append(
                                (
                                    snap,
                                    coin,
                                    pair["ex1"],
                                    pair["ex2"],
                                    float(pair["spread_bps"]),
                                    float(pair["p1"]),
                                    float(pair["p2"]),
                                )
                            )
                elif local_name == "ticker" and exch in TOP_EXCHANGES:
                    records.append(
                        (
                            snap,
                            coin,
                            exch,
                            float(p.get("mid") or 0),
                            float(p.get("spread_bps") or 0),
                            float(p.get("bid_volume") or 0),
                            float(p.get("ask_volume") or 0),
                        )
                    )
                elif local_name == "orderbook" and exch in TOP_EXCHANGES:
                    records.append((snap, coin, exch, float(p.get("imbalance") or 0)))
                elif local_name == "trades" and exch in TOP_EXCHANGES:
                    records.append(
                        (
                            snap,
                            coin,
                            exch,
                            float(p.get("buy_sell_ratio") or 0),
                            float(p.get("total_volume") or 0),
                        )
                    )
                elif local_name == "funding_rate" and exch in TOP_EXCHANGES:
                    records.append((snap, coin, exch, float(p.get("funding_rate") or 0)))
                elif local_name == "open_interest" and exch in TOP_EXCHANGES:
                    records.append(
                        (snap, coin, exch, float(p.get("open_interest_amount") or 0))
                    )
            except Exception:
                continue

        del df
        gc.collect()

        cols = {
            "spread_matrix": [
                "snapshot_idx",
                "coin",
                "exchange_a",
                "exchange_b",
                "spread_bps",
                "p1",
                "p2",
            ],
            "ticker": [
                "snapshot_idx",
                "coin",
                "exchange",
                "mid",
                "spread_bps",
                "bid_volume",
                "ask_volume",
            ],
            "orderbook": ["snapshot_idx", "coin", "exchange", "imbalance"],
            "trades": ["snapshot_idx", "coin", "exchange", "buy_sell_ratio", "total_volume"],
            "funding_rate": ["snapshot_idx", "coin", "exchange", "funding_rate"],
            "open_interest": ["snapshot_idx", "coin", "exchange", "oi_amount"],
        }
        out = pd.DataFrame(records, columns=cols[local_name])
        out["window_id"] = wid
        num_cols = out.select_dtypes(include="float64").columns
        out[num_cols] = out[num_cols].astype("float32")
        int_cols = out.select_dtypes(include="int64").columns
        out[int_cols] = out[int_cols].astype("int32")
        result[local_name] = out
        print(out.shape, f"  mem: {out.memory_usage(deep=True).sum() / 1e6:.1f} MB")
        del records, out
        gc.collect()

    return result


def pool_windows(
    windows: list[dict],
    *,
    local_root: Path | None,
    hf_token: str | None,
    use_hf: bool = True,
    load_tables: dict[str, bool] | None = None,
) -> dict[str, pd.DataFrame]:
    accumulated: dict[str, list[pd.DataFrame]] = {s: [] for s in SUBSETS}
    for w in windows:
        print(f"\n  -- {w['label']} ({w['id']}) --")
        window_data = _load_one_window(
            w, local_root=local_root, hf_token=hf_token, use_hf=use_hf, load_tables=load_tables
        )
        for table, df in window_data.items():
            if not df.empty:
                accumulated[table].append(df)

    pooled: dict[str, pd.DataFrame] = {}
    for table, frames in accumulated.items():
        if frames:
            pooled[table] = pd.concat(frames, ignore_index=True)
            print(f"  pooled {table:20s} → {len(pooled[table]):>10,} rows")
        else:
            pooled[table] = pd.DataFrame()
    return pooled


def build_spread_target(sm: pd.DataFrame) -> pd.DataFrame:
    sm = sm.copy()
    sm["pair"] = sm["exchange_a"] + "__" + sm["exchange_b"]
    grp_keys = ["window_id", "coin", "pair"]
    sm = sm.sort_values(grp_keys + ["snapshot_idx"]).reset_index(drop=True)
    grp = sm.groupby(grp_keys, sort=False)["spread_bps"]
    roll_mean = grp.transform(
        lambda x: x.rolling(ZSCORE_WINDOW, min_periods=MIN_PERIODS).mean()
    )
    roll_std = grp.transform(
        lambda x: x.rolling(ZSCORE_WINDOW, min_periods=MIN_PERIODS).std()
    )
    sm["zscore"] = ((sm["spread_bps"] - roll_mean) / roll_std.replace(0, np.nan)).astype(
        "float32"
    )
    sm["target"] = sm.groupby(grp_keys)["zscore"].transform(lambda x: x.shift(-HORIZON))
    return sm


def _pivot_venue(
    df: pd.DataFrame, value_cols: list[str], prefix: str
) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    keys = ["window_id", "snapshot_idx", "coin"]
    pieces = []
    for col in value_cols:
        wide = df.pivot_table(
            index=keys, columns="exchange", values=col, aggfunc="last"
        )
        wide.columns = [f"{prefix}{col}_{ex}" for ex in wide.columns]
        pieces.append(wide)
    out = pd.concat(pieces, axis=1).reset_index()
    return out


def build_panel(raw: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Join spread backbone with venue micros and light cross-exchange stats."""
    sm = build_spread_target(raw["spread_matrix"])
    if sm.empty:
        return sm

    tk = raw.get("ticker", pd.DataFrame())
    ob = raw.get("orderbook", pd.DataFrame())
    tr = raw.get("trades", pd.DataFrame())

    tk_wide = _pivot_venue(tk, ["mid", "spread_bps", "bid_volume", "ask_volume"], "tk_")
    ob_wide = _pivot_venue(ob, ["imbalance"], "ob_")
    tr_wide = _pivot_venue(tr, ["buy_sell_ratio", "total_volume"], "tr_")

    panel = sm
    for wide in (tk_wide, ob_wide, tr_wide):
        if wide is not None and not wide.empty:
            panel = panel.merge(
                wide, on=["window_id", "snapshot_idx", "coin"], how="left"
            )

    # Light cross-venue context from ticker/OB (available venues)
    mid_cols = [c for c in panel.columns if c.startswith("tk_mid_")]
    ba_cols = [c for c in panel.columns if c.startswith("tk_spread_bps_")]
    imb_cols = [c for c in panel.columns if c.startswith("ob_imbalance_")]

    if mid_cols:
        panel["cross_mid_std"] = panel[mid_cols].std(axis=1, skipna=True).astype("float32")
    else:
        panel["cross_mid_std"] = np.float32(0)
    if ba_cols:
        panel["cross_ba_std"] = panel[ba_cols].std(axis=1, skipna=True).astype("float32")
    else:
        panel["cross_ba_std"] = np.float32(0)
    if imb_cols:
        panel["net_ob_pressure"] = panel[imb_cols].mean(axis=1, skipna=True).astype("float32")
    else:
        panel["net_ob_pressure"] = np.float32(0)

    return panel


def _leg_col(panel: pd.DataFrame, prefix: str, feature: str, exchange: pd.Series) -> pd.Series:
    """Gather venue-specific wide column into a series aligned to panel rows."""
    out = pd.Series(np.nan, index=panel.index, dtype="float32")
    for ex in TOP_EXCHANGES:
        col = f"{prefix}{feature}_{ex}"
        if col not in panel.columns:
            continue
        mask = exchange == ex
        out.loc[mask] = panel.loc[mask, col].astype("float32")
    return out


def panel_to_feature_frame(panel: pd.DataFrame) -> pd.DataFrame:
    """Collapse wide venue columns into pair-leg channels."""
    df = panel.copy()
    df["mid_a"] = _leg_col(df, "tk_", "mid", df["exchange_a"])
    df["mid_b"] = _leg_col(df, "tk_", "mid", df["exchange_b"])
    df["ba_bps_a"] = _leg_col(df, "tk_", "spread_bps", df["exchange_a"])
    df["ba_bps_b"] = _leg_col(df, "tk_", "spread_bps", df["exchange_b"])
    df["bid_vol_a"] = _leg_col(df, "tk_", "bid_volume", df["exchange_a"])
    df["ask_vol_a"] = _leg_col(df, "tk_", "ask_volume", df["exchange_a"])
    df["bid_vol_b"] = _leg_col(df, "tk_", "bid_volume", df["exchange_b"])
    df["ask_vol_b"] = _leg_col(df, "tk_", "ask_volume", df["exchange_b"])
    df["imb_a"] = _leg_col(df, "ob_", "imbalance", df["exchange_a"])
    df["imb_b"] = _leg_col(df, "ob_", "imbalance", df["exchange_b"])
    df["bsr_a"] = _leg_col(df, "tr_", "buy_sell_ratio", df["exchange_a"])
    df["bsr_b"] = _leg_col(df, "tr_", "buy_sell_ratio", df["exchange_b"])
    df["vol_a"] = _leg_col(df, "tr_", "total_volume", df["exchange_a"])
    df["vol_b"] = _leg_col(df, "tr_", "total_volume", df["exchange_b"])

    mid_a = df["mid_a"].astype("float64")
    mid_b = df["mid_b"].astype("float64")
    with np.errstate(divide="ignore", invalid="ignore"):
        df["log_mid_diff"] = (np.log(mid_a.replace(0, np.nan)) - np.log(mid_b.replace(0, np.nan))).astype(
            "float32"
        )

    keep = [
        "window_id",
        "snapshot_idx",
        "coin",
        "pair",
        "exchange_a",
        "exchange_b",
        "target",
        *CHANNEL_NAMES,
    ]
    # ensure all channels exist
    for c in CHANNEL_NAMES:
        if c not in df.columns:
            df[c] = np.float32(np.nan)
    return df[keep].copy()


def apply_row_transforms(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in VOLUME_CHANNELS:
        if c in out.columns:
            out[c] = np.log1p(out[c].clip(lower=0)).astype("float32")
    return out


def winsorize_fit(df: pd.DataFrame, cols: list[str]) -> dict[str, tuple[float, float]]:
    bounds: dict[str, tuple[float, float]] = {}
    for c in cols:
        if c not in df.columns:
            continue
        lo, hi = df[c].quantile([0.01, 0.99])
        bounds[c] = (float(lo), float(hi))
    return bounds


def winsorize_apply(df: pd.DataFrame, bounds: dict[str, tuple[float, float]]) -> pd.DataFrame:
    out = df.copy()
    for c, (lo, hi) in bounds.items():
        if c in out.columns:
            out[c] = out[c].clip(lo, hi)
    return out


def build_sequences(
    feat: pd.DataFrame,
    *,
    seq_len: int = SEQ_LEN,
    channel_names: list[str] | None = None,
    coin_to_id: dict[str, int] | None = None,
    pair_to_id: dict[str, int] | None = None,
    stride: int = 1,
) -> tuple[SequenceBundle, dict[str, int], dict[str, int]]:
    """Build supervised sequences.

    ``stride`` > 1 keeps every Nth decision time within each series (memory).
    """
    channel_names = channel_names or list(CHANNEL_NAMES)
    stride = max(1, int(stride))
    feat = feat.sort_values(["window_id", "coin", "pair", "snapshot_idx"]).reset_index(drop=True)

    if coin_to_id is None:
        coins = sorted(feat["coin"].dropna().unique().tolist())
        coin_to_id = {c: i for i, c in enumerate(coins)}
    if pair_to_id is None:
        pairs = sorted(feat["pair"].dropna().unique().tolist())
        pair_to_id = {p: i for i, p in enumerate(pairs)}

    Xs: list[np.ndarray] = []
    ys: list[float] = []
    coin_ids: list[int] = []
    pair_ids: list[int] = []
    z_nows: list[float] = []
    metas: list[dict[str, Any]] = []

    grp_keys = ["window_id", "coin", "pair"]
    for (wid, coin, pair), g in feat.groupby(grp_keys, sort=False):
        g = g.reset_index(drop=True)
        if len(g) < seq_len + HORIZON:
            continue
        arr = g[channel_names].to_numpy(dtype=np.float32)
        targets = g["target"].to_numpy(dtype=np.float32)
        zscores = g["zscore"].to_numpy(dtype=np.float32)
        snaps = g["snapshot_idx"].to_numpy()

        # valid decision indices t where history [t-seq_len+1, t] exists and target ok
        for t in range(seq_len - 1, len(g), stride):
            y = targets[t]
            z = zscores[t]
            if not np.isfinite(y) or not np.isfinite(z):
                continue
            window = arr[t - seq_len + 1 : t + 1]
            if window.shape[0] != seq_len:
                continue
            if not np.isfinite(window).all():
                # allow partial nan → fill later; skip if entire row nan-heavy
                if np.isnan(window).mean() > 0.35:
                    continue
                window = np.nan_to_num(window, nan=0.0, posinf=0.0, neginf=0.0)

            if coin not in coin_to_id or pair not in pair_to_id:
                continue

            Xs.append(window)
            ys.append(float(y))
            coin_ids.append(coin_to_id[coin])
            pair_ids.append(pair_to_id[pair])
            z_nows.append(float(z))
            metas.append(
                {
                    "window_id": wid,
                    "snapshot_idx": int(snaps[t]),
                    "coin": coin,
                    "pair": pair,
                }
            )

    if not Xs:
        empty = SequenceBundle(
            X=np.zeros((0, seq_len, len(channel_names)), dtype=np.float32),
            y=np.zeros((0,), dtype=np.float32),
            coin_id=np.zeros((0,), dtype=np.int64),
            pair_id=np.zeros((0,), dtype=np.int64),
            z_now=np.zeros((0,), dtype=np.float32),
            meta=pd.DataFrame(columns=["window_id", "snapshot_idx", "coin", "pair"]),
            channel_names=channel_names,
        )
        return empty, coin_to_id, pair_to_id

    bundle = SequenceBundle(
        X=np.stack(Xs).astype(np.float32),
        y=np.asarray(ys, dtype=np.float32),
        coin_id=np.asarray(coin_ids, dtype=np.int64),
        pair_id=np.asarray(pair_ids, dtype=np.int64),
        z_now=np.asarray(z_nows, dtype=np.float32),
        meta=pd.DataFrame(metas),
        channel_names=channel_names,
    )
    return bundle, coin_to_id, pair_to_id


def _chrono_order(meta: pd.DataFrame) -> np.ndarray:
    win_order = {w["id"]: i for i, w in enumerate(WINDOWS)}
    order_key = (
        meta["window_id"].map(win_order).fillna(999).to_numpy() * 1_000_000_000
        + meta["snapshot_idx"].to_numpy()
    )
    return np.argsort(order_key, kind="mergesort")


def chronological_val_split(
    bundle: SequenceBundle, val_fraction: float = 0.15
) -> tuple[np.ndarray, np.ndarray]:
    """Return boolean masks (train_mask, val_mask) with chronological val carve."""
    n = len(bundle.y)
    if n == 0:
        return np.array([], dtype=bool), np.array([], dtype=bool)

    order = _chrono_order(bundle.meta)
    n_val = max(1, int(round(n * val_fraction)))
    val_idx = order[-n_val:]
    train_idx = order[:-n_val] if n_val < n else order[:0]
    train_mask = np.zeros(n, dtype=bool)
    val_mask = np.zeros(n, dtype=bool)
    train_mask[train_idx] = True
    val_mask[val_idx] = True
    return train_mask, val_mask


def subsample_bundle(bundle: SequenceBundle, max_n: int, *, keep_tail: bool = True) -> SequenceBundle:
    """Chronologically subsample a bundle to at most ``max_n`` sequences."""
    n = len(bundle.y)
    if max_n is None or n <= max_n or max_n <= 0:
        return bundle
    order = _chrono_order(bundle.meta)
    keep = order[-max_n:] if keep_tail else order[:max_n]
    keep = np.sort(keep)
    return SequenceBundle(
        X=bundle.X[keep],
        y=bundle.y[keep],
        coin_id=bundle.coin_id[keep],
        pair_id=bundle.pair_id[keep],
        z_now=bundle.z_now[keep],
        meta=bundle.meta.iloc[keep].reset_index(drop=True),
        channel_names=bundle.channel_names,
    )


def fit_scaler(X: np.ndarray, max_rows: int = 2_000_000) -> StandardScaler:
    """Fit StandardScaler without materializing the full N*T float64 matrix."""
    scaler = StandardScaler()
    n, t, c = X.shape
    # Prefer last-timestep rows (one per sequence); augment with a timestep subsample if small.
    last = np.ascontiguousarray(X[:, -1, :], dtype=np.float32)
    if len(last) > max_rows:
        rng = np.random.default_rng(SEED)
        idx = rng.choice(len(last), size=max_rows, replace=False)
        sample = last[idx]
    else:
        sample = last
        if n * min(t, 4) <= max_rows:
            # add a few extra timesteps for stabler channel stats
            extra_t = [0, t // 3, (2 * t) // 3] if t >= 3 else [0]
            parts = [sample] + [np.ascontiguousarray(X[:, j, :], dtype=np.float32) for j in extra_t]
            sample = np.concatenate(parts, axis=0)
            if len(sample) > max_rows:
                sample = sample[:max_rows]
    scaler.fit(sample)
    return scaler


def transform_X(X: np.ndarray, scaler: StandardScaler, chunk_seqs: int = 50_000) -> np.ndarray:
    """Transform in sequence chunks to avoid huge temporary allocations."""
    n, t, c = X.shape
    out = np.empty_like(X, dtype=np.float32)
    mean = scaler.mean_.astype(np.float32)
    scale = scaler.scale_.astype(np.float32)
    scale = np.where(scale == 0, 1.0, scale)
    for i in range(0, n, chunk_seqs):
        sl = slice(i, min(i + chunk_seqs, n))
        block = X[sl].astype(np.float32, copy=False)
        out[sl] = (block - mean) / scale
    return out


def sharpe_ratio(x: np.ndarray) -> float | None:
    """Rf=0 Sharpe = mean/std (sample std). Matches scripts/portfolio_sharpe_paper_session.py."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return None
    s = float(np.std(x, ddof=1))
    return float(np.mean(x) / s) if s > 0 else None


def metrics_block(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    tau: float = ENTRY_TAU,
    *,
    meta: pd.DataFrame | None = None,
    snaps_per_hour: int = 33,
) -> dict[str, Any]:
    """Score predictions. Headline trading metrics use the |pred|>tau filter."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true, y_pred = y_true[mask], y_pred[mask]
    meta_m = meta.iloc[np.where(mask)[0]].reset_index(drop=True) if meta is not None else None

    def _pack(yt, yp, meta_slice: pd.DataFrame | None) -> dict[str, Any]:
        if len(yt) == 0:
            return {
                "n": 0,
                "mae": None,
                "rmse": None,
                "r2": None,
                "diracc": None,
                "mean_pnl_proxy": None,
                "sharpe_per_trade": None,
                "sharpe_closed_hourly_A": None,
            }
        mae = float(mean_absolute_error(yt, yp))
        rmse = float(math.sqrt(mean_squared_error(yt, yp)))
        r2 = float(r2_score(yt, yp)) if len(yt) > 1 else None
        sign_ok = np.sign(yp) == np.sign(yt)
        nz = (yp != 0) & (yt != 0)
        diracc = float(sign_ok[nz].mean()) if nz.any() else None
        direction = np.sign(yp)
        pnl = direction * yt
        sharpe_pt = sharpe_ratio(pnl)

        # Closed-only pseudo-hourly Sharpe A: sum pnl by (window_id, hour_bucket)
        # ~33 snaps/hour at ~110s cadence (docs). Not live portfolio Sharpe B (no open MTM).
        sharpe_h = None
        if meta_slice is not None and len(meta_slice) == len(pnl):
            hour_key = (
                meta_slice["window_id"].astype(str)
                + "|"
                + (meta_slice["snapshot_idx"].astype(int) // max(1, snaps_per_hour)).astype(str)
            )
            hourly = pd.Series(pnl, index=hour_key.values).groupby(level=0).sum().to_numpy()
            sharpe_h = sharpe_ratio(hourly)

        return {
            "n": int(len(yt)),
            "mae": mae,
            "rmse": rmse,
            "r2": r2,
            "diracc": diracc,
            "mean_pnl_proxy": float(pnl.mean()),
            "sharpe_per_trade": sharpe_pt,
            "sharpe_closed_hourly_A": sharpe_h,
        }

    traded = np.abs(y_pred) > tau
    meta_f = meta_m.loc[traded].reset_index(drop=True) if meta_m is not None else None
    all_m = _pack(y_true, y_pred, meta_m)
    filt_m = _pack(y_true[traded], y_pred[traded], meta_f)
    headline = {
        "filter": f"|pred| > {tau}",
        "n": filt_m["n"],
        "diracc": filt_m["diracc"],
        "r2": filt_m["r2"],
        "mean_pnl_proxy": filt_m["mean_pnl_proxy"],
        "sharpe_per_trade": filt_m["sharpe_per_trade"],
        "sharpe_closed_hourly_A": filt_m["sharpe_closed_hourly_A"],
    }
    return {"all": all_m, "filtered": filt_m, "headline_filtered": headline}


def evaluate_model_and_naive(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    z_now: np.ndarray,
    tau: float = ENTRY_TAU,
    *,
    meta: pd.DataFrame | None = None,
) -> dict[str, Any]:
    return {
        "lstm": metrics_block(y_true, y_pred, tau=tau, meta=meta),
        "naive_zt": metrics_block(y_true, z_now, tau=tau, meta=meta),
        "notes": {
            "primary_slice": f"|pred| > {tau} (and |z_t| > {tau} for naive)",
            "primary_metrics": [
                "diracc",
                "r2",
                "mean_pnl_proxy",
                "sharpe_per_trade",
                "sharpe_closed_hourly_A",
            ],
            "sharpe_closed_hourly_A": (
                "sum filtered pnl_proxy by (window_id, snapshot_idx//33); "
                "Rf=0 mean/std. Offline closed-only proxy (Sharpe A family). "
                "Not live hourly portfolio Sharpe B with open MTM."
            ),
            "pnl_proxy": "sign(pred) * z_{t+1}",
        },
    }


# ── PyTorch model ─────────────────────────────────────────────────────────────

def get_torch():
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    return torch, nn, DataLoader, TensorDataset


class ZScoreLSTM:
    """Thin wrapper so notebook can construct without importing torch at module import time."""

    def __init__(
        self,
        n_channels: int,
        n_coins: int,
        n_pairs: int,
        cfg: TrainConfig | None = None,
    ):
        torch, nn, _, _ = get_torch()
        cfg = cfg or TrainConfig()
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        class _Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.coin_emb = nn.Embedding(n_coins, cfg.emb_dim)
                self.pair_emb = nn.Embedding(n_pairs, cfg.emb_dim)
                self.lstm = nn.LSTM(
                    input_size=n_channels,
                    hidden_size=cfg.hidden_size,
                    num_layers=cfg.num_layers,
                    dropout=cfg.dropout if cfg.num_layers > 1 else 0.0,
                    batch_first=True,
                )
                self.head = nn.Sequential(
                    nn.Dropout(cfg.dropout),
                    nn.Linear(cfg.hidden_size + 2 * cfg.emb_dim, 1),
                )

            def forward(self, x, coin_id, pair_id):
                out, _ = self.lstm(x)
                h = out[:, -1, :]
                e = torch.cat([self.coin_emb(coin_id), self.pair_emb(pair_id)], dim=-1)
                return self.head(torch.cat([h, e], dim=-1)).squeeze(-1)

        self.model = _Net().to(self.device)

    def fit(
        self,
        X_train,
        y_train,
        coin_train,
        pair_train,
        X_val,
        y_val,
        coin_val,
        pair_val,
        checkpoint_path: Path | None = None,
    ) -> dict[str, list[float]]:
        torch, nn, DataLoader, TensorDataset = get_torch()
        cfg = self.cfg
        set_seed(cfg.seed)

        def _ds(X, y, c, p):
            return TensorDataset(
                torch.from_numpy(np.ascontiguousarray(X)),
                torch.from_numpy(np.ascontiguousarray(y)),
                torch.from_numpy(np.ascontiguousarray(c)),
                torch.from_numpy(np.ascontiguousarray(p)),
            )

        train_loader = DataLoader(
            _ds(X_train, y_train, coin_train, pair_train),
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=cfg.num_workers,
        )
        val_loader = DataLoader(
            _ds(X_val, y_val, coin_val, pair_val),
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
        )

        opt = torch.optim.Adam(
            self.model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
        )
        loss_fn = nn.MSELoss()
        history = {"train_mse": [], "val_mse": [], "val_rmse": []}
        best_rmse = float("inf")
        bad_epochs = 0
        best_state = None

        for epoch in range(1, cfg.max_epochs + 1):
            self.model.train()
            train_losses = []
            for xb, yb, cb, pb in train_loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                cb = cb.to(self.device)
                pb = pb.to(self.device)
                opt.zero_grad()
                pred = self.model(xb, cb, pb)
                loss = loss_fn(pred, yb)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                opt.step()
                train_losses.append(float(loss.item()))

            self.model.eval()
            val_losses = []
            with torch.no_grad():
                for xb, yb, cb, pb in val_loader:
                    xb = xb.to(self.device)
                    yb = yb.to(self.device)
                    cb = cb.to(self.device)
                    pb = pb.to(self.device)
                    pred = self.model(xb, cb, pb)
                    val_losses.append(float(loss_fn(pred, yb).item()))

            tr = float(np.mean(train_losses)) if train_losses else float("nan")
            va = float(np.mean(val_losses)) if val_losses else float("nan")
            rmse = math.sqrt(va) if np.isfinite(va) else float("inf")
            history["train_mse"].append(tr)
            history["val_mse"].append(va)
            history["val_rmse"].append(rmse)
            print(f"epoch {epoch:03d}  train_mse={tr:.5f}  val_mse={va:.5f}  val_rmse={rmse:.5f}")

            if rmse < best_rmse - 1e-6:
                best_rmse = rmse
                bad_epochs = 0
                best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
                if checkpoint_path is not None:
                    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(best_state, checkpoint_path)
            else:
                bad_epochs += 1
                if bad_epochs >= cfg.patience:
                    print(f"early stop at epoch {epoch} (best val_rmse={best_rmse:.5f})")
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)
        return history

    def predict(self, X, coin_id, pair_id, batch_size: int | None = None) -> np.ndarray:
        torch, _, DataLoader, TensorDataset = get_torch()
        batch_size = batch_size or self.cfg.batch_size
        ds = TensorDataset(
            torch.from_numpy(np.ascontiguousarray(X)),
            torch.from_numpy(np.ascontiguousarray(coin_id)),
            torch.from_numpy(np.ascontiguousarray(pair_id)),
        )
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=self.cfg.num_workers)
        preds = []
        self.model.eval()
        with torch.no_grad():
            for xb, cb, pb in loader:
                xb = xb.to(self.device)
                cb = cb.to(self.device)
                pb = pb.to(self.device)
                preds.append(self.model(xb, cb, pb).cpu().numpy())
        if not preds:
            return np.zeros((0,), dtype=np.float32)
        return np.concatenate(preds).astype(np.float32)


def export_artifacts(
    *,
    output_dir: Path,
    model: ZScoreLSTM,
    scaler: StandardScaler,
    coin_to_id: dict[str, int],
    pair_to_id: dict[str, int],
    channel_names: list[str],
    cfg: TrainConfig,
    metrics: dict[str, Any],
    winsor_bounds: dict[str, tuple[float, float]],
) -> None:
    torch, _, _, _ = get_torch()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = output_dir / "statarb_lstm.pt"
    torch.save(
        {
            "model_state": model.model.state_dict(),
            "n_channels": len(channel_names),
            "n_coins": len(coin_to_id),
            "n_pairs": len(pair_to_id),
            "cfg": asdict(cfg),
        },
        ckpt_path,
    )

    schema = {
        "protocol": {
            "horizon": HORIZON,
            "zscore_window": ZSCORE_WINDOW,
            "min_periods": MIN_PERIODS,
            "seq_len": cfg.seq_len,
            "entry_tau": cfg.entry_tau,
            "top_exchanges": TOP_EXCHANGES,
        },
        "channel_names": channel_names,
        "volume_channels_log1p": VOLUME_CHANNELS,
        "winsor_bounds": {k: {"lo": v[0], "hi": v[1]} for k, v in winsor_bounds.items()},
        "scaler": {
            "mean": scaler.mean_.tolist(),
            "scale": scaler.scale_.tolist(),
        },
        "coin_to_id": coin_to_id,
        "pair_to_id": pair_to_id,
        "literature_notes": {
            "tsoku_makatjane_2026": "MSE z/spread regression + dual metrics",
            "han_li_2024": "PyTorch LSTM; |pred|>0.5 as abstention filter",
            "sheng_ma_2022": "2-layer LSTM + Adam defaults / dual R2+trading table",
        },
    }
    (output_dir / "feature_schema.json").write_text(json.dumps(schema, indent=2), encoding="utf-8")
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    def _fmt(block: dict | None) -> str:
        if not block or block.get("n", 0) == 0:
            return "_no samples_"

        def _f(key: str, pct: bool = False) -> str:
            v = block.get(key)
            if v is None:
                return "n/a"
            return f"{100 * v:.2f}%" if pct else f"{v:.4f}"

        return (
            f"n={block['n']}, DirAcc={_f('diracc', True)}, R2={_f('r2')}, "
            f"mean_pnl_proxy={_f('mean_pnl_proxy')}, "
            f"sharpe_per_trade={_f('sharpe_per_trade')}, "
            f"sharpe_closed_hourly_A={_f('sharpe_closed_hourly_A')}"
        )

    h_lstm = metrics["lstm"].get("headline_filtered", metrics["lstm"]["filtered"])
    h_naive = metrics["naive_zt"].get("headline_filtered", metrics["naive_zt"]["filtered"])

    md = f"""# LSTM Z-Score Metrics

Protocol: predict z_(t+1) with W={ZSCORE_WINDOW}, min_periods={MIN_PERIODS}, SEQ_LEN={cfg.seq_len}.
**Primary slice:** `|pred| > {cfg.entry_tau}` (same filter axis as LGBM campaigns).
Split: train < Jul 25 / test Jul 25-28 (`snapshot_idx` cut 3584).

Features are **LSTM-native** (pair-leg microstructure sequences), not the 68 LGBM columns.

## Headline (filtered) — compare to LGBM

| Model | n | DirAcc | R2 | mean pnl_proxy | Sharpe (per-trade) | Sharpe A (closed hourly) |
|---|---:|---:|---:|---:|---:|---:|
| LSTM | {h_lstm.get('n')} | {h_lstm.get('diracc')} | {h_lstm.get('r2')} | {h_lstm.get('mean_pnl_proxy')} | {h_lstm.get('sharpe_per_trade')} | {h_lstm.get('sharpe_closed_hourly_A')} |
| Naive z_t | {h_naive.get('n')} | {h_naive.get('diracc')} | {h_naive.get('r2')} | {h_naive.get('mean_pnl_proxy')} | {h_naive.get('sharpe_per_trade')} | {h_naive.get('sharpe_closed_hourly_A')} |

Definitions (aligned with live LGBM reporting):
- `pnl_proxy = sign(pred) * z_(t+1)`
- DirAcc / R2 / mean pnl_proxy reported on **filtered** rows
- `sharpe_per_trade = mean(pnl_proxy) / std(pnl_proxy)` on filtered trades (Rf=0)
- `sharpe_closed_hourly_A` = mean/std of **summed** filtered pnl_proxy per `(window_id, snapshot_idx//33)` bucket (closed-only; offline proxy of Sharpe A). **Not** live portfolio Sharpe B with open MTM.

## Full slices

### LSTM
- All: {_fmt(metrics['lstm']['all'])}
- Filtered: {_fmt(metrics['lstm']['filtered'])}

### Naive z_t -> z_(t+1)
- All: {_fmt(metrics['naive_zt']['all'])}
- Filtered: {_fmt(metrics['naive_zt']['filtered'])}

## Literature adaptations

- **Tsoku & Makatjane (2026):** MSE regression of standardized spread/z; report forecast + trading metrics.
- **Han & Li (2024):** PyTorch LSTM; `|pred|>0.5` as trade abstention filter (not their 3-class trend head).
- **Sheng & Ma (2022)** (repo notes mis-cite as Shen): 2-layer LSTM + Adam; dual error/trading table.

## Artifacts

- `statarb_lstm.pt`
- `feature_schema.json`
- `metrics.json`
"""
    (output_dir / "METRICS.md").write_text(md, encoding="utf-8")
    print(f"Exported artifacts -> {output_dir}")
