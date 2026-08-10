#!/usr/bin/env python3
"""Random Forest z-score baseline vs LightGBM (paper Jul-25 protocol).

Role: classical tabular peer (like LSTM is the deep peer). LightGBM stays the
deployed paper-trading head.

Shared contract (must match paper / LOGO cache):
  target y = z_{t+1}, W=300, min_periods=90, H=1, N_LAGS=3
  train = pre Jul 25 · test = Jul 25–28
  metrics: DirAcc, R², mean pnl_proxy = sign(pred) * y
  gates: all-rows + tau in {0.5, 0.9}  (paper headline uses |pred| >= 0.9)

Prefer LOGO cache frames so feature engineering is not a confounder:
  statarb/outputs_logo/cache/df_{train,test}_logo.parquet

Usage:
  .\\.venv\\Scripts\\python.exe statarb\\run_rf_zscore_baseline.py --smoke
  .\\.venv\\Scripts\\python.exe statarb\\run_rf_zscore_baseline.py --out-dir statarb\\outputs_rf
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path
from typing import Any

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import OrdinalEncoder

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path[:0] = [str(HERE), str(ROOT)]

from run_logo_ablation import (  # noqa: E402
    ID_COLS,
    PUBLISHED_FEATURES,
    build_feature_frames,
)

DEFAULT_CACHE = HERE / "outputs_logo" / "cache"
DEFAULT_OUT = HERE / "outputs_rf"
DEFAULT_LGBM = HERE / "outputs" / "statarb_lgbm.txt"
DEFAULT_TAUS = (0.5, 0.9)

# Paper-spirited RF defaults (regularization analogous to LGBM min_child_samples=200).
# n_jobs is capped: sklearn RF + n_jobs=-1 on many-core boxes multiplies RAM hard.
RF_DEFAULTS = dict(
    n_estimators=400,
    max_depth=20,
    min_samples_leaf=200,
    max_features="sqrt",
    n_jobs=4,
    random_state=42,
)
# Keep the dense X matrix itself small; max_samples alone is NOT enough (sklearn still
# materializes the full passed array).
DEFAULT_TRAIN_MAX_ROWS = 1_000_000
DEFAULT_MAX_SAMPLES = 500_000
# Refuse to start fit if estimated peak > free RAM * this fraction (unless --force-mem).
MEM_SAFETY_FRAC = 0.65


def _json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        x = float(obj)
        return None if not np.isfinite(x) else x
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return _json_safe(obj.tolist())
    return obj


def score_predictions(
    y: np.ndarray,
    pred: np.ndarray,
    *,
    tau: float | None = None,
    ge: bool = True,
) -> dict[str, Any]:
    """DirAcc / R² / mean pnl_proxy. If tau set, filter on |pred| >= (or >) tau."""
    y = np.asarray(y, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    mask = np.isfinite(y) & np.isfinite(pred)
    y, pred = y[mask], pred[mask]
    if tau is not None:
        traded = np.abs(pred) >= tau if ge else np.abs(pred) > tau
        y, pred = y[traded], pred[traded]
    n = int(len(y))
    if n == 0:
        return {
            "n": 0,
            "mae": None,
            "rmse": None,
            "r2": None,
            "diracc": None,
            "mean_pnl_proxy": None,
            "sharpe_per_trade": None,
            "filter": None if tau is None else (f"|pred| >= {tau}" if ge else f"|pred| > {tau}"),
        }
    mae = float(mean_absolute_error(y, pred))
    rmse = float(np.sqrt(mean_squared_error(y, pred)))
    r2 = float(r2_score(y, pred)) if n > 1 else None
    nz = (pred != 0) & (y != 0)
    diracc = float(np.mean(np.sign(pred[nz]) == np.sign(y[nz]))) if nz.any() else None
    pnl = np.sign(pred) * y
    s = float(np.std(pnl, ddof=1)) if n > 1 else 0.0
    sharpe = float(np.mean(pnl) / s) if s > 0 else None
    return {
        "n": n,
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "diracc": diracc,
        "mean_pnl_proxy": float(np.mean(pnl)),
        "sharpe_per_trade": sharpe,
        "filter": None if tau is None else (f"|pred| >= {tau}" if ge else f"|pred| > {tau}"),
    }


def score_suite(y: np.ndarray, pred: np.ndarray, taus: list[float]) -> dict[str, Any]:
    out: dict[str, Any] = {"all": score_predictions(y, pred, tau=None)}
    for tau in taus:
        key = f"tau_{str(tau).replace('.', 'p')}"
        out[key] = score_predictions(y, pred, tau=tau, ge=True)
    return out


def feature_columns(df: pd.DataFrame, prefer_published: bool = True) -> list[str]:
    feat = [c for c in df.columns if c not in ID_COLS]
    if prefer_published:
        published = [c for c in PUBLISHED_FEATURES if c in feat]
        if len(published) >= 50:
            return published
    return feat


def prepare_matrix(
    df: pd.DataFrame,
    feat_cols: list[str],
    *,
    encoder: OrdinalEncoder | None = None,
    medians: pd.Series | None = None,
    fit: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, OrdinalEncoder, pd.Series, list[str]]:
    """Return X_num, y, z_now, encoder, medians, cat_cols.

    RF cannot take LGBM categorical dtype; OrdinalEncoder is fit on train only.
    Missing numeric cells filled with train medians.
    """
    work = df.dropna(subset=["target"]).copy()
    y = work["target"].to_numpy(dtype=np.float64)
    z_now = work["zscore"].to_numpy(dtype=np.float64) if "zscore" in work.columns else np.full(len(work), np.nan)

    cat_cols = [c for c in ("coin", "pair") if c in feat_cols]
    num_cols = [c for c in feat_cols if c not in cat_cols]

    X_cat = work[cat_cols].astype(str).fillna("__NA__") if cat_cols else pd.DataFrame(index=work.index)
    X_num = work[num_cols].apply(pd.to_numeric, errors="coerce")

    if fit:
        encoder = OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,
            encoded_missing_value=-1,
        )
        if cat_cols:
            encoder.fit(X_cat)
        medians = X_num.median(numeric_only=True)
    assert encoder is not None and medians is not None

    if cat_cols:
        cat_enc = encoder.transform(X_cat)
    else:
        cat_enc = np.zeros((len(work), 0), dtype=np.float64)

    X_num = X_num.fillna(medians)
    # Align any columns that were all-NaN on train
    for c in num_cols:
        if c not in medians.index or pd.isna(medians[c]):
            X_num[c] = X_num[c].fillna(0.0)
        else:
            X_num[c] = X_num[c].fillna(float(medians[c]))

    X = np.hstack([cat_enc.astype(np.float64), X_num.to_numpy(dtype=np.float64)])
    return X, y, z_now, encoder, medians, cat_cols


def predict_lgbm_on_frame(
    df: pd.DataFrame,
    model: lgb.Booster,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Score production booster on LOGO-style frame (missing cols → NaN)."""
    work = df.dropna(subset=["target"]).copy()
    y = work["target"].to_numpy(dtype=np.float64)
    z_now = work["zscore"].to_numpy(dtype=np.float64)
    feat_names = model.feature_name()
    cat_maps = list(model.pandas_categorical or [])
    X = work.reindex(columns=feat_names).copy()
    if "coin" in X.columns and len(cat_maps) >= 1:
        X["coin"] = pd.Categorical(X["coin"], categories=cat_maps[0])
    if "pair" in X.columns and len(cat_maps) >= 2:
        X["pair"] = pd.Categorical(X["pair"], categories=cat_maps[1])
    pred = np.asarray(model.predict(X), dtype=np.float64)
    return y, pred, z_now


def maybe_subsample(df: pd.DataFrame, max_rows: int | None, *, keep_tail: bool, seed: int) -> pd.DataFrame:
    if max_rows is None or len(df) <= max_rows:
        return df
    if keep_tail:
        return df.iloc[-max_rows:].copy()
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(df), size=max_rows, replace=False))
    return df.iloc[idx].copy()


def _bytes_to_gb(n: float) -> float:
    return float(n) / (1024.0 ** 3)


def system_ram_gb() -> tuple[float, float]:
    """Return (total_gb, free_gb). Best-effort across platforms."""
    try:
        import psutil  # type: ignore

        vm = psutil.virtual_memory()
        return _bytes_to_gb(vm.total), _bytes_to_gb(vm.available)
    except Exception:
        pass
    if sys.platform == "win32":
        try:
            import ctypes

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
            return _bytes_to_gb(stat.ullTotalPhys), _bytes_to_gb(stat.ullAvailPhys)
        except Exception:
            pass
    return float("nan"), float("nan")


def resolve_bootstrap_rows(n_train: int, max_samples: float | int | None) -> int:
    if max_samples is None:
        return int(n_train)
    if isinstance(max_samples, float) and max_samples <= 1.0:
        return max(1, int(n_train * max_samples))
    return max(1, min(int(max_samples), int(n_train)))


def estimate_rf_peak_gb(
    *,
    n_train: int,
    n_test: int,
    n_features: int,
    n_jobs: int,
    max_samples: float | int | None,
    hold_test_during_fit: bool = False,
) -> dict[str, float]:
    """Order-of-magnitude peak RAM for sklearn RF fit on float64 X.

    Dominant terms:
      - dense X_train (always held)
      - optional X_test if built before fit
      - per-worker bootstrap copy ≈ max_samples * n_features * 8
      - tree / allocator overhead fudge
    """
    item = 8.0  # float64
    x_train = n_train * n_features * item
    x_test = n_test * n_features * item
    boot = resolve_bootstrap_rows(n_train, max_samples) * n_features * item
    jobs = max(1, int(n_jobs) if int(n_jobs) > 0 else 4)
    # joblib/loky often keeps a bootstrap-sized working set per worker
    workers = jobs * boot
    # y, medians, encoder temporaries, fragmentation
    fudge = 0.35 * (x_train + (x_test if hold_test_during_fit else 0.0) + workers)
    peak = x_train + (x_test if hold_test_during_fit else 0.0) + workers + fudge
    return {
        "x_train_gb": _bytes_to_gb(x_train),
        "x_test_gb": _bytes_to_gb(x_test),
        "bootstrap_per_worker_gb": _bytes_to_gb(boot),
        "workers_gb": _bytes_to_gb(workers),
        "fudge_gb": _bytes_to_gb(fudge),
        "peak_est_gb": _bytes_to_gb(peak),
        "n_jobs_effective": float(jobs),
        "bootstrap_rows": float(resolve_bootstrap_rows(n_train, max_samples)),
    }


def assert_memory_budget(
    est: dict[str, float],
    *,
    force: bool,
    safety_frac: float = MEM_SAFETY_FRAC,
) -> None:
    total_gb, free_gb = system_ram_gb()
    peak = est["peak_est_gb"]
    budget = free_gb * safety_frac if np.isfinite(free_gb) else float("nan")
    print(
        "Memory budget check:\n"
        f"  system total={total_gb:.1f} GiB  free/available={free_gb:.1f} GiB\n"
        f"  est X_train={est['x_train_gb']:.2f}  bootstrap/worker={est['bootstrap_per_worker_gb']:.2f}  "
        f"workers({int(est['n_jobs_effective'])})={est['workers_gb']:.2f}  "
        f"fudge={est['fudge_gb']:.2f}\n"
        f"  peak_est={peak:.2f} GiB  budget={budget:.2f} GiB "
        f"(safety_frac={safety_frac:.2f} of free)"
    )
    if not np.isfinite(free_gb):
        print("  [WARN] could not read free RAM; proceeding cautiously")
        return
    if peak > budget:
        msg = (
            f"Refusing RF fit: estimated peak {peak:.2f} GiB exceeds "
            f"{safety_frac:.0%} of free RAM ({budget:.2f} GiB). "
            "Shrink with --train-max-rows / --max-samples / --n-jobs, "
            "or pass --force-mem to override."
        )
        if force:
            print(f"  [WARN] {msg} (--force-mem set; continuing)")
            return
        raise SystemExit(msg)


def fmt_pct(x: float | None) -> str:
    return "—" if x is None else f"{100.0 * x:.1f}%"


def fmt_f(x: float | None, digits: int = 3) -> str:
    return "—" if x is None else f"{x:.{digits}f}"


def write_metrics_md(
    path: Path,
    *,
    rows: list[dict[str, Any]],
    protocol: dict[str, Any],
    resources: dict[str, Any],
    paper_ref: dict[str, Any],
) -> None:
    lines = [
        "# Random Forest vs LightGBM (paper protocol)",
        "",
        "Classical tabular baseline on the **same LOGO / Jul-25 feature frames** as LightGBM.",
        "RF is a literature peer — not a live replacement. Production head remains LightGBM.",
        "",
        "## Protocol",
        "",
        f"- Target: `y = z_{{t+1}}` · W={protocol.get('zscore_window', 300)} · H={protocol.get('horizon', 1)} · N_LAGS={protocol.get('n_lags', 3)}",
        f"- Split: train pre Jul 25 · test Jul 25–28 (LOGO cache)",
        f"- Features: {protocol.get('n_features')} tabular cols (published intersection on cache)",
        f"- Missing vs 68-feat booster: {protocol.get('missing_published', [])}",
        f"- Gate: `|pred| >= tau` (paper headline τ=0.9); also report all-rows and τ=0.5",
        f"- Naive peer: `ẑ ← z_t` on identical rows",
        "",
        "## Comparison table",
        "",
        "| Model | Set | Filter | n | DirAcc | R² | mean pnl_proxy |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['model']} | {r['set']} | {r['filter']} | {r['n']} | "
            f"{fmt_pct(r.get('diracc'))} | {fmt_f(r.get('r2'))} | {fmt_f(r.get('mean_pnl_proxy'))} |"
        )

    lines += [
        "",
        "### Paper reference cells (LGBM validation live book)",
        "",
        "| Model | Set | Filter | n | DirAcc | R² | mean pnl_proxy |",
        "|---|---|---|---:|---:|---:|---:|",
        f"| LightGBM (paper / tau09 report) | validation | `|ẑ|≥0.9` | "
        f"{paper_ref.get('n', 12795)} | {fmt_pct(paper_ref.get('diracc', 0.867))} | "
        f"{fmt_f(paper_ref.get('r2', 0.599))} | {fmt_f(paper_ref.get('mean_pnl', 1.372))} |",
        "",
        "## Resources",
        "",
        f"- Backend: `{resources.get('backend', 'sklearn.RandomForestRegressor')}` (CPU)",
        f"- Fit wall-clock: {resources.get('fit_seconds')} s",
        f"- Predict (test) wall-clock: {resources.get('predict_seconds')} s",
        f"- Train rows used: {resources.get('n_train_used')} / available {resources.get('n_train_available')}",
        f"- Test rows: {resources.get('n_test')}",
        f"- RF params: `{json.dumps(resources.get('rf_params', {}), sort_keys=True)}`",
        f"- Peak note: {resources.get('note', 'CPU multi-core; no CUDA required.')}",
        "",
        "## Artifacts",
        "",
        "- `rf_model.joblib` — fitted RandomForestRegressor",
        "- `encoder.joblib` — OrdinalEncoder for coin/pair (+ medians)",
        "- `metrics_test.json` — full metric block",
        "- `METRICS.md` — this file",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def load_paper_tau09_ref(path: Path | None) -> dict[str, Any]:
    default = {
        "n": 12795,
        "diracc": 0.8671355998436889,
        "r2": 0.5990371393573842,
        "mean_pnl": 1.3722711889530343,
        "source": "paper / data/paper_trading/5day_Aug4_2026/tau09_w300_report.json",
    }
    if path is None or not path.exists():
        return default
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
        for row in doc.get("grid", []):
            if abs(float(row.get("tau", -1)) - 0.9) < 1e-9:
                return {
                    "n": int(row["n"]),
                    "diracc": float(row["diracc"]),
                    "r2": float(row["r2_pred_vs_exit_z"]),
                    "mean_pnl": float(row["mean_pnl"]),
                    "source": str(path),
                }
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        pass
    return default


def build_table_rows(
    *,
    set_name: str,
    suites: dict[str, dict[str, Any]],
    taus: list[float],
) -> list[dict[str, Any]]:
    """Flatten model→suite into METRICS table rows (all + each tau)."""
    rows: list[dict[str, Any]] = []
    order = ["LightGBM", "Random Forest", "Naive z_t"]
    for model in order:
        suite = suites[model]
        all_m = suite["all"]
        rows.append(
            {
                "model": model,
                "set": set_name,
                "filter": "all",
                "n": all_m["n"],
                "diracc": all_m["diracc"],
                "r2": all_m["r2"],
                "mean_pnl_proxy": all_m["mean_pnl_proxy"],
            }
        )
        for tau in taus:
            key = f"tau_{str(tau).replace('.', 'p')}"
            m = suite[key]
            rows.append(
                {
                    "model": model,
                    "set": set_name,
                    "filter": f"|ẑ|≥{tau}",
                    "n": m["n"],
                    "diracc": m["diracc"],
                    "r2": m["r2"],
                    "mean_pnl_proxy": m["mean_pnl_proxy"],
                }
            )
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--lgbm-model", type=Path, default=DEFAULT_LGBM)
    ap.add_argument("--force-rebuild-cache", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="200k train / 50 trees plumbing check")
    ap.add_argument(
        "--train-max-rows",
        type=int,
        default=None,
        help=f"Cap train rows chronological tail (default {DEFAULT_TRAIN_MAX_ROWS}; 0 = use all)",
    )
    ap.add_argument("--test-max-rows", type=int, default=None, help="Cap test rows (random subsample)")
    ap.add_argument("--n-estimators", type=int, default=None)
    ap.add_argument("--max-depth", type=int, default=None)
    ap.add_argument("--min-samples-leaf", type=int, default=None)
    ap.add_argument(
        "--n-jobs",
        type=int,
        default=None,
        help="RF parallel workers (default 4). Avoid -1 on high-core boxes — RAM multiplies.",
    )
    ap.add_argument(
        "--max-samples",
        type=float,
        default=None,
        help="RF bootstrap sample size (int) or fraction (float in (0,1]).",
    )
    ap.add_argument(
        "--force-mem",
        action="store_true",
        help="Override pre-fit memory budget refusal",
    )
    ap.add_argument(
        "--mem-check-only",
        action="store_true",
        help="Print memory estimate for the chosen caps and exit (no fit)",
    )
    ap.add_argument("--tau", type=float, nargs="+", default=list(DEFAULT_TAUS))
    ap.add_argument(
        "--tau09-report",
        type=Path,
        default=ROOT / "data" / "paper_trading" / "5day_Aug4_2026" / "tau09_w300_report.json",
    )
    ap.add_argument(
        "--skip-lgbm",
        action="store_true",
        help="Skip production booster scoring (RF + naive only)",
    )
    ap.add_argument("--skip-fit", action="store_true", help="Load rf_model.joblib from out-dir")
    args = ap.parse_args()

    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    taus = [float(t) for t in args.tau]

    rf_params = dict(RF_DEFAULTS)
    train_max = args.train_max_rows
    test_max = args.test_max_rows
    if args.smoke:
        rf_params.update(n_estimators=50, max_depth=12, min_samples_leaf=100, n_jobs=2)
        train_max = 200_000 if train_max is None else train_max
        test_max = test_max or 100_000
    else:
        # Default caps: full 5M×62 with n_jobs=-1 OOMs a 32 GiB box.
        if train_max is None:
            train_max = DEFAULT_TRAIN_MAX_ROWS
        elif train_max == 0:
            train_max = None  # explicit "use all"
    if args.n_estimators is not None:
        rf_params["n_estimators"] = args.n_estimators
    if args.max_depth is not None:
        rf_params["max_depth"] = args.max_depth
    if args.min_samples_leaf is not None:
        rf_params["min_samples_leaf"] = args.min_samples_leaf
    if args.n_jobs is not None:
        rf_params["n_jobs"] = args.n_jobs
    if args.max_samples is not None:
        ms = args.max_samples
        rf_params["max_samples"] = int(ms) if ms >= 1.0 else float(ms)
    elif not args.smoke:
        rf_params["max_samples"] = DEFAULT_MAX_SAMPLES

    # Load frames (reuse LOGO builder / cache)
    cache_train = args.cache_dir / "df_train_logo.parquet"
    cache_test = args.cache_dir / "df_test_logo.parquet"
    if args.force_rebuild_cache or not (cache_train.exists() and cache_test.exists()):
        print("Building / refreshing LOGO feature frames ...")
        # build_feature_frames writes to outputs_logo/cache; if custom cache-dir, copy after
        df_train, df_test, meta = build_feature_frames(force=args.force_rebuild_cache)
        if args.cache_dir.resolve() != DEFAULT_CACHE.resolve():
            args.cache_dir.mkdir(parents=True, exist_ok=True)
            df_train.to_parquet(cache_train, index=False)
            df_test.to_parquet(cache_test, index=False)
            (args.cache_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    else:
        print(f"Loading cached frames from {args.cache_dir}")
        df_train = pd.read_parquet(cache_train)
        df_test = pd.read_parquet(cache_test)
        meta_path = args.cache_dir / "meta.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}

    n_train_available = len(df_train)
    n_test_available = len(df_test)
    print(f"cache shapes train={df_train.shape} test={df_test.shape}")

    feat_cols = feature_columns(df_train)
    missing_published = [c for c in PUBLISHED_FEATURES if c not in feat_cols]
    print(f"features={len(feat_cols)}  missing_vs_published_68={missing_published}")

    df_train_fit = maybe_subsample(df_train, train_max, keep_tail=True, seed=42)
    df_test_eval = maybe_subsample(df_test, test_max, keep_tail=False, seed=42)
    n_train_used = int(len(df_train_fit))
    n_test_eval = int(len(df_test_eval))
    # Drop full frames early — LOGO cache is multi-GB in pandas.
    del df_train, df_test
    print(f"fit rows={n_train_used:,}  eval rows={n_test_eval:,}")

    n_feat = len(feat_cols)
    est = estimate_rf_peak_gb(
        n_train=n_train_used,
        n_test=n_test_eval,
        n_features=n_feat,
        n_jobs=int(rf_params.get("n_jobs", 4)),
        max_samples=rf_params.get("max_samples"),
        hold_test_during_fit=False,
    )
    assert_memory_budget(est, force=bool(args.force_mem))
    if args.mem_check_only:
        print("mem-check-only: exiting before fit")
        return

    # ── RF fit / load ────────────────────────────────────────────────────────
    enc_path = out / "encoder.joblib"
    model_path = out / "rf_model.joblib"
    t_fit = 0.0
    if args.skip_fit and model_path.exists() and enc_path.exists():
        print(f"Loading RF + encoder from {out}")
        bundle = joblib.load(enc_path)
        encoder, medians, cat_cols, feat_cols = (
            bundle["encoder"],
            bundle["medians"],
            bundle["cat_cols"],
            bundle["feat_cols"],
        )
        rf = joblib.load(model_path)
        X_te, y_te, z_te, _, _, _ = prepare_matrix(
            df_test_eval, feat_cols, encoder=encoder, medians=medians, fit=False
        )
    else:
        X_tr, y_tr, _, encoder, medians, cat_cols = prepare_matrix(
            df_train_fit, feat_cols, fit=True
        )
        del df_train_fit
        print(f"Fitting RandomForestRegressor {rf_params} on X={X_tr.shape} ...")
        t0 = time.time()
        rf = RandomForestRegressor(**rf_params)
        rf.fit(X_tr, y_tr)
        t_fit = time.time() - t0
        print(f"  fit done in {t_fit:.1f}s")
        del X_tr, y_tr
        joblib.dump(rf, model_path)
        joblib.dump(
            {
                "encoder": encoder,
                "medians": medians,
                "cat_cols": cat_cols,
                "feat_cols": feat_cols,
                "rf_params": rf_params,
            },
            enc_path,
        )
        X_te, y_te, z_te, _, _, _ = prepare_matrix(
            df_test_eval, feat_cols, encoder=encoder, medians=medians, fit=False
        )

    t0 = time.time()
    pred_rf = np.asarray(rf.predict(X_te), dtype=np.float64)
    t_pred = time.time() - t0
    print(f"  RF predict done in {t_pred:.1f}s")

    suites: dict[str, dict[str, Any]] = {
        "Random Forest": score_suite(y_te, pred_rf, taus),
        "Naive z_t": score_suite(y_te, z_te, taus),
    }

    if not args.skip_lgbm:
        if not args.lgbm_model.exists():
            raise SystemExit(f"Missing LGBM model: {args.lgbm_model}")
        print(f"Scoring LightGBM booster {args.lgbm_model} on same test rows ...")
        booster = lgb.Booster(model_file=str(args.lgbm_model))
        y_lgb, pred_lgb, z_lgb = predict_lgbm_on_frame(df_test_eval, booster)
        # Sanity: y should match RF y (same dropna target rows, same subsample order)
        if len(y_lgb) != len(y_te) or not np.allclose(y_lgb, y_te, equal_nan=True):
            print("  [WARN] LGBM y alignment differs from RF frame; reporting on LGBM's own row set")
        suites["LightGBM"] = score_suite(y_lgb, pred_lgb, taus)
        # Matched-row naive already in suites; keep LGBM-native z for transparency
        _ = z_lgb
    else:
        suites["LightGBM"] = {
            "all": score_predictions(np.array([]), np.array([])),
            **{f"tau_{str(t).replace('.', 'p')}": score_predictions(np.array([]), np.array([]), tau=t) for t in taus},
        }

    # Print headline (ASCII-only for Windows cp1252 consoles)
    for name, suite in suites.items():
        a = suite["all"]
        print(
            f"{name:16s} all  n={a['n']:,}  DirAcc={fmt_pct(a['diracc'])}  "
            f"R2={fmt_f(a['r2'])}  pnl={fmt_f(a['mean_pnl_proxy'])}"
        )
        for tau in taus:
            m = suite[f"tau_{str(tau).replace('.', 'p')}"]
            print(
                f"{name:16s} tau>={tau:<4} n={m['n']:,}  DirAcc={fmt_pct(m['diracc'])}  "
                f"R2={fmt_f(m['r2'])}  pnl={fmt_f(m['mean_pnl_proxy'])}"
            )

    table_rows = build_table_rows(set_name="test", suites=suites, taus=taus)
    paper_ref = load_paper_tau09_ref(args.tau09_report)

    protocol = {
        "horizon": 1,
        "zscore_window": 300,
        "n_lags": 3,
        "n_features": len(feat_cols),
        "feat_cols": feat_cols,
        "missing_published": missing_published,
        "taus": taus,
        "filter_op": ">=",
        "cache_dir": str(args.cache_dir),
        "lgbm_model": str(args.lgbm_model),
        "smoke": bool(args.smoke),
        "meta": meta,
    }
    resources = {
        "backend": "sklearn.ensemble.RandomForestRegressor",
        "fit_seconds": round(t_fit, 2),
        "predict_seconds": round(t_pred, 2),
        "n_train_used": n_train_used,
        "n_train_available": int(n_train_available),
        "n_test": int(len(y_te)),
        "n_test_available": int(n_test_available),
        "rf_params": rf_params,
        "mem_estimate": est,
        "note": (
            "CPU only. Defaults cap train rows / n_jobs / max_samples after a 32 GiB box "
            "hit ~13 GiB private with full 4.7M X and n_jobs=-1. Use --mem-check-only first."
        ),
    }

    metrics_doc = {
        "protocol": protocol,
        "resources": resources,
        "test": suites,
        "table_rows": table_rows,
        "paper_validation_ref": paper_ref,
        "validation_note": (
            "Aug 4–7 paper validation is the live LGBM book (tau09_w300_report). "
            "RF validation requires the same campaign feature/signal panel for offline "
            "re-prediction; not fitted here. See --help / handoff for HF download steps."
        ),
    }
    (out / "metrics_test.json").write_text(
        json.dumps(_json_safe(metrics_doc), indent=2), encoding="utf-8"
    )
    write_metrics_md(
        out / "METRICS.md",
        rows=table_rows,
        protocol=protocol,
        resources=resources,
        paper_ref=paper_ref,
    )
    print(f"\nWrote {out / 'metrics_test.json'} and {out / 'METRICS.md'}")


if __name__ == "__main__":
    main()
