"""Build cex_gbm_more_data.ipynb from cex_gbm_new.ipynb (local-first + HF fallback)."""
import json
from pathlib import Path

src_path = Path(__file__).with_name("cex_gbm_new.ipynb")
dst_path = Path(__file__).with_name("cex_gbm_more_data.ipynb")
nb = json.loads(src_path.read_text(encoding="utf-8"))


def set_src(cell, text: str) -> None:
    lines = text.split("\n")
    cell["source"] = [l + "\n" for l in lines[:-1]] + ([lines[-1]] if lines[-1] != "" else [])
    cell["outputs"] = []
    cell["execution_count"] = None


for c in nb["cells"]:
    if c["cell_type"] == "markdown" and "StatArb LightGBM Pipeline" in "".join(c.get("source", [])):
        set_src(
            c,
            """# StatArb LightGBM — More Data Ablation
Branch: `experiment/alt-prediction-target`

Adds **Jul 22–28 (~160h)** into the **train** pool. Keeps **Jul 19–22** as held-out **test**.

**Data loading:** prefers local unified cache `data/cex_unified` (offline). Falls back to Hugging Face if a file is missing.

Also reports a **naive persistence baseline** (`pred = zscore_lag1`) so we judge ΔR² / ΔDirAcc.
""",
        )
        break

c3 = "".join(nb["cells"][3]["source"])
c3 = c3.replace('OUTPUT_DIR = Path("./outputs")', 'OUTPUT_DIR = Path("./outputs_more_data")')
if "LOCAL_DATA_ROOT" not in c3:
    c3 = c3.replace(
        'HF_REPO    = "SFU-fintech-AI/statarb-crypto-research"',
        '''HF_REPO    = "SFU-fintech-AI/statarb-crypto-research"
# Config-based HF fallback (older windows) — pinned stable layout
HF_REVISION = "c5c695d3cec28db8801fe6de173b3c21f3803436"
HF_PARQUET_REVISION = "main"  # jul22-28 folder on newer Hub layout

# Local-first unified parquet cache (HF-mirrored paths). Originals left in place.
_LOCAL_CANDIDATES = [
    Path(r"C:/Users/Kev/repos/stochastic-spread-modeling/data/cex_unified"),
    Path("../data/cex_unified"),
    Path("../../stochastic-spread-modeling/data/cex_unified"),
    Path("./cex_unified"),
]
LOCAL_DATA_ROOT = next((p.resolve() for p in _LOCAL_CANDIDATES if p.exists()), None)
print("LOCAL_DATA_ROOT:", LOCAL_DATA_ROOT if LOCAL_DATA_ROOT else "(none — will use Hugging Face)")''',
    )
set_src(nb["cells"][3], c3)

# Replace entire WINDOWS definition through hf_name_for with enriched windows + helpers
c6 = "".join(nb["cells"][6]["source"])

# Find start of WINDOWS and end before _SIGNAL_TABLES or after WINDOWS list
marker_start = "WINDOWS = ["
marker_end = "_SIGNAL_TABLES = ["
if marker_start not in c6 or marker_end not in c6:
    raise SystemExit("WINDOWS markers not found")

pre = c6[: c6.find(marker_start)]
post = c6[c6.find(marker_end) :]

new_windows_and_helpers = '''WINDOWS = [
    # ── Training pool ─────────────────────────────────────────────────────────
    {
        "id":           "jun13",
        "role":         "train",
        "label":        "Jun 13-16",
        "hf_prefix":    "",
        "hf_split":     "train",
        "hf_ohlcv":     "ohlcv",
        "ohlcv_schema": "flat",
        "local_dir":    "",                      # root of cex_unified
        "load_mode":    "dataset_config",
    },
    {
        "id":           "jun22",
        "role":         "train",
        "label":        "Jun 22-24",
        "hf_prefix":    "test_",
        "hf_split":     "train",
        "hf_ohlcv":     "test_ohlcv_live",
        "ohlcv_schema": "flat",
        "local_dir":    "test",
        "load_mode":    "dataset_config",
    },
    {
        "id":           "jul13",
        "role":         "train",
        "label":        "Jul 13-15 (partial)",
        "hf_prefix":    "validation_",
        "hf_split":     "train",
        "hf_ohlcv":     None,
        "ohlcv_schema": "flat",
        "local_dir":    "validation",
        "load_mode":    "dataset_config",
    },
    {
        "id":           "jul22_28",
        "role":         "train",
        "label":        "Jul 22-28 (~160h)",
        "hf_prefix":    "",
        "hf_split":     "train",
        "hf_ohlcv":     None,
        "ohlcv_schema": "flat",
        "local_dir":    "validation_jul22-28",
        "load_mode":    "parquet_dir",           # no dataset config on Hub
        "hf_parquet_dir": "validation_jul22-28",
        "hf_parquet_revision": "main",
    },
    # ── Test pool (Jul 19-22, split around power-loss outage) ─────────────────
    {
        "id":           "jul19_pre",
        "role":         "test",
        "label":        "Jul 19-22 pre-outage",
        "hf_prefix":    "jul19_22_pre_outage_",
        "hf_split":     "train",
        "hf_ohlcv":     "jul19_22_pre_outage_ohlcv_snapshot",
        "ohlcv_schema": "snapshot",
        "local_dir":    "validation_jul19-22/pre_outage",
        "load_mode":    "dataset_config",
    },
    {
        "id":           "jul19_post",
        "role":         "test",
        "label":        "Jul 19-22 post-outage",
        "hf_prefix":    "jul19_22_post_outage_",
        "hf_split":     "train",
        "hf_ohlcv":     "jul19_22_post_outage_ohlcv_snapshot",
        "ohlcv_schema": "snapshot",
        "local_dir":    "validation_jul19-22/post_outage",
        "load_mode":    "dataset_config",
    },
]

def local_parquet_path(window: dict, local_name: str) -> Path | None:
    """Return local parquet path if unified cache has it."""
    if LOCAL_DATA_ROOT is None:
        return None
    rel_dir = (window.get("local_dir") or "").strip("/\\\\")
    path = LOCAL_DATA_ROOT / rel_dir / f"{local_name}.parquet" if rel_dir else LOCAL_DATA_ROOT / f"{local_name}.parquet"
    return path if path.exists() else None

'''

c6 = pre + new_windows_and_helpers + post

# Patch load block
old_load = """        hf = hf_name_for(window, local_name)
        print(f"    loading {hf} …", end=" ")
        try:
            ds = load_dataset(HF_REPO, hf, split="train", token=HF_TOKEN)
        except Exception as e:
            print(f"[WARN: {e}]")
            result[local_name] = pd.DataFrame()
            continue

        # drop columns we never need before materialising to pandas
        drop = [c for c in ["run_id", "ts", "market", "symbol", "error"]
                if c in ds.column_names]
        if drop:
            ds = ds.remove_columns(drop)
        df = ds.to_pandas()
        del ds
        gc.collect()"""

# may already be patched from previous build — handle both
if "loading {local_name} via {load_mode}" in c6 or "local_parquet_path" in c6 and "pd.read_parquet(local_path)" in c6:
    # still replace the load try-block if older more_data patch exists
    pass

new_load = """        local_path = local_parquet_path(window, local_name)
        load_mode = window.get("load_mode", "dataset_config")
        print(f"    loading {local_name} …", end=" ")
        try:
            if local_path is not None:
                print(f"[local {local_path.name}] ", end="")
                df = pd.read_parquet(local_path)
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
                    token=HF_TOKEN,
                )
                df = pd.read_parquet(path)
            else:
                hf = hf_name_for(window, local_name)
                print(f"[hf config {hf}] ", end="")
                ds = load_dataset(
                    HF_REPO, hf, split="train", token=HF_TOKEN, revision=HF_REVISION,
                )
                drop = [c for c in ["run_id", "ts", "market", "symbol", "error"]
                        if c in ds.column_names]
                if drop:
                    ds = ds.remove_columns(drop)
                df = ds.to_pandas()
                del ds
                gc.collect()
        except Exception as e:
            print(f"[WARN: {e}]")
            result[local_name] = pd.DataFrame()
            continue

        # normalize error filter for parquet loads
        if "error" in df.columns:
            df = df[df["error"].isna()].drop(columns=["error"], errors="ignore")
        drop_cols = [c for c in ["run_id", "ts", "market", "symbol"] if c in df.columns]
        if drop_cols:
            df = df.drop(columns=drop_cols)
        gc.collect()"""

# Try previous patched block first, then original
old_load_alt = """        load_mode = window.get("load_mode", "dataset_config")
        print(f"    loading {local_name} via {load_mode} …", end=" ")
        try:
            if load_mode == "parquet_dir":
                from huggingface_hub import hf_hub_download
                rel = f"{window['hf_parquet_dir'].rstrip('/')}/{local_name}.parquet"
                rev = window.get("hf_parquet_revision", HF_PARQUET_REVISION)
                path = hf_hub_download(
                    repo_id=HF_REPO,
                    repo_type="dataset",
                    filename=rel,
                    revision=rev,
                    token=HF_TOKEN,
                )
                df = pd.read_parquet(path)
            else:
                hf = hf_name_for(window, local_name)
                print(f"[{hf}] ", end="")
                ds = load_dataset(
                    HF_REPO, hf, split="train", token=HF_TOKEN, revision=HF_REVISION,
                )
                drop = [c for c in ["run_id", "ts", "market", "symbol", "error"]
                        if c in ds.column_names]
                if drop:
                    ds = ds.remove_columns(drop)
                df = ds.to_pandas()
                del ds
                gc.collect()
        except Exception as e:
            print(f"[WARN: {e}]")
            result[local_name] = pd.DataFrame()
            continue

        # normalize error filter for parquet loads
        if "error" in df.columns:
            df = df[df["error"].isna()].drop(columns=["error"], errors="ignore")
        drop_cols = [c for c in ["run_id", "ts", "market", "symbol"] if c in df.columns]
        if drop_cols:
            df = df.drop(columns=drop_cols)
        gc.collect()"""

if old_load_alt in c6:
    c6 = c6.replace(old_load_alt, new_load)
elif old_load in c6:
    c6 = c6.replace(old_load, new_load)
elif "pd.read_parquet(local_path)" in c6:
    print("load block already local-first")
else:
    raise SystemExit("load block not found")

set_src(nb["cells"][6], c6)

new_eval = """def evaluate(model, X, y, label):
    preds   = model.predict(X, num_iteration=model.best_iteration)
    mae     = mean_absolute_error(y, preds)
    rmse = np.sqrt(mean_squared_error(y, preds))
    r2      = r2_score(y, preds)
    dir_acc = np.mean(np.sign(preds) == np.sign(y))
    print(f"{label:20s}  MAE={mae:.4f}  RMSE={rmse:.4f}  R²={r2:.4f}  DirAcc={dir_acc:.3%}")
    return {"label": label, "mae": mae, "rmse": rmse, "r2": r2, "dir_acc": dir_acc, "model": "lgbm"}

def evaluate_naive(X, y, label, col="zscore_lag1"):
    \"\"\"Persistence baseline: predict future z ≈ last observed z lag.\"\"\"
    if col not in X.columns:
        print(f"{label:20s}  naive skipped (missing {col})")
        return None
    preds = X[col].astype(float).values
    mask = np.isfinite(preds) & np.isfinite(y)
    preds, yy = preds[mask], y[mask]
    mae     = mean_absolute_error(yy, preds)
    rmse = np.sqrt(mean_squared_error(yy, preds))
    r2      = r2_score(yy, preds)
    dir_acc = np.mean(np.sign(preds) == np.sign(yy))
    print(f"{label:20s}  MAE={mae:.4f}  RMSE={rmse:.4f}  R²={r2:.4f}  DirAcc={dir_acc:.3%}  [naive {col}]")
    return {"label": label, "mae": mae, "rmse": rmse, "r2": r2, "dir_acc": dir_acc, "model": f"naive:{col}"}

results = []
results.append(evaluate(model, X_train, y_train, "train"))
results.append(evaluate(model, X_test,  y_test,  "test"))
results.append(evaluate_naive(X_train, y_train, "train"))
results.append(evaluate_naive(X_test,  y_test,  "test"))

results_df = pd.DataFrame([r for r in results if r is not None])
try:
    lgbm_test = results_df[(results_df.label=="test") & (results_df.model=="lgbm")].iloc[0]
    naive_test = results_df[(results_df.label=="test") & (results_df.model.str.startswith("naive"))].iloc[0]
    print(f"\\nΔR² test (lgbm - naive)     = {lgbm_test.r2 - naive_test.r2:+.4f}")
    print(f"ΔDirAcc test (lgbm - naive) = {lgbm_test.dir_acc - naive_test.dir_acc:+.3%}")
except Exception as e:
    print("delta summary skipped:", e)
results_df
"""

for i, c in enumerate(nb["cells"]):
    src = "".join(c.get("source", []))
    if "def evaluate" in src and "dir_acc" in src:
        set_src(c, new_eval)
        print("patched evaluate cell", i)
        break
else:
    raise SystemExit("evaluate cell not found")

dst_path.write_text(json.dumps(nb, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print("wrote", dst_path)
