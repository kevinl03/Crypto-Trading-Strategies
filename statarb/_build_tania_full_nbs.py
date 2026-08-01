"""Rebuild Tania's two experiments on the full dataset.

Both variants start from cex_gbm_new_live_8h_july30.ipynb (the notebook that
trained the 8h live paper-trading model), so they inherit the Jul 22-28 train
window, local-first parquet loading, the orderbook slippage_bps fix, and the
naive persistence baseline. That keeps the only differences the ones she
actually made:

  cex_gbm_new_TEST_full.ipynb    fe5ca43 "get rid of data cleaning step"
                                 -> cleaning / log / winsor cell disabled
                                 -> her original hyperparameters kept

  cex_gbm_new_tinker_full.ipynb  3f28628 "tinker"
                                 -> learning_rate 0.075, lambda_l1 0.1
                                 -> funding_rate / open_interest off

Both train longer than the 2000/50 baseline via LONGER_ROUNDS/LONGER_PATIENCE.
"""
from __future__ import annotations

import json
from pathlib import Path

import nbformat

SRC = Path("cex_gbm_new_live_8h_july30.ipynb")
CONFIG_CELL = 3
LOAD_TABLES_CELL = 4
CLEANING_CELL = 25

LONGER_ROUNDS = 4000
LONGER_PATIENCE = 300

CLEANING_DISABLED = '''# ── 8.5.2 Chop chop — DISABLED for this variant ──────────────────────────────
# Tania's fe5ca43 removed the cleaning step entirely: no dropping of high-null /
# near-zero / weak-correlation columns, no signed-log transform, no winsorizing.
# Kept as a no-op cell so the downstream cell layout is unchanged.
print("cleaning step skipped (no drops, no log transform, no winsorize)")
print(f"train cols kept: {df_train.shape[1]}  test cols kept: {df_test.shape[1]}")
'''


def replace(text: str, old: str, new: str, what: str) -> str:
    if old not in text:
        raise SystemExit(f"could not find {what!r}: {old!r}")
    return text.replace(old, new, 1)


def build(variant: str) -> Path:
    nb = nbformat.read(SRC, as_version=4)
    cfg = nb.cells[CONFIG_CELL].source
    tables = nb.cells[LOAD_TABLES_CELL].source

    cfg = replace(
        cfg,
        'OUTPUT_DIR = Path("./outputs_ob_fix")',
        f'OUTPUT_DIR = Path("./outputs_{variant}")',
        "OUTPUT_DIR",
    )
    cfg = replace(cfg, "NUM_BOOST_ROUND = 2000 #1000", f"NUM_BOOST_ROUND = {LONGER_ROUNDS}", "rounds")
    cfg = replace(cfg, "EARLY_STOPPING  = 50 #50", f"EARLY_STOPPING  = {LONGER_PATIENCE}", "patience")

    if variant == "TEST_full":
        nb.cells[CLEANING_CELL].source = CLEANING_DISABLED
    elif variant == "tinker_full":
        cfg = replace(cfg, '"learning_rate":     0.05, #0.05', '"learning_rate":     0.075,', "lr")
        cfg = replace(cfg, '"lambda_l1":         0.01, #0.1', '"lambda_l1":         0.1,', "l1")
        tables = replace(
            tables,
            '"funding_rate":  True,  # toggle off to save RAM if still crashing',
            '"funding_rate":  False,  # tinker: disabled',
            "funding_rate",
        )
        tables = replace(
            tables,
            '"open_interest": True,  # toggle off to save RAM if still crashing',
            '"open_interest": False,  # tinker: disabled',
            "open_interest",
        )
    else:
        raise SystemExit(f"unknown variant {variant}")

    nb.cells[CONFIG_CELL].source = cfg
    nb.cells[LOAD_TABLES_CELL].source = tables

    for cell in nb.cells:
        if cell.cell_type == "code":
            cell.outputs = []
            cell.execution_count = None
        else:
            # Colab leaves outputs/execution_count on markdown cells, which fails
            # nbformat validation and makes nbconvert refuse the notebook.
            cell.pop("outputs", None)
            cell.pop("execution_count", None)

    nbformat.validate(nb)
    out = Path(f"cex_gbm_new_{variant}.ipynb")
    nbformat.write(nb, out)
    return out


def main() -> None:
    for variant in ("TEST_full", "tinker_full"):
        path = build(variant)
        nb = json.loads(path.read_text(encoding="utf-8"))
        cfg = nb["cells"][CONFIG_CELL]["source"]
        tables = nb["cells"][LOAD_TABLES_CELL]["source"]
        cfg = "".join(cfg) if isinstance(cfg, list) else cfg
        tables = "".join(tables) if isinstance(tables, list) else tables
        clean = nb["cells"][CLEANING_CELL]["source"]
        clean = "".join(clean) if isinstance(clean, list) else clean
        print(f"\n=== {path} ===")
        for line in cfg.splitlines():
            if any(k in line for k in ("OUTPUT_DIR =", "learning_rate", "lambda_l1", "NUM_BOOST_ROUND", "EARLY_STOPPING")):
                print("  ", line.strip())
        for line in tables.splitlines():
            if "funding_rate" in line or "open_interest" in line:
                print("  ", line.strip())
        print("   cleaning:", "DISABLED" if "skipped" in clean else "active")
        print("   jul22_28 in windows:", "jul22_28" in "".join(nb["cells"][6]["source"]))


if __name__ == "__main__":
    main()
