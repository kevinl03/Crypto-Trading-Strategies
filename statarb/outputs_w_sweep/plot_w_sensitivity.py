"""Plot W-sensitivity figures from w_sweep CSV (no retraining).

Prefers tau=0.9 filtered metrics when present; falls back to tau=0.5.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent
# Prefer a fresh tau-aware CSV if present; else the full archived sweep.
CSV_CANDIDATES = [
    ROOT.parent / "outputs_w_sweep_tau09" / "w_sweep.csv",
    ROOT / "w_sweep_tau09.csv",
    ROOT / "w_sweep.csv",
    ROOT / "w_sweep_all.csv",
]
OUT_MAIN = ROOT / "w_vs_r2_diracc.png"
OUT_N = ROOT / "w_vs_sample_size.png"
BASELINE_W = 300


def resolve_csv() -> Path:
    for p in CSV_CANDIDATES:
        if p.exists():
            return p
    raise FileNotFoundError(f"No sweep CSV among {CSV_CANDIDATES}")


def load_lgbm(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    out = df.loc[df["model"] == "lgbm_reg"].copy()
    return out.sort_values("zscore_window")


def tau_cols(d: pd.DataFrame) -> tuple[str, str, str]:
    if "r2_tau0.9" in d.columns and d["r2_tau0.9"].notna().any():
        return "r2_tau0.9", "dir_acc_tau0.9", "0.9"
    return "r2_tau0.5", "dir_acc_tau0.5", "0.5"


def add_baseline_vline(ax) -> None:
    ax.axvline(BASELINE_W, color="0.45", linestyle="--", linewidth=1.0, zorder=0)
    ax.text(
        BASELINE_W,
        0.98,
        "W=300",
        transform=ax.get_xaxis_transform(),
        va="top",
        ha="left",
        fontsize=8,
        color="0.45",
        clip_on=False,
    )


def plot_sample_size(ax, d: pd.DataFrame) -> None:
    w = d["zscore_window"]
    ax.plot(w, d["n_train"], "o-", label="train", markersize=4)
    ax.plot(w, d["n_val"], "s-", label="val", markersize=4)
    ax.plot(w, d["n"], "^-", label="test", markersize=4)
    ax.set_xlabel("W (z-score window)")
    ax.set_ylabel("Rows")
    ax.set_title("Usable rows vs W")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)
    add_baseline_vline(ax)


def main() -> None:
    csv_path = resolve_csv()
    d = load_lgbm(csv_path)
    w = d["zscore_window"]
    r2_col, dir_col, tau = tau_cols(d)
    print(f"Loaded {len(d)} lgbm_reg rows from {csv_path} (tau={tau})")

    fig, axes = plt.subplots(2, 1, figsize=(8.5, 7.0), sharex=True)
    fig.suptitle(
        rf"Z-score window sensitivity (test set; $|\hat{{z}}|\geq{tau}$)",
        fontsize=13,
    )

    ax = axes[0]
    ax.plot(w, d["r2"], "o-", label=r"$R^2$ (all)", markersize=5, alpha=0.55)
    ax.plot(w, d[r2_col], "s-", label=rf"$R^2$ ($|\mathrm{{pred}}| \geq {tau}$)", markersize=5)
    ax.set_ylabel(r"$R^2$")
    ax.set_title(r"Test $R^2$")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)
    add_baseline_vline(ax)

    ax = axes[1]
    ax.plot(w, d["dir_acc"], "o-", label="DirAcc (all)", markersize=5, alpha=0.55)
    ax.plot(
        w,
        d[dir_col],
        "s-",
        label=rf"DirAcc ($|\mathrm{{pred}}| \geq {tau}$)",
        markersize=5,
    )
    ax.set_xlabel("W (z-score window)")
    ax.set_ylabel("DirAcc")
    ax.set_title("Test directional accuracy")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)
    add_baseline_vline(ax)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(OUT_MAIN, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {OUT_MAIN}")

    if {"n_train", "n_val", "n"}.issubset(d.columns):
        fig2, ax2 = plt.subplots(figsize=(8.5, 4.5))
        plot_sample_size(ax2, d)
        fig2.tight_layout()
        fig2.savefig(OUT_N, dpi=150, bbox_inches="tight")
        plt.close(fig2)
        print(f"Wrote {OUT_N}")


if __name__ == "__main__":
    main()
