"""Plot W-sensitivity figures from existing w_sweep_all.csv (no retraining)."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent
CSV_PATH = ROOT / "w_sweep_all.csv"
OUT_MAIN = ROOT / "w_vs_r2_diracc.png"
OUT_N = ROOT / "w_vs_sample_size.png"
BASELINE_W = 300


def load_lgbm(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    out = df.loc[df["model"] == "lgbm_reg"].copy()
    return out.sort_values("zscore_window")


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
    d = load_lgbm(CSV_PATH)
    w = d["zscore_window"]
    print(f"Loaded {len(d)} lgbm_reg rows from {CSV_PATH}")

    fig, axes = plt.subplots(2, 1, figsize=(8.5, 7.0), sharex=False)
    fig.suptitle("Z-score window sensitivity (Jul25 test)", fontsize=13)

    ax = axes[0]
    ax.plot(w, d["r2"], "o-", label=r"$R^2$", markersize=5)
    ax.plot(w, d["r2_tau0.5"], "s-", label=r"$R^2$ ($|\mathrm{pred}| > 0.5$)", markersize=5)
    ax.set_xlabel("W (z-score window)")
    ax.set_ylabel(r"$R^2$")
    ax.set_title(r"Test $R^2$")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)
    add_baseline_vline(ax)

    ax = axes[1]
    ax.plot(w, d["dir_acc"], "o-", label="DirAcc", markersize=5)
    ax.plot(
        w,
        d["dir_acc_tau0.5"],
        "s-",
        label=r"DirAcc ($|\mathrm{pred}| > 0.5$)",
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
