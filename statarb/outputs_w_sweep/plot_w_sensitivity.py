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
    out = out.sort_values("zscore_window")
    return out


def add_baseline_vline(ax) -> None:
    ax.axvline(BASELINE_W, color="0.35", linestyle="--", linewidth=1.2, zorder=0)
    ymin, ymax = ax.get_ylim()
    ax.text(
        BASELINE_W,
        ymax,
        " paper baseline (W=300)",
        va="top",
        ha="left",
        fontsize=8,
        color="0.35",
    )


def plot_sample_size(ax, d: pd.DataFrame, title: str | None = None) -> None:
    w = d["zscore_window"]
    ax.plot(w, d["n_train"], "o-", label="n_train", markersize=4)
    ax.plot(w, d["n_val"], "s-", label="n_val", markersize=4)
    ax.plot(w, d["n"], "^-", label="n (test)", markersize=4)
    ax.set_xlabel("zscore_window W")
    ax.set_ylabel("row count")
    ax.set_title(title or "Row counts vs W (MIN_PERIODS burn-in; no retest)")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)
    add_baseline_vline(ax)
    ax.annotate(
        "larger W => larger MIN_PERIODS~0.3W\\n=> fewer usable rows after burn-in",
        xy=(0.02, 0.02),
        xycoords="axes fraction",
        fontsize=8,
        va="bottom",
        ha="left",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="0.7", alpha=0.9),
    )
    w_hi = d["zscore_window"].iloc[-1]
    n_val_hi = float(d["n_val"].iloc[-1])
    ax.annotate(
        f"n_val collapse at high W\\n(n_val={n_val_hi:.0f} at W={w_hi})",
        xy=(w_hi, n_val_hi),
        xytext=(-80, 40),
        textcoords="offset points",
        fontsize=8,
        arrowprops=dict(arrowstyle="->", color="0.4"),
        color="0.25",
    )


def main() -> None:
    d = load_lgbm(CSV_PATH)
    has_n_train = "n_train" in d.columns
    has_n_val = "n_val" in d.columns
    has_n = "n" in d.columns
    print(f"Loaded: {CSV_PATH}")
    print(f"lgbm_reg rows: {len(d)}; W range: {d['zscore_window'].min()} .. {d['zscore_window'].max()}")
    print(f"columns n_train exist: {has_n_train}")
    print(f"columns n_val exist: {has_n_val}")
    print(f"columns n exist: {has_n}")

    w = d["zscore_window"]

    fig, axes = plt.subplots(3, 1, figsize=(8.5, 10.5), sharex=True)
    fig.suptitle(
        "LightGBM W sensitivity -- TEST-set metrics (jul25_28 holdout)",
        fontsize=12,
        y=0.995,
    )

    ax = axes[0]
    ax.plot(w, d["r2"], "o-", label="r2", markersize=4)
    ax.plot(w, d["r2_tau0.5"], "s-", label="r2 |pred|>0.5", markersize=4)
    ax.set_ylabel("R2")
    ax.set_title(
        "Test R2 vs W (jul25_28 holdout)\\n"
        "Validation (jul19_pre) used only for early stopping -- NOT plotted as skill"
    )
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)
    add_baseline_vline(ax)

    ax = axes[1]
    ax.plot(w, d["dir_acc"], "o-", label="dir_acc", markersize=4)
    ax.plot(w, d["dir_acc_tau0.5"], "s-", label="dir_acc |pred|>0.5", markersize=4)
    ax.set_ylabel("DirAcc")
    ax.set_title("Test DirAcc vs W (jul25_28 holdout)")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)
    add_baseline_vline(ax)

    ax = axes[2]
    if has_n_train and has_n_val and has_n:
        plot_sample_size(ax, d)
    else:
        ax.text(0.5, 0.5, "n_train / n_val / n columns missing", ha="center", va="center")
    axes[2].set_xlabel("zscore_window W")

    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.subplots_adjust(hspace=0.38)
    fig.savefig(OUT_MAIN, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote: {OUT_MAIN}")

    if has_n_train and has_n_val and has_n:
        fig2, ax2 = plt.subplots(figsize=(9, 5.5))
        plot_sample_size(
            ax2,
            d,
            title="Row counts vs W (MIN_PERIODS burn-in; no retest)",
        )
        fig2.tight_layout()
        fig2.savefig(OUT_N, dpi=150, bbox_inches="tight")
        plt.close(fig2)
        print(f"Wrote: {OUT_N}")
    else:
        print(f"Skipped sample-size figure (missing columns): {OUT_N}")


if __name__ == "__main__":
    main()
