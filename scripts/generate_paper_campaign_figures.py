"""Generate paper campaign figures for Jul30 / Jul31 LGBM paper sessions."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "paper_campaign_figures"
OUT.mkdir(parents=True, exist_ok=True)

SESSIONS = {
    "Jul30": ROOT / "data" / "paper_trading" / "lgbm_8h_20260730",
    "Jul31": ROOT / "data" / "paper_trading" / "July31st_8_hr",
}


def load_metrics(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path / "metrics_report.csv")
    parts = df["set"].str.split(r"\s*\|\s*", n=1, expand=True)
    df["predictor"] = parts[0].str.strip()
    df["slice"] = parts[1].str.strip()
    return df


def style():
    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 160,
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def fig1_ablation():
    """Model vs naive R2 and DirAcc, all vs |pred|>=0.5, both campaigns."""
    rows = []
    for name, path in SESSIONS.items():
        m = load_metrics(path)
        m["campaign"] = name
        rows.append(m)
    df = pd.concat(rows, ignore_index=True)

    slice_order = ["all_predictions", "entries_abs_pred>=0.5"]
    slice_labels = {"all_predictions": "All preds", "entries_abs_pred>=0.5": "|pred|≥0.5"}
    campaigns = list(SESSIONS.keys())
    predictors = ["model", "naive_persistence"]
    pred_labels = {"model": "Model", "naive_persistence": "Naive"}
    colors = {"model": "#1f77b4", "naive_persistence": "#ff7f0e"}

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), constrained_layout=True)
    metrics = [("r2", "R²"), ("dir_acc", "DirAcc")]
    x = np.arange(len(campaigns) * len(slice_order))
    width = 0.36

    for ax, (col, title) in zip(axes, metrics):
        for i, pred in enumerate(predictors):
            vals = []
            for camp in campaigns:
                for sl in slice_order:
                    hit = df[
                        (df["campaign"] == camp)
                        & (df["predictor"] == pred)
                        & (df["slice"] == sl)
                    ]
                    vals.append(float(hit[col].iloc[0]) if len(hit) else np.nan)
            offset = (i - 0.5) * width
            bars = ax.bar(
                x + offset,
                vals,
                width,
                label=pred_labels[pred],
                color=colors[pred],
                edgecolor="white",
                linewidth=0.5,
            )
            for b, v in zip(bars, vals):
                if np.isnan(v):
                    continue
                ax.annotate(
                    f"{v:.2f}" if col == "r2" else f"{v:.1%}",
                    (b.get_x() + b.get_width() / 2, b.get_height()),
                    ha="center",
                    va="bottom" if v >= 0 else "top",
                    fontsize=7,
                    xytext=(0, 2 if v >= 0 else -2),
                    textcoords="offset points",
                )
        ax.axhline(0, color="0.4", lw=0.8)
        ax.set_xticks(x)
        tick_labels = [
            f"{c}\n{slice_labels[s]}" for c in campaigns for s in slice_order
        ]
        ax.set_xticklabels(tick_labels, fontsize=8)
        ax.set_title(title)
        if col == "dir_acc":
            ax.set_ylim(0, 1.05)
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
        else:
            ax.set_ylabel("R²")
        ax.legend(frameon=False, loc="best")

    fig.suptitle("Ablation: model vs naive persistence (matched rows)", fontsize=12)
    out = OUT / "fig1_ablation_r2_diracc.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def load_signals(session_dir: Path) -> pd.DataFrame:
    files = sorted(session_dir.glob("signals*.jsonl"))
    if not files:
        raise FileNotFoundError(f"No signals*.jsonl in {session_dir}")
    rows = []
    for f in files:
        with f.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
    return pd.DataFrame(rows)


def fig2_pred_vs_realized(horizon: int = 1):
    """Scatter pred vs z_fwd H=1 from Jul31 signals."""
    session = SESSIONS["Jul31"]
    sig = load_signals(session)
    # Map (coin, pair, snapshot_idx) -> zscore for forward lookup
    key = sig.set_index(["coin", "pair", "snapshot_idx"])["zscore"]
    fwd_idx = list(
        zip(sig["coin"], sig["pair"], sig["snapshot_idx"].astype(int) + horizon)
    )
    z_fwd = key.reindex(fwd_idx).to_numpy()
    pred = sig["pred"].to_numpy(dtype=float)
    mask = np.isfinite(pred) & np.isfinite(z_fwd)
    pred, z_fwd = pred[mask], z_fwd[mask]

    # Subsample for plotting if huge
    n = len(pred)
    rng = np.random.default_rng(42)
    if n > 40000:
        idx = rng.choice(n, 40000, replace=False)
        pred_s, z_s = pred[idx], z_fwd[idx]
    else:
        pred_s, z_s = pred, z_fwd

    r2 = 1.0 - np.sum((z_fwd - pred) ** 2) / np.sum((z_fwd - z_fwd.mean()) ** 2)
    corr = np.corrcoef(pred, z_fwd)[0, 1]
    dir_acc = float(np.mean(np.sign(pred) == np.sign(z_fwd)))

    fig, ax = plt.subplots(figsize=(6.2, 5.6), constrained_layout=True)
    ax.scatter(pred_s, z_s, s=4, alpha=0.15, c="#1f77b4", linewidths=0, rasterized=True)
    lim = float(np.nanpercentile(np.abs(np.concatenate([pred_s, z_s])), 99.5))
    lim = max(lim, 1.0)
    ax.plot([-lim, lim], [-lim, lim], "k--", lw=1, alpha=0.6, label="y = x")
    ax.axhline(0, color="0.5", lw=0.6)
    ax.axvline(0, color="0.5", lw=0.6)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Predicted z (pred)")
    ax.set_ylabel(f"Realized z_fwd (H={horizon})")
    ax.set_title("Jul31: pred vs realized z_fwd")
    ax.text(
        0.02,
        0.98,
        f"n={n:,}\nR²={r2:.3f}\ncorr={corr:.3f}\nDirAcc={dir_acc:.1%}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85, edgecolor="0.8"),
    )
    ax.legend(frameon=False, loc="lower right")
    out = OUT / "fig2_pred_vs_realized_jul31.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def fig3_filter_r2_lift():
    """Model R2 before/after |pred|>=0.5 filter, both campaigns."""
    camps, before, after = [], [], []
    for name, path in SESSIONS.items():
        m = load_metrics(path)
        b = m[(m["predictor"] == "model") & (m["slice"] == "all_predictions")]["r2"].iloc[0]
        a = m[(m["predictor"] == "model") & (m["slice"] == "entries_abs_pred>=0.5")]["r2"].iloc[0]
        camps.append(name)
        before.append(float(b))
        after.append(float(a))

    x = np.arange(len(camps))
    width = 0.35
    fig, ax = plt.subplots(figsize=(6.5, 4.2), constrained_layout=True)
    b1 = ax.bar(x - width / 2, before, width, label="All predictions", color="#6baed6")
    b2 = ax.bar(x + width / 2, after, width, label="|pred|≥0.5", color="#2171b5")
    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            ax.annotate(
                f"{h:.3f}",
                (bar.get_x() + bar.get_width() / 2, h),
                ha="center",
                va="bottom",
                fontsize=8,
                xytext=(0, 2),
                textcoords="offset points",
            )
    for i, (b, a) in enumerate(zip(before, after)):
        mid_x = x[i] + width / 4
        ax.annotate(
            f"Δ = {a - b:+.3f}",
            (mid_x, a),
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
            color="#333",
            xytext=(0, 18),
            textcoords="offset points",
            arrowprops=dict(arrowstyle="-", color="#999", lw=0.8),
        )
    ax.set_xticks(x)
    ax.set_xticklabels(camps)
    ax.set_ylabel("Model R²")
    ax.set_title("Filter lift: model R² before vs after |pred|≥0.5")
    ax.axhline(0, color="0.4", lw=0.8)
    ax.set_ylim(0, max(after) * 1.35)
    ax.legend(frameon=False)
    out = OUT / "fig3_filter_r2_lift.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def fig4_feature_importance(top_n: int = 10):
    fi_path = ROOT / "statarb" / "outputs" / "feature_importance.csv"
    if not fi_path.exists():
        fi_path = ROOT / "statarb" / "outputs_ob_fix" / "feature_importance.csv"
    if not fi_path.exists():
        print(f"SKIP fig4: missing feature_importance.csv")
        return None
    fi = pd.read_csv(fi_path)
    cols = {c.lower(): c for c in fi.columns}
    feat_col = cols.get("feature", list(fi.columns)[0])
    imp_col = cols.get("importance", list(fi.columns)[1])
    fi = fi.sort_values(imp_col, ascending=False).head(top_n).iloc[::-1]

    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    ax.barh(fi[feat_col], fi[imp_col], color="#3182bd")
    ax.set_xlabel("Importance (gain)")
    ax.set_title(f"Top {top_n} feature importances")
    out = OUT / "fig4_feature_importance_top20.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def fig7_baseline_diracc_pnl():
    """Side-by-side DirAcc and Mean PnL for LightGBM vs mechanical baselines (τ=0.9 ranked capacity)."""
    session = SESSIONS["Jul31"]
    report = json.loads(
        (session / "baseline_strengthening" / "baseline_strengthening_report.json")
        .read_text(encoding="utf-8")
    )
    ranked = report["ranked_capacity"]
    tau = report.get("tau", 0.9)

    strategies = ["LightGBM", "Mech.\npersistence", "Mech.\nmean-reversion"]
    dir_accs = [
        ranked["lgbm_ranked_by_abs_pred"]["dir_acc"],
        ranked["mechanical_persistence_ranked_by_abs_z"]["dir_acc"],
        ranked["mechanical_mean_reversion_ranked_by_abs_z"]["dir_acc"],
    ]
    mean_pnls = [
        ranked["lgbm_ranked_by_abs_pred"]["mean_pnl_proxy"],
        ranked["mechanical_persistence_ranked_by_abs_z"]["mean_pnl_proxy"],
        ranked["mechanical_mean_reversion_ranked_by_abs_z"]["mean_pnl_proxy"],
    ]
    colors = ["#2171b5", "#6baed6", "#bdd7e7"]

    fig, axes = plt.subplots(1, 2, figsize=(8, 4), constrained_layout=True)
    x = np.arange(len(strategies))

    ax = axes[0]
    bars = ax.bar(x, [d * 100 for d in dir_accs], color=colors, edgecolor="white", linewidth=0.5)
    for bar, v in zip(bars, dir_accs):
        ax.annotate(
            f"{v:.1%}",
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            ha="center", va="bottom", fontsize=9, fontweight="bold",
            xytext=(0, 3), textcoords="offset points",
        )
    ax.set_xticks(x)
    ax.set_xticklabels(strategies, fontsize=9)
    ax.set_ylabel("Directional Accuracy (%)")
    ax.set_title("DirAcc (capacity-matched)")
    ax.set_ylim(0, 100)

    ax = axes[1]
    bars = ax.bar(x, mean_pnls, color=colors, edgecolor="white", linewidth=0.5)
    for bar, v in zip(bars, mean_pnls):
        ax.annotate(
            f"{v:+.3f}",
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            ha="center",
            va="bottom" if v >= 0 else "top",
            fontsize=9, fontweight="bold",
            xytext=(0, 3 if v >= 0 else -3),
            textcoords="offset points",
        )
    ax.set_xticks(x)
    ax.set_xticklabels(strategies, fontsize=9)
    ax.set_ylabel("Mean per-trade PnL proxy (z-units)")
    ax.set_title("Mean PnL proxy (capacity-matched)")
    ax.axhline(0, color="0.4", lw=0.8)

    pnl_improve = (mean_pnls[0] - mean_pnls[1]) / mean_pnls[1] * 100
    fig.suptitle(
        f"Jul 31 baseline comparison (|signal|≥{tau:g}, max_open=50, ranked fill): "
        f"+{dir_accs[0] - dir_accs[1]:.1%} DirAcc, +{pnl_improve:.0f}% mean PnL vs persistence",
        fontsize=10,
    )
    out = OUT / "fig7_baseline_diracc_pnl.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def fig5_cum_pnl_jul31():
    """Cumulative PnL for τ=0.9 ranked-capacity LightGBM vs mechanical peers."""
    base = SESSIONS["Jul31"] / "baseline_strengthening"
    series = [
        ("LightGBM", base / "trades_lgbm_ranked_capacity.jsonl", "#1f77b4"),
        ("Mech. persistence", base / "trades_mechanical_persistence_ranked_capacity.jsonl", "#ff7f0e"),
        ("Mech. mean-reversion", base / "trades_mechanical_mean_reversion_ranked_capacity.jsonl", "#2ca02c"),
    ]
    fig, ax = plt.subplots(figsize=(8, 4.2), constrained_layout=True)
    for label, path, color in series:
        rows = []
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        df = pd.DataFrame(rows)
        ts_col = "exit_ts" if "exit_ts" in df.columns else "ts"
        df[ts_col] = pd.to_datetime(df[ts_col], utc=True)
        df = df.sort_values(ts_col)
        df["cum_pnl"] = df["pnl_proxy"].cumsum()
        ax.plot(df[ts_col], df["cum_pnl"], color=color, lw=1.2, label=label)
    ax.axhline(0, color="0.4", lw=0.8)
    ax.set_xlabel("Exit time (UTC)")
    ax.set_ylabel("Cumulative pnl_proxy (z-units)")
    ax.set_title("Jul31 cumulative PnL proxy (|signal|≥0.9, ranked capacity, max_open=50)")
    ax.legend(fontsize=8)
    fig.autofmt_xdate()
    out = OUT / "fig5_cum_pnl_proxy_jul31.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def fig6_model_minus_naive_diracc():
    camps = []
    all_delta = []
    filt_delta = []
    for name, path in SESSIONS.items():
        m = load_metrics(path)
        for sl, bucket in [
            ("all_predictions", all_delta),
            ("entries_abs_pred>=0.5", filt_delta),
        ]:
            model = m[(m["predictor"] == "model") & (m["slice"] == sl)]["dir_acc"].iloc[0]
            naive = m[(m["predictor"] == "naive_persistence") & (m["slice"] == sl)]["dir_acc"].iloc[0]
            bucket.append(float(model) - float(naive))
        camps.append(name)

    x = np.arange(len(camps))
    width = 0.35
    fig, ax = plt.subplots(figsize=(6.5, 4.2), constrained_layout=True)
    b1 = ax.bar(x - width / 2, all_delta, width, label="All predictions", color="#fc8d59")
    b2 = ax.bar(x + width / 2, filt_delta, width, label="|pred|≥0.5", color="#b30000")
    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            ax.annotate(
                f"{h:+.1%}",
                (bar.get_x() + bar.get_width() / 2, h),
                ha="center",
                va="bottom" if h >= 0 else "top",
                fontsize=8,
                xytext=(0, 2 if h >= 0 else -2),
                textcoords="offset points",
            )
    ax.axhline(0, color="0.3", lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels(camps)
    ax.set_ylabel("DirAcc(model) − DirAcc(naive)")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:+.0%}"))
    ax.set_title("Directional accuracy edge vs persistence")
    ax.legend(frameon=False)
    out = OUT / "fig6_model_minus_naive_diracc.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def main():
    style()
    written = []
    for fn in (
        fig1_ablation,
        fig2_pred_vs_realized,
        fig3_filter_r2_lift,
        fig4_feature_importance,
        fig5_cum_pnl_jul31,
        fig6_model_minus_naive_diracc,
        fig7_baseline_diracc_pnl,
    ):
        print(f"Running {fn.__name__}...")
        out = fn()
        if out is not None:
            written.append(out)
            print(f"  -> {out} ({out.stat().st_size} bytes)")
    print("\n=== Written files ===")
    for p in sorted(OUT.glob("*.png")):
        print(f"{p.stat().st_size:10d}  {p}")


if __name__ == "__main__":
    main()
