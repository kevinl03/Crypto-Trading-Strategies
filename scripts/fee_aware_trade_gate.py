"""Fee-aware profit gate: calibrate E[gross_bps | |pred|] on campaign trades.

Rechecks settled LightGBM paper-trading trades and finds whether any |pred|
slice has positive expected edge after round-trip fees. Writes:
  - fee_aware_gate_report.json  (per-session analysis)
  - pred_bps_calib.json         (live-trader calibration table)
  - docs/fee_aware_gate_summary.md

Usage:
  python scripts/fee_aware_trade_gate.py
  python scripts/fee_aware_trade_gate.py --sessions data/paper_trading/July31st_8_hr
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.fees import TRADING_FEES_TAKER  # noqa: E402

COST_GRID_BPS = (0.0, 5.0, 10.0, 15.0, 16.0)
DEFAULT_SESSIONS = [
    ROOT / "data" / "paper_trading" / "July31st_8_hr",
    ROOT / "data" / "paper_trading" / "lgbm_8h_20260730",
]
N_QUANTILE_BINS = 10
TOP_K_FRACS = (1.0, 0.5, 0.25, 0.10, 0.05, 0.02, 0.01)


def load_trades(session: Path) -> pd.DataFrame:
    path = session / "trades.jsonl"
    if not path.exists():
        # Prefer newest shard if primary is empty / missing
        shards = sorted(session.glob("trades*.jsonl"))
        if not shards:
            return pd.DataFrame()
        path = shards[0]
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    need = {"direction", "pred", "spread_delta_bps"}
    if not need.issubset(df.columns):
        return pd.DataFrame()
    df = df.dropna(subset=["direction", "pred", "spread_delta_bps"]).copy()
    df["abs_pred"] = df["pred"].astype(float).abs()
    df["gross_bps"] = df["direction"].astype(float) * df["spread_delta_bps"].astype(float)
    df["pair"] = df.get("pair", pd.Series([""] * len(df)))
    df["round_trip_fee_bps"] = df["pair"].map(pair_round_trip_fee_bps)
    df["net_bps_pair_fee"] = df["gross_bps"] - df["round_trip_fee_bps"]
    return df


def pair_round_trip_fee_bps(pair: str, default_leg_bps: float = 4.0) -> float:
    """Four-leg taker round-trip in bps for exchange_a__exchange_b."""
    if not isinstance(pair, str) or "__" not in pair:
        return 4.0 * default_leg_bps
    a, b = pair.split("__", 1)
    fa = TRADING_FEES_TAKER.get(a)
    fb = TRADING_FEES_TAKER.get(b)
    if fa is None or fb is None:
        return 4.0 * default_leg_bps
    # open both legs + close both legs
    return float(2.0 * (fa + fb) * 10_000)


def summarize_book(gross: np.ndarray, fee: float) -> dict:
    if len(gross) == 0:
        return {
            "n": 0,
            "mean_gross_bps": None,
            "mean_net_bps": None,
            "win_rate_gross": None,
            "win_rate_net": None,
            "median_gross_bps": None,
        }
    net = gross - fee
    return {
        "n": int(len(gross)),
        "mean_gross_bps": float(np.mean(gross)),
        "median_gross_bps": float(np.median(gross)),
        "mean_net_bps": float(np.mean(net)),
        "win_rate_gross": float(np.mean(gross > 0)),
        "win_rate_net": float(np.mean(net > 0)),
    }


def quantile_bins(df: pd.DataFrame, n_bins: int = N_QUANTILE_BINS) -> list[dict]:
    work = df.sort_values("abs_pred").reset_index(drop=True)
    try:
        work["bin"] = pd.qcut(work["abs_pred"], q=n_bins, duplicates="drop")
    except ValueError:
        work["bin"] = pd.cut(work["abs_pred"], bins=min(n_bins, max(len(work), 1)), duplicates="drop")
    out = []
    for i, (label, g) in enumerate(work.groupby("bin", observed=True)):
        gross = g["gross_bps"].to_numpy(float)
        abs_lo = float(g["abs_pred"].min())
        abs_hi = float(g["abs_pred"].max())
        row = {
            "bin_idx": i,
            "abs_pred_lo": abs_lo,
            "abs_pred_hi": abs_hi,
            "abs_pred_mean": float(g["abs_pred"].mean()),
            "n": int(len(g)),
            "mean_gross_bps": float(np.mean(gross)),
            "median_gross_bps": float(np.median(gross)),
            "win_rate_gross": float(np.mean(gross > 0)),
            "mean_net_bps_fee16": float(np.mean(gross - 16.0)),
            "mean_net_bps_pair_fee": float(np.mean(g["net_bps_pair_fee"])),
            "mean_pair_fee_bps": float(g["round_trip_fee_bps"].mean()),
        }
        out.append(row)
    return out


def cumulative_from_top(df: pd.DataFrame, fracs: tuple[float, ...] = TOP_K_FRACS) -> list[dict]:
    """Take the top-f fraction by |pred| and report mean gross/net."""
    work = df.sort_values("abs_pred", ascending=False).reset_index(drop=True)
    n = len(work)
    rows = []
    for f in fracs:
        k = max(1, int(np.ceil(n * f)))
        sub = work.iloc[:k]
        gross = sub["gross_bps"].to_numpy(float)
        abs_floor = float(sub["abs_pred"].min())
        rows.append(
            {
                "top_frac": f,
                "n": int(k),
                "abs_pred_min": abs_floor,
                "mean_gross_bps": float(np.mean(gross)),
                "mean_net_bps_fee16": float(np.mean(gross - 16.0)),
                "mean_net_bps_pair_fee": float(sub["net_bps_pair_fee"].mean()),
                "win_rate_gross": float(np.mean(gross > 0)),
                "clears_fee16": bool(np.mean(gross) > 16.0),
                "clears_pair_fee": bool(np.mean(sub["net_bps_pair_fee"]) > 0),
            }
        )
    return rows


def isotonic_abs_pred_to_gross(df: pd.DataFrame) -> list[dict]:
    """Piecewise-constant isotonic regression: |pred| → E[gross_bps]."""
    try:
        from sklearn.isotonic import IsotonicRegression
    except ImportError:
        # Fallback: use quantile-bin means sorted by abs_pred
        bins = quantile_bins(df)
        return [
            {
                "abs_pred_min": b["abs_pred_lo"],
                "expected_gross_bps": b["mean_gross_bps"],
                "n": b["n"],
            }
            for b in bins
        ]

    x = df["abs_pred"].to_numpy(float)
    y = df["gross_bps"].to_numpy(float)
    # Pool nearby |pred| for stability before isotonic
    order = np.argsort(x)
    x_s, y_s = x[order], y[order]
    # Block-average into ~40 blocks
    n_blocks = min(40, max(5, len(x_s) // 50))
    edges = np.linspace(0, len(x_s), n_blocks + 1).astype(int)
    xb, yb, nb = [], [], []
    for i in range(n_blocks):
        lo, hi = edges[i], edges[i + 1]
        if hi <= lo:
            continue
        xb.append(float(np.mean(x_s[lo:hi])))
        yb.append(float(np.mean(y_s[lo:hi])))
        nb.append(int(hi - lo))
    ir = IsotonicRegression(increasing=True, out_of_bounds="clip")
    y_hat = ir.fit_transform(np.array(xb), np.array(yb))
    return [
        {
            "abs_pred_min": xb[i],
            "expected_gross_bps": float(y_hat[i]),
            "n": nb[i],
        }
        for i in range(len(xb))
    ]


def expected_gross_from_calib(abs_pred: float, calib: list[dict]) -> float:
    """Step / interpolate expected gross from sorted calib knots."""
    if not calib:
        return float("nan")
    knots = sorted(calib, key=lambda r: r["abs_pred_min"])
    if abs_pred <= knots[0]["abs_pred_min"]:
        return float(knots[0]["expected_gross_bps"])
    for i in range(1, len(knots)):
        if abs_pred <= knots[i]["abs_pred_min"]:
            # linear interpolate
            x0, y0 = knots[i - 1]["abs_pred_min"], knots[i - 1]["expected_gross_bps"]
            x1, y1 = knots[i]["abs_pred_min"], knots[i]["expected_gross_bps"]
            if x1 == x0:
                return float(y1)
            t = (abs_pred - x0) / (x1 - x0)
            return float(y0 + t * (y1 - y0))
    return float(knots[-1]["expected_gross_bps"])


def find_fee_clearing_gates(df: pd.DataFrame, fees: tuple[float, ...] = COST_GRID_BPS) -> dict:
    """Smallest abs_pred floor such that mean(gross - fee) > 0 on survivors (in-sample)."""
    work = df.sort_values("abs_pred", ascending=False).reset_index(drop=True)
    abs_vals = work["abs_pred"].to_numpy(float)
    gross = work["gross_bps"].to_numpy(float)
    # Unique candidate floors = each trade's |pred| when included as the marginal row
    out: dict = {}
    for fee in fees:
        best = None
        # Scan from strictest (fewest trades) to loosest
        # Use prefix means of top-k by |pred|
        csum = np.cumsum(gross)
        for k in range(1, len(gross) + 1):
            mean_net = float(csum[k - 1] / k - fee)
            if mean_net > 0:
                floor = float(abs_vals[k - 1])  # min |pred| in top-k
                # Prefer the *largest* k (loosest gate) that still clears — scan continues
                best = {
                    "fee_bps": fee,
                    "abs_pred_min": floor,
                    "n_kept": int(k),
                    "mean_gross_bps": float(csum[k - 1] / k),
                    "mean_net_bps": mean_net,
                    "win_rate_gross": float(np.mean(gross[:k] > 0)),
                    "win_rate_net": float(np.mean(gross[:k] - fee > 0)),
                }
        if best is None:
            out[str(fee)] = {
                "fee_bps": fee,
                "clears": False,
                "note": "no |pred| floor yields positive in-sample mean net",
            }
        else:
            best["clears"] = True
            # Recompute tightest floor that still clears: walk from loosest best.n back
            # The loop kept the loosest (largest k). Also report tightest (smallest k).
            tightest = None
            for k in range(1, len(gross) + 1):
                mean_net = float(csum[k - 1] / k - fee)
                if mean_net > 0:
                    tightest = {
                        "abs_pred_min": float(abs_vals[k - 1]),
                        "n_kept": int(k),
                        "mean_net_bps": mean_net,
                        "mean_gross_bps": float(csum[k - 1] / k),
                    }
                    break
            best["tightest_top_k"] = tightest
            best["loosest_top_k"] = {
                "abs_pred_min": best["abs_pred_min"],
                "n_kept": best["n_kept"],
                "mean_net_bps": best["mean_net_bps"],
                "mean_gross_bps": best["mean_gross_bps"],
            }
            out[str(fee)] = best
    return out


def analyze_session(session: Path) -> dict:
    df = load_trades(session)
    if df.empty:
        return {"session": str(session), "n": 0, "note": "no trades"}

    gross = df["gross_bps"].to_numpy(float)
    baseline = {
        "n": int(len(df)),
        "abs_pred_min": float(df["abs_pred"].min()),
        "abs_pred_median": float(df["abs_pred"].median()),
        "mean_gross_bps": float(np.mean(gross)),
        "gross_win_rate": float(np.mean(gross > 0)),
        "mean_pair_fee_bps": float(df["round_trip_fee_bps"].mean()),
        "mean_net_bps_pair_fee": float(df["net_bps_pair_fee"].mean()),
        "by_fixed_fee": {str(c): summarize_book(gross, c) for c in COST_GRID_BPS},
    }
    bins = quantile_bins(df)
    top = cumulative_from_top(df)
    calib = isotonic_abs_pred_to_gross(df)
    gates = find_fee_clearing_gates(df)

    # Apply calib gate at fee=16: keep rows where E[gross|pred] >= 16
    df = df.copy()
    df["expected_gross_bps"] = df["abs_pred"].map(lambda x: expected_gross_from_calib(float(x), calib))
    kept_calib = df[df["expected_gross_bps"] >= 16.0]
    calib_gate_summary = summarize_book(
        kept_calib["gross_bps"].to_numpy(float) if len(kept_calib) else np.array([]),
        16.0,
    )
    calib_gate_summary["n_eligible_by_expected"] = int(len(kept_calib))

    # abs_pred floor from loosest clearing gate at 16 if any
    g16 = gates.get("16.0", {})
    abs_pred_floor = g16.get("abs_pred_min") if g16.get("clears") else None

    return {
        "session": str(session),
        "n": int(len(df)),
        "baseline_tau0p5_book": baseline,
        "quantile_bins": bins,
        "cumulative_from_top": top,
        "fee_clearing_gates": gates,
        "calib_gate_at_fee16": calib_gate_summary,
        "recommended_abs_pred_floor_fee16": abs_pred_floor,
        "any_slice_clears_fee16": bool(g16.get("clears")),
        "any_top_frac_clears_fee16": any(r["clears_fee16"] for r in top),
        "pred_bps_calib": calib,
    }


def write_calib_json(primary: dict, out_path: Path) -> None:
    """Live-trader calibration artifact."""
    calib = primary.get("pred_bps_calib") or []
    payload = {
        "source_session": primary.get("session"),
        "definition": "expected_gross_bps ≈ E[direction * spread_delta_bps | abs_pred]",
        "round_trip_fee_bps_default": 16.0,
        "recommended_abs_pred_floor_fee16": primary.get("recommended_abs_pred_floor_fee16"),
        "any_slice_clears_fee16": primary.get("any_slice_clears_fee16"),
        "knots": calib,
        "note": (
            "Enter only if expected_gross_bps(abs_pred) >= min_expected_gross_bps. "
            "If any_slice_clears_fee16 is false, no in-sample |pred| floor clears 16 bps."
        ),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_markdown(results: list[dict], out_path: Path) -> None:
    lines = [
        "# Fee-aware profit gate — campaign recheck",
        "",
        "Calibrates `E[gross_bps | |pred|]` on settled LightGBM paper trades and asks",
        "whether any confidence slice clears round-trip fees (default **16 bps** = 4×4).",
        "Pair-specific fees use `scripts/fees.py` taker schedules (4-leg RT).",
        "",
    ]
    for r in results:
        if r.get("n", 0) == 0:
            lines.append(f"## {r.get('session')}\n\nNo trades.\n")
            continue
        base = r["baseline_tau0p5_book"]
        lines += [
            f"## {Path(r['session']).name}",
            "",
            f"- n trades: **{r['n']}**",
            f"- mean gross bps: **{base['mean_gross_bps']:.3f}**",
            f"- mean pair RT fee bps: **{base['mean_pair_fee_bps']:.2f}**",
            f"- mean net (pair fees): **{base['mean_net_bps_pair_fee']:.3f}**",
            f"- any |pred| slice clears 16 bps: **{r['any_slice_clears_fee16']}**",
            "",
            "### Cumulative from top |pred|",
            "",
            "| top frac | n | abs_pred min | mean gross | net@16 | clears@16 |",
            "|---:|---:|---:|---:|---:|:---:|",
        ]
        for row in r["cumulative_from_top"]:
            lines.append(
                f"| {row['top_frac']:.0%} | {row['n']} | {row['abs_pred_min']:.3f} | "
                f"{row['mean_gross_bps']:.3f} | {row['mean_net_bps_fee16']:.3f} | "
                f"{'yes' if row['clears_fee16'] else 'no'} |"
            )
        lines += ["", "### Fee-clearing gates (in-sample mean net > 0)", ""]
        for fee, g in r["fee_clearing_gates"].items():
            if not g.get("clears"):
                lines.append(f"- fee={fee}: **no clearing subset**")
            else:
                lines.append(
                    f"- fee={fee}: clears with loosest `|pred|≥{g['abs_pred_min']:.3f}` "
                    f"(n={g['n_kept']}, mean net={g['mean_net_bps']:.3f})"
                )
        lines.append("")
        lines += [
            "### Quantile bins by |pred|",
            "",
            "| bin | |pred| lo–hi | n | mean gross | net@16 |",
            "|---:|---|---:|---:|---:|",
        ]
        for b in r["quantile_bins"]:
            lines.append(
                f"| {b['bin_idx']} | {b['abs_pred_lo']:.3f}–{b['abs_pred_hi']:.3f} | "
                f"{b['n']} | {b['mean_gross_bps']:.3f} | {b['mean_net_bps_fee16']:.3f} |"
            )
        lines.append("")

    lines += [
        "## Verdict",
        "",
    ]
    primary = next((r for r in results if r.get("n", 0) > 0), None)
    if primary is None:
        lines.append("No trades available.")
    elif not primary.get("any_slice_clears_fee16"):
        lines.append(
            "Under the published LightGBM campaign book, **no `|pred|` confidence slice** "
            "has positive in-sample mean gross above a 16 bps round-trip fee. "
            "A fee-aware gate therefore declines essentially all trades; profitability "
            "requires a different target (bps-net-of-cost) or lower-cost execution, "
            "not a tighter z-filter alone."
        )
    else:
        floor = primary.get("recommended_abs_pred_floor_fee16")
        lines.append(
            f"A fee-clearing subset exists at `|pred| ≥ {floor}` for fee=16 bps "
            "(in-sample). Use `pred_bps_calib.json` with `--enable-fee-gate` in "
            "`paper_trade_lgbm`."
        )
    lines.append("")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Fee-aware |pred|→bps gate calibration")
    ap.add_argument(
        "--sessions",
        nargs="*",
        type=Path,
        default=None,
        help="Paper-trading session dirs (default: Jul31 + Jul30)",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Where to write calib + markdown (default: first session / docs)",
    )
    args = ap.parse_args()
    sessions = args.sessions or [p for p in DEFAULT_SESSIONS if p.exists()]
    if not sessions:
        raise SystemExit("No sessions found")

    results = []
    for s in sessions:
        print(f"Analyzing {s} …", flush=True)
        r = analyze_session(s)
        results.append(r)
        print(
            f"  n={r.get('n', 0)}  clears_fee16={r.get('any_slice_clears_fee16')}  "
            f"floor={r.get('recommended_abs_pred_floor_fee16')}",
            flush=True,
        )
        # Per-session report next to trades
        if s.exists() and r.get("n", 0) > 0:
            (s / "fee_aware_gate_report.json").write_text(
                json.dumps(r, indent=2), encoding="utf-8"
            )

    primary = next((r for r in results if Path(r.get("session", "")).name.startswith("July31")), None)
    if primary is None:
        primary = next((r for r in results if r.get("n", 0) > 0), results[0])

    out_dir = args.out_dir or Path(primary["session"])
    calib_path = out_dir / "pred_bps_calib.json"
    write_calib_json(primary, calib_path)

    md_path = ROOT / "docs" / "fee_aware_gate_summary.md"
    write_markdown(results, md_path)
    # Also mirror under outputs_logo if present
    logo = ROOT / "statarb" / "outputs_logo"
    if logo.exists() or True:
        logo.mkdir(parents=True, exist_ok=True)
        write_markdown(results, logo / "fee_aware_gate_summary.md")

    print(f"\nWrote {calib_path}")
    print(f"Wrote {md_path}")
    print(f"any_slice_clears_fee16={primary.get('any_slice_clears_fee16')}")


if __name__ == "__main__":
    main()
