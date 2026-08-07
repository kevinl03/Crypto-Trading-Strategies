"""
Realistic Market Friction Analysis for Paper Trading Results

Enhances paper trading results by incorporating actual market conditions:
- Real bid-ask spreads at trade execution time
- Order book depth and slippage for the trade size
- Withdrawal/deposit status (capital mobility friction)
- Exchange operational status
- Network transfer costs

Usage:
    python scripts/paper_trading_3day/analyze_friction_realistic.py data/paper_trading/5day_Aug3_2026
    python scripts/paper_trading_3day/analyze_friction_realistic.py data/paper_trading/5day_Aug3_2026 --trade-size-usd 5000
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent


@dataclass
class FrictionSnapshot:
    """Market friction at a specific point in time."""
    ts: datetime
    snapshot_idx: int
    
    # Execution friction
    spread_bps: float  # bid-ask spread
    slippage_buy_bps: float  # slippage to buy at size
    slippage_sell_bps: float  # slippage to sell at size
    bid_depth_units: float
    ask_depth_units: float
    
    # Capital mobility friction
    withdrawal_blocked: bool
    deposit_blocked: bool
    exchange_down: bool
    
    # Latency friction
    data_latency_ms: float


def load_jsonl_shards(session_dir: Path, stem: str) -> list[dict]:
    """Load all shards of a JSONL file."""
    paths = []
    primary = session_dir / f"{stem}.jsonl"
    if primary.exists():
        paths.append(primary)
    paths.extend(sorted(session_dir.glob(f"{stem}_*.jsonl")))
    
    rows = []
    for path in paths:
        for line in path.read_text(encoding="utf-8-sig").splitlines():
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return rows


def find_collector_run(session_dir: Path) -> Optional[Path]:
    """Find the collector run directory used by this session."""
    config_path = session_dir / "session_config.json"
    if not config_path.exists():
        return None
    
    # Try to find from dashboard
    dashboard_path = session_dir / "dashboard.json"
    if dashboard_path.exists():
        try:
            dashboard = json.loads(dashboard_path.read_text())
            run_dir = dashboard.get("health", {}).get("collector", {}).get("run_dir")
            if run_dir:
                p = Path(run_dir)
                if p.exists():
                    return p
        except Exception:
            pass
    
    # Fall back to finding latest run
    statarb_root = ROOT / "data" / "statarb"
    if not statarb_root.exists():
        return None
    
    dirs = [
        p for p in statarb_root.iterdir()
        if p.is_dir() and (p / "spread_matrix").exists()
    ]
    if not dirs:
        return None
    
    return max(dirs, key=lambda p: p.stat().st_mtime)


def load_friction_data(
    run_dir: Path,
    coin: str,
    exchange_a: str,
    exchange_b: str,
    trade_size_usd: float = 5000,
) -> dict[int, tuple[FrictionSnapshot, FrictionSnapshot]]:
    """
    Load friction snapshots for both exchanges per snapshot_idx.
    
    Returns: {snapshot_idx: (friction_a, friction_b)}
    """
    # Load ticker data (spread, latency)
    ticker_files = sorted((run_dir / "ticker").glob("*.jsonl"))
    tickers = defaultdict(lambda: defaultdict(dict))
    
    for tf in ticker_files:
        for line in tf.read_text(encoding="utf-8-sig").splitlines():
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
                if rec.get("coin") == coin and rec.get("exchange") in (exchange_a, exchange_b):
                    snap = rec.get("snapshot_idx")
                    ex = rec["exchange"]
                    tickers[snap][ex] = rec
            except Exception:
                continue
    
    # Load orderbook data (slippage, depth)
    ob_files = sorted((run_dir / "orderbook").glob("*.jsonl"))
    orderbooks = defaultdict(lambda: defaultdict(dict))
    
    for obf in ob_files:
        for line in obf.read_text(encoding="utf-8-sig").splitlines():
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
                if rec.get("coin") == coin and rec.get("exchange") in (exchange_a, exchange_b):
                    snap = rec.get("snapshot_idx")
                    ex = rec["exchange"]
                    orderbooks[snap][ex] = rec
            except Exception:
                continue
    
    # Load withdrawal status (capital mobility)
    ws_files = sorted((run_dir / "withdrawal_status").glob("*.jsonl"))
    withdrawal_status = defaultdict(lambda: defaultdict(dict))
    
    for wsf in ws_files:
        for line in wsf.read_text(encoding="utf-8-sig").splitlines():
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
                if rec.get("coin") == coin and rec.get("exchange") in (exchange_a, exchange_b):
                    snap = rec.get("snapshot_idx")
                    ex = rec["exchange"]
                    withdrawal_status[snap][ex] = rec
            except Exception:
                continue
    
    # Load exchange status
    es_files = sorted((run_dir / "exchange_status").glob("*.jsonl"))
    exchange_status = defaultdict(dict)
    
    for esf in es_files:
        for line in esf.read_text(encoding="utf-8-sig").splitlines():
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
                ex = rec.get("exchange")
                if ex in (exchange_a, exchange_b):
                    snap = rec.get("snapshot_idx")
                    exchange_status[snap][ex] = rec
            except Exception:
                continue
    
    # Build friction snapshots
    friction = {}
    all_snaps = set(tickers.keys()) | set(orderbooks.keys())
    
    for snap in sorted(all_snaps):
        if exchange_a not in tickers[snap] or exchange_b not in tickers[snap]:
            continue
        
        def make_friction(ex: str) -> FrictionSnapshot:
            tk = tickers[snap][ex]
            ob = orderbooks[snap].get(ex, {})
            ws = withdrawal_status[snap].get(ex, {})
            es = exchange_status[snap].get(ex, {})
            
            # Execution friction
            spread_bps = tk.get("spread_bps", 0) or 0
            
            # Slippage for trade size
            slippage = ob.get("slippage_bps", {})
            slippage_buy = slippage.get(f"buy_{int(trade_size_usd)}", 0) or 0
            slippage_sell = slippage.get(f"sell_{int(trade_size_usd)}", 0) or 0
            
            bid_depth = ob.get("bid_depth_units", 0) or 0
            ask_depth = ob.get("ask_depth_units", 0) or 0
            
            # Capital mobility friction
            withdraw_blocked = not ws.get("withdraw", True)
            deposit_blocked = not ws.get("deposit", True)
            
            # Exchange status
            ex_down = es.get("status") != "ok" if es else False
            
            # Latency
            latency_ms = tk.get("recv_latency_ms", 0) or 0
            
            ts_str = tk.get("ts")
            ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00")) if ts_str else datetime.now(timezone.utc)
            
            return FrictionSnapshot(
                ts=ts,
                snapshot_idx=snap,
                spread_bps=spread_bps,
                slippage_buy_bps=slippage_buy,
                slippage_sell_bps=slippage_sell,
                bid_depth_units=bid_depth,
                ask_depth_units=ask_depth,
                withdrawal_blocked=withdraw_blocked,
                deposit_blocked=deposit_blocked,
                exchange_down=ex_down,
                data_latency_ms=latency_ms,
            )
        
        try:
            friction[snap] = (make_friction(exchange_a), make_friction(exchange_b))
        except Exception:
            continue
    
    return friction


def compute_realistic_costs(
    trade: dict,
    friction_entry: tuple[FrictionSnapshot, FrictionSnapshot],
    friction_exit: Optional[tuple[FrictionSnapshot, FrictionSnapshot]],
) -> dict:
    """
    Compute realistic execution costs for a trade.
    
    For a stat-arb trade:
    - Entry: buy on cheap exchange (A), sell on expensive exchange (B)
    - Exit: reverse (sell A, buy B)
    
    Costs:
    1. Bid-ask spread crossing (both entry and exit)
    2. Slippage (size-dependent)
    3. Taker fees (already in paper_trade_lgbm.py as 4 bps/leg)
    4. Capital mobility penalty (if withdrawal blocked)
    """
    direction = trade.get("direction", 1)
    
    frict_a_entry, frict_b_entry = friction_entry
    
    # Entry costs
    if direction > 0:  # Long spread: buy A (cheap), sell B (expensive)
        # Buy on A: pay the ask, suffer slippage
        entry_spread_a = frict_a_entry.spread_bps / 2  # cross half spread
        entry_slip_a = frict_a_entry.slippage_buy_bps
        
        # Sell on B: hit the bid
        entry_spread_b = frict_b_entry.spread_bps / 2
        entry_slip_b = frict_b_entry.slippage_sell_bps
    else:  # Short spread: sell A, buy B
        entry_spread_a = frict_a_entry.spread_bps / 2
        entry_slip_a = frict_a_entry.slippage_sell_bps
        
        entry_spread_b = frict_b_entry.spread_bps / 2
        entry_slip_b = frict_b_entry.slippage_buy_bps
    
    total_entry_spread = entry_spread_a + entry_spread_b
    total_entry_slip = entry_slip_a + entry_slip_b
    
    # Exit costs (reverse)
    if friction_exit:
        frict_a_exit, frict_b_exit = friction_exit
        
        if direction > 0:  # Long spread: exit = sell A, buy B
            exit_spread_a = frict_a_exit.spread_bps / 2
            exit_slip_a = frict_a_exit.slippage_sell_bps
            
            exit_spread_b = frict_b_exit.spread_bps / 2
            exit_slip_b = frict_b_exit.slippage_buy_bps
        else:  # Short spread: exit = buy A, sell B
            exit_spread_a = frict_a_exit.spread_bps / 2
            exit_slip_a = frict_a_exit.slippage_buy_bps
            
            exit_spread_b = frict_b_exit.spread_bps / 2
            exit_slip_b = frict_b_exit.slippage_sell_bps
        
        total_exit_spread = exit_spread_a + exit_spread_b
        total_exit_slip = exit_slip_a + exit_slip_b
    else:
        # No exit friction data, use entry as proxy
        total_exit_spread = total_entry_spread
        total_exit_slip = total_entry_slip
    
    # Capital mobility penalty
    # If withdrawal blocked on either exchange at entry, add penalty
    mobility_penalty = 0
    if frict_a_entry.withdrawal_blocked or frict_b_entry.withdrawal_blocked:
        # When withdrawals blocked, arbitrageurs can't move capital
        # This is WHY the spread exists - add opportunity cost
        mobility_penalty = 10  # 10 bps penalty for trapped capital
    
    # Exchange downtime penalty
    if frict_a_entry.exchange_down or frict_b_entry.exchange_down:
        mobility_penalty += 50  # Can't trade if exchange down
    
    # Total friction
    total_spread_cost = total_entry_spread + total_exit_spread
    total_slippage_cost = total_entry_slip + total_exit_slip
    
    return {
        "entry_spread_bps": total_entry_spread,
        "entry_slippage_bps": total_entry_slip,
        "exit_spread_bps": total_exit_spread,
        "exit_slippage_bps": total_exit_slip,
        "total_spread_cost_bps": total_spread_cost,
        "total_slippage_cost_bps": total_slippage_cost,
        "mobility_penalty_bps": mobility_penalty,
        "withdrawal_blocked_a_entry": frict_a_entry.withdrawal_blocked,
        "withdrawal_blocked_b_entry": frict_b_entry.withdrawal_blocked,
        "exchange_down_a_entry": frict_a_entry.exchange_down,
        "exchange_down_b_entry": frict_b_entry.exchange_down,
        "bid_depth_a_entry": frict_a_entry.bid_depth_units,
        "ask_depth_a_entry": frict_a_entry.ask_depth_units,
        "bid_depth_b_entry": frict_b_entry.bid_depth_units,
        "ask_depth_b_entry": frict_b_entry.ask_depth_units,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Analyze paper trading with realistic market friction"
    )
    parser.add_argument("session_dir", type=Path, help="Paper trading session directory")
    parser.add_argument(
        "--trade-size-usd",
        type=float,
        default=5000,
        help="Trade size in USD for slippage calculation (default: 5000)"
    )
    parser.add_argument(
        "--taker-fee-bps",
        type=float,
        default=4.0,
        help="Taker fee per leg in bps (default: 4.0)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output path for enhanced results (default: session_dir/friction_analysis.json)"
    )
    
    args = parser.parse_args()
    
    session_dir = args.session_dir.resolve()
    if not session_dir.exists():
        print(f"ERROR: Session directory not found: {session_dir}")
        return
    
    # Find collector run
    run_dir = find_collector_run(session_dir)
    if not run_dir:
        print("ERROR: Could not find collector run directory")
        print("Make sure the data collector has run and produced data")
        return
    
    print(f"Session:   {session_dir.name}")
    print(f"Data run:  {run_dir.name}")
    print(f"Trade size: ${args.trade_size_usd:,.0f}")
    print()
    
    # Load trades
    trades = load_jsonl_shards(session_dir, "trades")
    if not trades:
        print("ERROR: No trades found in session")
        return
    
    print(f"Loaded {len(trades)} trades")
    
    # Extract coin/pair from first trade
    first_trade = trades[0]
    coin = first_trade.get("coin", "UNKNOWN")
    pair = first_trade.get("pair", "unknown__unknown")
    exchange_a, exchange_b = pair.split("__")
    
    print(f"Pair: {coin} on {exchange_a} vs {exchange_b}")
    print()
    
    # Load friction data
    print("Loading market friction data...")
    friction_data = load_friction_data(
        run_dir,
        coin,
        exchange_a,
        exchange_b,
        args.trade_size_usd
    )
    print(f"Loaded friction for {len(friction_data)} snapshots")
    print()
    
    # Enhance trades with realistic friction
    enhanced = []
    matched = 0
    
    for trade in trades:
        entry_snap = trade.get("entry_snap")
        exit_snap = trade.get("exit_snap")
        
        if entry_snap not in friction_data:
            continue
        
        friction_entry = friction_data[entry_snap]
        friction_exit = friction_data.get(exit_snap)
        
        costs = compute_realistic_costs(trade, friction_entry, friction_exit)
        
        # Original PnL proxy (from paper trader)
        original_pnl = trade.get("pnl_proxy", 0)
        
        # Realistic PnL after all friction
        taker_fees_total = args.taker_fee_bps * 4  # 4 legs (entry buy, entry sell, exit sell, exit buy)
        total_friction = (
            costs["total_spread_cost_bps"] +
            costs["total_slippage_cost_bps"] +
            taker_fees_total +
            costs["mobility_penalty_bps"]
        )
        
        realistic_pnl = original_pnl - total_friction
        
        enhanced_trade = {
            **trade,
            **costs,
            "taker_fees_bps": taker_fees_total,
            "total_friction_bps": total_friction,
            "original_pnl_proxy": original_pnl,
            "realistic_pnl_bps": realistic_pnl,
            "friction_adjusted": True,
        }
        
        enhanced.append(enhanced_trade)
        matched += 1
    
    print(f"Matched {matched}/{len(trades)} trades with friction data")
    print()
    
    if not enhanced:
        print("ERROR: No trades could be matched with friction data")
        return
    
    # Summary statistics
    df = pd.DataFrame(enhanced)
    
    print("="*70)
    print("  FRICTION ANALYSIS SUMMARY")
    print("="*70)
    print()
    
    print("Average Costs (bps per trade):")
    print(f"  Spread crossing:    {df['total_spread_cost_bps'].mean():.2f}")
    print(f"  Slippage:           {df['total_slippage_cost_bps'].mean():.2f}")
    print(f"  Taker fees:         {df['taker_fees_bps'].mean():.2f}")
    print(f"  Mobility penalty:   {df['mobility_penalty_bps'].mean():.2f}")
    print(f"  Total friction:     {df['total_friction_bps'].mean():.2f}")
    print()
    
    print("PnL Comparison:")
    print(f"  Original (no friction):     {df['original_pnl_proxy'].mean():+.3f} bps")
    print(f"  Realistic (with friction):  {df['realistic_pnl_bps'].mean():+.3f} bps")
    print(f"  Friction impact:            {(df['original_pnl_proxy'] - df['realistic_pnl_bps']).mean():.3f} bps")
    print()
    
    # Win rates
    original_wins = (df['original_pnl_proxy'] > 0).sum()
    realistic_wins = (df['realistic_pnl_bps'] > 0).sum()
    
    print("Win Rates:")
    print(f"  Original:  {original_wins}/{len(df)} ({original_wins/len(df)*100:.1f}%)")
    print(f"  Realistic: {realistic_wins}/{len(df)} ({realistic_wins/len(df)*100:.1f}%)")
    print()
    
    # Capital mobility issues
    blocked_trades = (
        df['withdrawal_blocked_a_entry'] | df['withdrawal_blocked_b_entry']
    ).sum()
    
    if blocked_trades > 0:
        print("Capital Mobility:")
        print(f"  Trades with withdrawal blocked: {blocked_trades} ({blocked_trades/len(df)*100:.1f}%)")
        print(f"  Mean PnL when blocked:  {df[df['withdrawal_blocked_a_entry'] | df['withdrawal_blocked_b_entry']]['realistic_pnl_bps'].mean():+.3f} bps")
        print(f"  Mean PnL when open:     {df[~(df['withdrawal_blocked_a_entry'] | df['withdrawal_blocked_b_entry'])]['realistic_pnl_bps'].mean():+.3f} bps")
        print()
    
    # Exchange downtime
    downtime_trades = (
        df['exchange_down_a_entry'] | df['exchange_down_b_entry']
    ).sum()
    
    if downtime_trades > 0:
        print(f"Exchange Downtime: {downtime_trades} trades affected")
        print()
    
    # Order book depth analysis
    print("Order Book Depth (mean):")
    print(f"  Exchange A bid depth:  {df['bid_depth_a_entry'].mean():.2f} units")
    print(f"  Exchange A ask depth:  {df['ask_depth_a_entry'].mean():.2f} units")
    print(f"  Exchange B bid depth:  {df['bid_depth_b_entry'].mean():.2f} units")
    print(f"  Exchange B ask depth:  {df['ask_depth_b_entry'].mean():.2f} units")
    print()
    
    print("="*70)
    
    # Save results
    output_path = args.output or (session_dir / "friction_analysis.json")
    
    summary = {
        "session_dir": str(session_dir),
        "collector_run": str(run_dir),
        "coin": coin,
        "pair": pair,
        "trade_size_usd": args.trade_size_usd,
        "n_trades": len(enhanced),
        "n_matched": matched,
        "costs": {
            "mean_spread_bps": float(df['total_spread_cost_bps'].mean()),
            "mean_slippage_bps": float(df['total_slippage_cost_bps'].mean()),
            "mean_fees_bps": float(df['taker_fees_bps'].mean()),
            "mean_mobility_penalty_bps": float(df['mobility_penalty_bps'].mean()),
            "mean_total_friction_bps": float(df['total_friction_bps'].mean()),
        },
        "pnl": {
            "original_mean": float(df['original_pnl_proxy'].mean()),
            "realistic_mean": float(df['realistic_pnl_bps'].mean()),
            "friction_impact": float((df['original_pnl_proxy'] - df['realistic_pnl_bps']).mean()),
        },
        "win_rates": {
            "original": float(original_wins / len(df)),
            "realistic": float(realistic_wins / len(df)),
        },
        "capital_mobility": {
            "trades_with_withdrawal_blocked": int(blocked_trades),
            "pct_blocked": float(blocked_trades / len(df)),
        },
        "trades": enhanced,
    }
    
    output_path.write_text(json.dumps(summary, indent=2))
    print(f"\nResults saved to: {output_path}")
    
    # Also save enhanced trades as CSV for easy analysis
    csv_path = output_path.with_suffix(".csv")
    df.to_csv(csv_path, index=False)
    print(f"Trade details saved to: {csv_path}")


if __name__ == "__main__":
    main()
