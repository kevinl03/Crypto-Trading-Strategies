"""
Monitor a running 3-day paper trading session.

Usage:
    python scripts/paper_trading_3day/monitor_paper_session.py data/paper_trading/3day_20260803_180000
    python scripts/paper_trading_3day/monitor_paper_session.py --latest
    python scripts/paper_trading_3day/monitor_paper_session.py --latest --follow

Also finds legacy session dirs named 5day_* (Campaign C used that prefix).
"""

import argparse
import json
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent


def find_latest_session() -> Path:
    """Find the most recent 3-day (or legacy 5day_*) session directory."""
    paper_root = ROOT / "data" / "paper_trading"
    if not paper_root.exists():
        print("ERROR: No paper trading sessions found")
        sys.exit(1)
    
    dirs = [
        p for p in paper_root.iterdir()
        if p.is_dir() and (p.name.startswith("3day_") or p.name.startswith("5day_"))
    ]
    if not dirs:
        print("ERROR: No 3-day sessions found")
        sys.exit(1)
    
    return max(dirs, key=lambda p: p.stat().st_mtime)


def load_session_data(session_dir: Path) -> dict:
    """Load all monitoring data for a session."""
    data = {}
    
    # Session config
    config_path = session_dir / "session_config.json"
    if config_path.exists():
        data["config"] = json.loads(config_path.read_text())
    
    # Dashboard (health)
    dashboard_path = session_dir / "dashboard.json"
    if dashboard_path.exists():
        data["dashboard"] = json.loads(dashboard_path.read_text())
    
    # Summary (trader stats)
    summary_path = session_dir / "summary.json"
    if summary_path.exists():
        data["summary"] = json.loads(summary_path.read_text())
    
    # Logs size
    collector_log = session_dir / "collector.log"
    trader_log = session_dir / "trader.log"
    data["log_sizes"] = {
        "collector_kb": collector_log.stat().st_size / 1024 if collector_log.exists() else 0,
        "trader_kb": trader_log.stat().st_size / 1024 if trader_log.exists() else 0,
    }
    
    return data


def format_duration(hours: float) -> str:
    """Format hours as 'Xd Yh Zm'."""
    days = int(hours // 24)
    hours_rem = int(hours % 24)
    mins = int((hours % 1) * 60)
    
    parts = []
    if days > 0:
        parts.append(f"{days}d")
    if hours_rem > 0 or (days == 0 and mins == 0):
        parts.append(f"{hours_rem}h")
    if mins > 0:
        parts.append(f"{mins}m")
    
    return " ".join(parts)


def print_status(data: dict):
    """Print formatted session status."""
    config = data.get("config", {})
    dashboard = data.get("dashboard", {})
    summary = data.get("summary", {})
    health = dashboard.get("health", {})
    
    # Header
    print("\n" + "="*80)
    print("  5-DAY PAPER TRADING SESSION - STATUS MONITOR")
    print("="*80)
    
    # Session info
    session_id = config.get("session_id", "Unknown")
    start_time = config.get("start_time")
    if start_time:
        start_dt = datetime.fromisoformat(start_time.replace("Z", "+00:00"))
        print(f"  Session:    {session_id}")
        print(f"  Started:    {start_dt.strftime('%Y-%m-%d %H:%M:%S UTC')}")
        
        # Progress
        now = datetime.now(timezone.utc)
        elapsed = (now - start_dt).total_seconds() / 3600
        progress = min(100, elapsed / 120 * 100)
        end_time = start_dt + timedelta(hours=120)
        remaining = max(0, (end_time - now).total_seconds() / 3600)
        
        # Progress bar
        bar_width = 50
        filled = int(bar_width * progress / 100)
        bar = "█" * filled + "░" * (bar_width - filled)
        
        print(f"  Progress:   [{bar}] {progress:.1f}%")
        print(f"  Uptime:     {format_duration(elapsed)} / 5d 0h")
        print(f"  Remaining:  {format_duration(remaining)}")
        
        if remaining <= 0:
            print(f"  Status:     [COMPLETED]")
        else:
            print(f"  Ends at:    {end_time.strftime('%Y-%m-%d %H:%M:%S UTC')}")
    
    print()
    print("-" * 80)
    
    # Collector status
    coll = health.get("collector", {})
    print("  DATA COLLECTOR")
    status = "[Running]" if coll.get("running") else "[Stopped]"
    print(f"    Status:     {status}")
    if coll.get("running"):
        print(f"    PID:        {coll.get('pid')}")
    if coll.get("restarts", 0) > 0:
        print(f"    Restarts:   {coll.get('restarts')}")
    if coll.get("data_stale"):
        print(f"    WARNING: Data stale (>5min since last update)")
    run_dir = coll.get("run_dir")
    if run_dir:
        print(f"    Run dir:    {Path(run_dir).name}")
    
    print()
    
    # Trader status
    trader = health.get("trader", {})
    print("  LGBM PAPER TRADER")
    status = "[Running]" if trader.get("running") else "[Stopped]"
    print(f"    Status:     {status}")
    if trader.get("running"):
        print(f"    PID:        {trader.get('pid')}")
    if trader.get("restarts", 0) > 0:
        print(f"    Restarts:   {trader.get('restarts')}")
    
    # Trading stats
    n_preds = summary.get("n_preds", 0) or trader.get("n_preds", 0)
    n_closed = summary.get("n_closed", 0) or trader.get("n_closed", 0)
    n_open = summary.get("n_open", 0) or trader.get("n_open", 0)
    
    print()
    print("  TRADING STATISTICS")
    print(f"    Predictions:      {n_preds:,}")
    print(f"    Closed trades:    {n_closed:,}")
    print(f"    Open positions:   {n_open}")
    
    if n_closed > 0:
        dir_acc = summary.get("dir_acc")
        mean_pnl = summary.get("mean_pnl_proxy")
        
        if dir_acc is not None:
            pct = dir_acc * 100
            status_icon = "[+]" if dir_acc >= 0.5 else "[-]"
            print(f"    Direction acc:    {status_icon} {pct:.1f}%")
        
        if mean_pnl is not None:
            sign = "+" if mean_pnl >= 0 else ""
            status_icon = "[UP]" if mean_pnl >= 0 else "[DN]"
            print(f"    Mean PnL proxy:   {status_icon} {sign}{mean_pnl:.3f}")
        
        # Trades per hour
        if start_time and n_closed > 0:
            elapsed = (datetime.now(timezone.utc) - start_dt).total_seconds() / 3600
            if elapsed > 0:
                rate = n_closed / elapsed
                print(f"    Trade rate:       {rate:.1f} trades/hour")
    
    print()
    print("-" * 80)
    
    # System stats
    log_sizes = data.get("log_sizes", {})
    print("  SYSTEM")
    print(f"    Collector log:    {log_sizes.get('collector_kb', 0):.1f} KB")
    print(f"    Trader log:       {log_sizes.get('trader_kb', 0):.1f} KB")
    
    updated = dashboard.get("updated_at")
    if updated:
        updated_dt = datetime.fromisoformat(updated.replace("Z", "+00:00"))
        age = (datetime.now(timezone.utc) - updated_dt).total_seconds()
        print(f"    Last update:      {int(age)}s ago")
    
    print("="*80 + "\n")


def monitor_loop(session_dir: Path, interval: int = 30):
    """Continuously monitor and display session status."""
    print(f"Monitoring session: {session_dir}")
    print(f"Refresh interval: {interval}s")
    print("Press Ctrl+C to exit\n")
    
    try:
        while True:
            try:
                data = load_session_data(session_dir)
                
                # Clear screen (cross-platform)
                print("\033[2J\033[H", end="")
                
                print_status(data)
                
                # Check if session is complete
                config = data.get("config", {})
                if config.get("start_time"):
                    start_dt = datetime.fromisoformat(config["start_time"].replace("Z", "+00:00"))
                    elapsed_h = (datetime.now(timezone.utc) - start_dt).total_seconds() / 3600
                    if elapsed_h >= 120:
                        print("Session complete! Exiting monitor.")
                        break
                
                time.sleep(interval)
                
            except KeyboardInterrupt:
                raise
            except Exception as e:
                print(f"Error reading session data: {e}")
                time.sleep(interval)
    
    except KeyboardInterrupt:
        print("\nMonitoring stopped.")


def main():
    parser = argparse.ArgumentParser(description="Monitor 3-day paper trading session")
    parser.add_argument(
        "session_dir",
        type=Path,
        nargs="?",
        help="Path to session directory"
    )
    parser.add_argument(
        "--latest",
        action="store_true",
        help="Monitor the most recent session"
    )
    parser.add_argument(
        "--follow",
        "-f",
        action="store_true",
        help="Continuously refresh (like tail -f)"
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=30,
        help="Refresh interval in seconds (default: 30)"
    )
    
    args = parser.parse_args()
    
    # Determine session directory
    if args.latest:
        session_dir = find_latest_session()
    elif args.session_dir:
        session_dir = args.session_dir
    else:
        print("ERROR: Must specify session directory or use --latest")
        sys.exit(1)
    
    if not session_dir.exists():
        print(f"ERROR: Session directory not found: {session_dir}")
        sys.exit(1)
    
    # Load and display
    if args.follow:
        monitor_loop(session_dir, args.interval)
    else:
        data = load_session_data(session_dir)
        print_status(data)


if __name__ == "__main__":
    main()
