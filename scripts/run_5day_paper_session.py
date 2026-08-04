"""
5-Day Continuous Paper Trading Session Manager

Orchestrates a multi-day paper trading run with:
- Continuous data collection (until stopped)
- LGBM paper trader (120 hours = 5 days)
- Automated health monitoring every 5 minutes
- Graceful shutdown on completion or Ctrl+C
- Auto-restart on failures (with exponential backoff)

Usage:
    python scripts/run_5day_paper_session.py --model statarb/outputs/statarb_lgbm.txt
    python scripts/run_5day_paper_session.py --model path/to/model.txt --entry-tau 0.6 --max-open 50

Monitoring:
    The script creates a monitoring dashboard at data/paper_trading/<session_id>/dashboard.json
    Health checks run every 5 minutes and auto-restart stalled processes

To stop early:
    Press Ctrl+C once for graceful shutdown (closes positions, saves state)
    Press Ctrl+C twice for immediate exit
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

# Windows sleep prevention
if sys.platform == "win32":
    import ctypes
    ES_CONTINUOUS = 0x80000000
    ES_SYSTEM_REQUIRED = 0x00000001

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# ---------------------------------------------------------------------------
# Process management
# ---------------------------------------------------------------------------

class ProcessManager:
    """Manages collector and paper trader subprocesses with auto-restart."""
    
    def __init__(self, session_dir: Path, model_path: Path, entry_tau: float, max_open: int):
        self.session_dir = session_dir
        self.model_path = model_path
        self.entry_tau = entry_tau
        self.max_open = max_open
        
        self.collector_proc: Optional[subprocess.Popen] = None
        self.trader_proc: Optional[subprocess.Popen] = None
        self.collector_run_dir: Optional[Path] = None
        
        self.collector_restarts = 0
        self.trader_restarts = 0
        self.max_restarts = 10
        
        self.start_time = time.time()
        self.shutdown_requested = False
        
    def start_collector(self) -> bool:
        """Start the data collector with --forever flag."""
        if self.collector_proc and self.collector_proc.poll() is None:
            return True  # Already running
            
        print(f"\n[{self._timestamp()}] Starting data collector...")
        
        cmd = [
            sys.executable,
            "-m", "experiments.collect_statarb_data",
            "--forever",  # Run until stopped
            "--interval", "60",  # 60-second snapshots for live trading
            "--assets", "volatile",  # Volatile coins (WIF, PEPE, CRV, etc.)
            "--slow-every", "10",  # Slow signals every 10 snapshots
        ]
        
        log_file = self.session_dir / "collector.log"
        try:
            with open(log_file, "a") as f:
                self.collector_proc = subprocess.Popen(
                    cmd,
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    cwd=ROOT,
                )
            self.collector_restarts += 1
            print(f"  [OK] Collector started (PID {self.collector_proc.pid})")
            print(f"    Log: {log_file}")
            
            # Wait for collector to create run directory
            time.sleep(5)
            self.collector_run_dir = self._find_latest_run()
            if self.collector_run_dir:
                print(f"    Run dir: {self.collector_run_dir}")
            return True
            
        except Exception as e:
            print(f"  [ERROR] Failed to start collector: {e}")
            return False
    
    def start_trader(self) -> bool:
        """Start the LGBM paper trader for 120 hours (5 days)."""
        if self.trader_proc and self.trader_proc.poll() is None:
            return True  # Already running
            
        if not self.collector_run_dir:
            self.collector_run_dir = self._find_latest_run()
            if not self.collector_run_dir:
                print(f"[{self._timestamp()}] Waiting for collector run directory...")
                return False
        
        print(f"\n[{self._timestamp()}] Starting LGBM paper trader...")
        
        cmd = [
            sys.executable,
            "-m", "experiments.paper_trade_lgbm",
            "--model", str(self.model_path),
            "--run-dir", str(self.collector_run_dir),
            "--hours", "120",  # 5 days
            "--entry-tau", str(self.entry_tau),
            "--poll-sec", "15",  # Check for new data every 15s
            "--output-dir", str(self.session_dir),
            "--max-open", str(self.max_open),
        ]
        
        log_file = self.session_dir / "trader.log"
        try:
            with open(log_file, "a") as f:
                self.trader_proc = subprocess.Popen(
                    cmd,
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    cwd=ROOT,
                )
            self.trader_restarts += 1
            print(f"  [OK] Trader started (PID {self.trader_proc.pid})")
            print(f"    Log: {log_file}")
            print(f"    Duration: 120 hours (5 days)")
            print(f"    Entry threshold: tau={self.entry_tau}")
            print(f"    Max open positions: {self.max_open}")
            return True
            
        except Exception as e:
            print(f"  [ERROR] Failed to start trader: {e}")
            return False
    
    def check_health(self) -> dict:
        """Check health of both processes and restart if needed."""
        health = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "uptime_hours": (time.time() - self.start_time) / 3600,
            "collector": self._check_collector(),
            "trader": self._check_trader(),
        }
        
        # Auto-restart failed processes
        if not health["collector"]["running"] and not self.shutdown_requested:
            if self.collector_restarts < self.max_restarts:
                print(f"\n[{self._timestamp()}] Collector died, restarting...")
                backoff = min(2 ** (self.collector_restarts - 1), 60)
                print(f"  Backoff: {backoff}s")
                time.sleep(backoff)
                self.start_collector()
            else:
                print(f"\n[{self._timestamp()}] Collector failed {self.max_restarts} times, giving up")
                
        if not health["trader"]["running"] and not self.shutdown_requested:
            # Check if it finished naturally (120 hours elapsed)
            elapsed_h = (time.time() - self.start_time) / 3600
            if elapsed_h < 119.5:  # Allow some margin
                if self.trader_restarts < self.max_restarts:
                    print(f"\n[{self._timestamp()}] Trader died early, restarting...")
                    backoff = min(2 ** (self.trader_restarts - 1), 60)
                    print(f"  Backoff: {backoff}s")
                    time.sleep(backoff)
                    self.start_trader()
                else:
                    print(f"\n[{self._timestamp()}] Trader failed {self.max_restarts} times, giving up")
            else:
                print(f"\n[{self._timestamp()}] Trader completed 5-day run!")
                self.shutdown_requested = True
        
        return health
    
    def _check_collector(self) -> dict:
        """Check collector health."""
        if not self.collector_proc:
            return {"running": False, "pid": None, "restarts": self.collector_restarts}
        
        poll = self.collector_proc.poll()
        running = poll is None
        
        # Check data freshness
        stale = False
        if self.collector_run_dir and (self.collector_run_dir / "spread_matrix").exists():
            files = sorted((self.collector_run_dir / "spread_matrix").glob("*.jsonl"))
            if files:
                age = time.time() - files[-1].stat().st_mtime
                stale = age > 300  # No data in 5 minutes
        
        return {
            "running": running,
            "pid": self.collector_proc.pid if running else None,
            "exit_code": poll,
            "restarts": self.collector_restarts,
            "data_stale": stale,
            "run_dir": str(self.collector_run_dir) if self.collector_run_dir else None,
        }
    
    def _check_trader(self) -> dict:
        """Check trader health."""
        if not self.trader_proc:
            return {"running": False, "pid": None, "restarts": self.trader_restarts}
        
        poll = self.trader_proc.poll()
        running = poll is None
        
        # Load summary for stats
        summary = {}
        summary_path = self.session_dir / "summary.json"
        if summary_path.exists():
            try:
                summary = json.loads(summary_path.read_text())
            except Exception:
                pass
        
        return {
            "running": running,
            "pid": self.trader_proc.pid if running else None,
            "exit_code": poll,
            "restarts": self.trader_restarts,
            "n_closed": summary.get("n_closed", 0),
            "n_open": summary.get("n_open", 0),
            "n_preds": summary.get("n_preds", 0),
            "dir_acc": summary.get("dir_acc"),
            "mean_pnl_proxy": summary.get("mean_pnl_proxy"),
        }
    
    def shutdown(self, timeout: int = 30):
        """Gracefully shutdown both processes."""
        self.shutdown_requested = True
        print(f"\n[{self._timestamp()}] Shutting down gracefully...")
        
        # Stop trader first (closes positions)
        if self.trader_proc and self.trader_proc.poll() is None:
            print("  Stopping trader...")
            try:
                self.trader_proc.terminate()
                self.trader_proc.wait(timeout=timeout)
                print(f"    [OK] Trader stopped (exit code {self.trader_proc.returncode})")
            except subprocess.TimeoutExpired:
                print("    [WARN] Trader didn't stop gracefully, forcing...")
                self.trader_proc.kill()
                self.trader_proc.wait()
        
        # Stop collector
        if self.collector_proc and self.collector_proc.poll() is None:
            print("  Stopping collector...")
            try:
                self.collector_proc.terminate()
                self.collector_proc.wait(timeout=timeout)
                print(f"    [OK] Collector stopped (exit code {self.collector_proc.returncode})")
            except subprocess.TimeoutExpired:
                print("    [WARN] Collector didn't stop gracefully, forcing...")
                self.collector_proc.kill()
                self.collector_proc.wait()
    
    def _find_latest_run(self) -> Optional[Path]:
        """Find the most recent collector run directory."""
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
    
    def _timestamp(self) -> str:
        return datetime.now(timezone.utc).strftime("%H:%M:%S")


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------

def save_dashboard(session_dir: Path, health: dict, session_info: dict):
    """Save monitoring dashboard."""
    dashboard = {
        **session_info,
        "health": health,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    
    path = session_dir / "dashboard.json"
    path.write_text(json.dumps(dashboard, indent=2))


def print_dashboard(health: dict, session_info: dict):
    """Print dashboard to console."""
    uptime_h = health.get("uptime_hours", 0)
    end_time = session_info["start_time"] + timedelta(hours=120)
    remaining_h = max(0, (end_time - datetime.now(timezone.utc)).total_seconds() / 3600)
    
    print("\n" + "="*70)
    print(f"  5-DAY PAPER TRADING SESSION")
    print("="*70)
    print(f"  Started:   {session_info['start_time'].strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(f"  Uptime:    {uptime_h:.1f}h / 120h ({uptime_h/120*100:.1f}%)")
    print(f"  Remaining: {remaining_h:.1f}h")
    print(f"  Session:   {session_info['session_dir']}")
    print()
    
    # Collector
    coll = health.get("collector", {})
    status = "[RUNNING]" if coll.get("running") else "[STOPPED]"
    print(f"  Data Collector: {status}")
    if coll.get("running"):
        print(f"    PID: {coll.get('pid')}")
        print(f"    Restarts: {coll.get('restarts', 0)}")
        if coll.get("data_stale"):
            print(f"    WARNING: Data is stale (>5min since last update)")
    print()
    
    # Trader
    trader = health.get("trader", {})
    status = "[RUNNING]" if trader.get("running") else "[STOPPED]"
    print(f"  LGBM Paper Trader: {status}")
    if trader.get("running") or trader.get("n_closed", 0) > 0:
        print(f"    PID: {trader.get('pid', 'N/A')}")
        print(f"    Predictions: {trader.get('n_preds', 0):,}")
        print(f"    Closed trades: {trader.get('n_closed', 0)}")
        print(f"    Open positions: {trader.get('n_open', 0)}")
        acc = trader.get("dir_acc")
        pnl = trader.get("mean_pnl_proxy")
        if acc is not None:
            print(f"    Direction accuracy: {acc:.3f}")
        if pnl is not None:
            print(f"    Mean PnL proxy: {pnl:+.3f}")
        print(f"    Restarts: {trader.get('restarts', 0)}")
    print("="*70 + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _prevent_sleep():
    """Prevent Windows from sleeping during the 5-day run."""
    if sys.platform == "win32":
        try:
            ctypes.windll.kernel32.SetThreadExecutionState(
                ES_CONTINUOUS | ES_SYSTEM_REQUIRED
            )
            print("  [POWER] Windows sleep prevented for this session")
        except Exception:
            pass


def _allow_sleep():
    """Restore normal Windows sleep behavior."""
    if sys.platform == "win32":
        try:
            ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)
        except Exception:
            pass


def main():
    parser = argparse.ArgumentParser(
        description="5-day continuous paper trading session manager"
    )
    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="Path to trained LightGBM model (.txt file)"
    )
    parser.add_argument(
        "--entry-tau",
        type=float,
        default=0.5,
        help="Entry threshold for |prediction| (default: 0.5)"
    )
    parser.add_argument(
        "--max-open",
        type=int,
        default=50,
        help="Maximum open positions (default: 50)"
    )
    parser.add_argument(
        "--check-interval",
        type=int,
        default=300,
        help="Health check interval in seconds (default: 300 = 5min)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Custom output directory (default: auto-generated)"
    )
    
    args = parser.parse_args()
    
    # Validate model
    if not args.model.exists():
        print(f"ERROR: Model file not found: {args.model}")
        sys.exit(1)
    
    # Create session directory
    session_id = datetime.now(timezone.utc).strftime("5day_%Y%m%d_%H%M%S")
    if args.output_dir:
        session_dir = args.output_dir
    else:
        session_dir = ROOT / "data" / "paper_trading" / session_id
    session_dir.mkdir(parents=True, exist_ok=True)
    
    # Session info
    session_info = {
        "session_id": session_id,
        "session_dir": str(session_dir),
        "start_time": datetime.now(timezone.utc),
        "model_path": str(args.model.resolve()),
        "entry_tau": args.entry_tau,
        "max_open": args.max_open,
        "duration_hours": 120,
    }
    
    # Save session config
    config_path = session_dir / "session_config.json"
    config_path.write_text(
        json.dumps({
            **session_info,
            "start_time": session_info["start_time"].isoformat(),
        }, indent=2)
    )
    
    print("="*70)
    print("  5-DAY CONTINUOUS PAPER TRADING SESSION")
    print("="*70)
    print(f"  Model:        {args.model}")
    print(f"  Entry tau:    {args.entry_tau}")
    print(f"  Max open:     {args.max_open}")
    print(f"  Duration:     120 hours (5 days)")
    print(f"  Output:       {session_dir}")
    print(f"  Health check: every {args.check_interval}s")
    print()
    print("  Starting processes...")
    print("="*70 + "\n")
    
    # Prevent Windows sleep
    _prevent_sleep()
    
    # Initialize process manager
    pm = ProcessManager(
        session_dir=session_dir,
        model_path=args.model.resolve(),
        entry_tau=args.entry_tau,
        max_open=args.max_open,
    )
    
    # Signal handlers for graceful shutdown
    def signal_handler(signum, frame):
        pm.shutdown()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Start processes
    pm.start_collector()
    time.sleep(10)  # Wait for collector to initialize
    pm.start_trader()
    
    # Monitoring loop
    last_check = 0
    last_dashboard_print = 0
    
    try:
        while not pm.shutdown_requested:
            now = time.time()
            
            # Health check
            if now - last_check >= args.check_interval:
                health = pm.check_health()
                save_dashboard(session_dir, health, {
                    **session_info,
                    "start_time": session_info["start_time"].isoformat(),
                })
                last_check = now
            
            # Print dashboard every 30 minutes
            if now - last_dashboard_print >= 1800:
                health = pm.check_health()
                print_dashboard(health, session_info)
                last_dashboard_print = now
            
            # Check if 5 days elapsed
            elapsed_h = (time.time() - pm.start_time) / 3600
            if elapsed_h >= 120:
                print(f"\n[{pm._timestamp()}] 5-day session complete!")
                pm.shutdown_requested = True
                break
            
            time.sleep(10)  # Check every 10s
    
    except KeyboardInterrupt:
        print("\n\n[INTERRUPT] Received Ctrl+C")
    
    finally:
        _allow_sleep()
        pm.shutdown()
        
        # Final summary
        print("\n" + "="*70)
        print("  SESSION COMPLETE")
        print("="*70)
        health = pm.check_health()
        trader = health.get("trader", {})
        print(f"  Total runtime: {health.get('uptime_hours', 0):.1f} hours")
        print(f"  Predictions: {trader.get('n_preds', 0):,}")
        print(f"  Closed trades: {trader.get('n_closed', 0)}")
        print(f"  Direction accuracy: {trader.get('dir_acc', 0):.3f}")
        print(f"  Mean PnL proxy: {trader.get('mean_pnl_proxy', 0):+.3f}")
        print()
        print(f"  Output directory: {session_dir}")
        print("="*70 + "\n")


if __name__ == "__main__":
    main()
