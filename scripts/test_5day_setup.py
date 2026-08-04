"""
Quick test of 5-day paper trading setup (runs for 5 minutes).

This runs a shortened version to verify:
- Data collector starts correctly
- Paper trader can read the data
- Monitoring works
- Health checks detect issues
- Shutdown is graceful

Usage:
    python scripts/test_5day_setup.py --model statarb/outputs/statarb_lgbm.txt
"""

import argparse
import json
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def find_latest_run() -> Path:
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


def main():
    parser = argparse.ArgumentParser(description="Test 5-day paper trading setup (5 min test)")
    parser.add_argument("--model", type=Path, required=True, help="Path to LightGBM model")
    args = parser.parse_args()
    
    if not args.model.exists():
        print(f"ERROR: Model not found: {args.model}")
        sys.exit(1)
    
    test_id = datetime.now(timezone.utc).strftime("test_%Y%m%d_%H%M%S")
    test_dir = ROOT / "data" / "paper_trading" / test_id
    test_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print("  5-DAY PAPER TRADING SETUP TEST")
    print("="*70)
    print(f"  Duration:  5 minutes (short test)")
    print(f"  Model:     {args.model}")
    print(f"  Output:    {test_dir}")
    print("="*70 + "\n")
    
    collector_proc = None
    trader_proc = None
    
    try:
        # Step 1: Start collector
        print("[1/4] Starting data collector...")
        collector_cmd = [
            sys.executable,
            "-m", "experiments.collect_statarb_data",
            "--interval", "30",  # Fast for testing
            "--hours", "1",  # 1 hour (we'll stop earlier)
            "--assets", "volatile",
        ]
        
        collector_log = test_dir / "collector.log"
        with open(collector_log, "w") as f:
            collector_proc = subprocess.Popen(
                collector_cmd,
                stdout=f,
                stderr=subprocess.STDOUT,
                cwd=ROOT,
            )
        print(f"  ✓ Collector started (PID {collector_proc.pid})")
        print(f"    Waiting 45s for initial data...")
        time.sleep(45)
        
        # Step 2: Find collector run
        print("\n[2/4] Looking for collector data...")
        run_dir = find_latest_run()
        if not run_dir:
            print("  ✗ ERROR: No collector run directory found")
            print("    Check collector.log for errors")
            return
        
        print(f"  ✓ Found run: {run_dir.name}")
        
        # Check if we have spreads
        spread_files = sorted((run_dir / "spread_matrix").glob("*.jsonl")) if (run_dir / "spread_matrix").exists() else []
        if not spread_files:
            print("  ✗ ERROR: No spread_matrix data")
            return
        
        spread_size = spread_files[-1].stat().st_size
        print(f"    Spread data: {spread_size:,} bytes")
        
        # Step 3: Start trader
        print("\n[3/4] Starting paper trader...")
        trader_cmd = [
            sys.executable,
            "-m", "experiments.paper_trade_lgbm",
            "--model", str(args.model),
            "--run-dir", str(run_dir),
            "--hours", "0.1",  # 6 minutes
            "--entry-tau", "0.5",
            "--poll-sec", "10",
            "--output-dir", str(test_dir),
            "--max-open", "20",
        ]
        
        trader_log = test_dir / "trader.log"
        with open(trader_log, "w") as f:
            trader_proc = subprocess.Popen(
                trader_cmd,
                stdout=f,
                stderr=subprocess.STDOUT,
                cwd=ROOT,
            )
        print(f"  ✓ Trader started (PID {trader_proc.pid})")
        
        # Step 4: Monitor for 3 minutes
        print("\n[4/4] Monitoring (3 minutes)...")
        start = time.time()
        check_count = 0
        
        while time.time() - start < 180:  # 3 minutes
            time.sleep(20)
            check_count += 1
            
            # Check processes
            coll_status = "✓ running" if collector_proc.poll() is None else f"✗ died (exit {collector_proc.poll()})"
            trader_status = "✓ running" if trader_proc.poll() is None else f"✗ died (exit {trader_proc.poll()})"
            
            print(f"\n  Check #{check_count} ({int(time.time() - start)}s elapsed):")
            print(f"    Collector: {coll_status}")
            print(f"    Trader:    {trader_status}")
            
            # Check summary
            summary_path = test_dir / "summary.json"
            if summary_path.exists():
                try:
                    summary = json.loads(summary_path.read_text())
                    print(f"    Snapshots: {summary.get('n_snaps', 0)}")
                    print(f"    Predictions: {summary.get('n_preds', 0)}")
                    print(f"    Closed: {summary.get('n_closed', 0)}")
                    print(f"    Open: {summary.get('n_open', 0)}")
                    if summary.get('dir_acc'):
                        print(f"    Dir Acc: {summary['dir_acc']:.3f}")
                except Exception:
                    pass
            
            # Stop if processes died
            if collector_proc.poll() is not None:
                print("\n  ⚠ Collector died early - check logs")
                break
            if trader_proc.poll() is not None:
                print("\n  ⚠ Trader died early - check logs")
                break
        
        print("\n" + "="*70)
        print("  TEST COMPLETE")
        print("="*70)
        
        # Final check
        if summary_path.exists():
            summary = json.loads(summary_path.read_text())
            n_preds = summary.get('n_preds', 0)
            n_closed = summary.get('n_closed', 0)
            
            print(f"  Predictions: {n_preds}")
            print(f"  Closed trades: {n_closed}")
            
            if n_preds > 0:
                print("\n  ✓ SUCCESS: System is working!")
                print(f"\n  Output directory: {test_dir}")
                print("\n  Ready to run full 5-day session:")
                print(f"    python scripts/run_5day_paper_session.py --model {args.model}")
            else:
                print("\n  ⚠ WARNING: No predictions generated")
                print("    This might be normal if warmup not complete.")
                print("    Check trader.log for details.")
        else:
            print("\n  ⚠ WARNING: No summary.json created")
            print("    Trader may not have started properly.")
            print("    Check trader.log for details.")
        
        print("="*70 + "\n")
    
    except KeyboardInterrupt:
        print("\n\nTest interrupted.")
    
    finally:
        # Cleanup
        print("\nStopping processes...")
        
        if trader_proc and trader_proc.poll() is None:
            trader_proc.terminate()
            try:
                trader_proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                trader_proc.kill()
        
        if collector_proc and collector_proc.poll() is None:
            collector_proc.terminate()
            try:
                collector_proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                collector_proc.kill()
        
        print("Done.\n")
        print(f"Test files saved to: {test_dir}")
        print(f"  Logs: collector.log, trader.log")
        print(f"  Data: signals.jsonl, trades.jsonl")


if __name__ == "__main__":
    main()
