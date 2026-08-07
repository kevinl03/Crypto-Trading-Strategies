# 3-Day Paper Trading Session - Execution Guide

This guide explains how to run and manage the 3-day continuous paper trading system.

## Quick Start

### 1. Start the 3-Day Session

**Windows:**
```bash
.venv\Scripts\python.exe scripts\paper_trading_3day\run_3day_paper_session.py --model statarb\outputs\statarb_lgbm.txt
```

**Linux/Mac:**
```bash
.venv/bin/python scripts/paper_trading_3day/run_3day_paper_session.py --model statarb/outputs/statarb_lgbm.txt
```

This single command:
- Starts the data collector (runs continuously until stopped)
- Starts the LGBM paper trader (runs for 72 hours)
- Monitors both processes every 5 minutes
- Auto-restarts crashed processes
- **Prevents Windows sleep** (keeps your PC awake)
- Saves all data to `data/paper_trading/3day_YYYYMMDD_HHMMSS/`

### 2. Monitor the Session (Optional)

Open a **separate terminal** to watch live progress:

```bash
# One-time status check
.venv\Scripts\python.exe scripts\paper_trading_3day\monitor_paper_session.py --latest

# Continuous live monitoring (refreshes every 30s)
.venv\Scripts\python.exe scripts\paper_trading_3day\monitor_paper_session.py --latest --follow
```

The monitor shows:
- Progress bar and time remaining
- Live trade statistics
- Process health (running/stopped)
- Predictions, win rate, PnL

### 3. Stop Early (If Needed)

In the main terminal, press **Ctrl+C** once:
- Trader closes any open positions
- Collector stops data collection
- All state is saved
- System shuts down gracefully

Press **Ctrl+C twice** for immediate force stop.

---

## What Gets Created

```
data/paper_trading/3day_YYYYMMDD_HHMMSS/
├── session_config.json       # Session parameters
├── dashboard.json             # Latest health status (updated every 5 min)
├── summary.json               # Trading statistics (updated every 1 min)
├── signals.jsonl             # All predictions (sharded if >50k lines)
├── trades.jsonl              # Closed trades (sharded if >50k lines)
├── collector.log             # Data collector output
└── trader.log                # Paper trader output

data/statarb/YYYYMMDD_HHMMSS/  (separate data collection run)
├── ticker/20260804.jsonl     # Bid-ask prices
├── orderbook/20260804.jsonl  # L2 order book (slippage, depth)
├── spread_matrix/20260804.jsonl  # Cross-exchange spreads
├── trades/20260804.jsonl     # Recent trade flow
├── ohlcv/20260804.jsonl      # 1-min candles
├── funding_rate/20260804.jsonl   # Perp funding rates
├── open_interest/20260804.jsonl  # Perp OI
├── withdrawal_status/20260804.jsonl  # Withdrawal/deposit status
├── exchange_status/20260804.jsonl    # Exchange health
└── _state.json               # Resumable checkpoint
```

**Note**: Files are partitioned by UTC day, so a 3-day run creates multiple dated files per signal.

---

## Scripts Explained

### 1. `run_3day_paper_session.py` - Main Orchestrator

**What it does:**
- Launches data collector as subprocess
- Waits for initial data (~90 snapshots warmup)
- Launches paper trader as subprocess
- Monitors both every 5 minutes
- Auto-restarts crashed processes (exponential backoff)
- Prevents Windows sleep
- Runs for exactly 72 hours (3 days)

**Usage:**
```bash
# Basic
python scripts/paper_trading_3day/run_3day_paper_session.py --model statarb/outputs/statarb_lgbm.txt

# Custom parameters
python scripts/paper_trading_3day/run_3day_paper_session.py \
    --model statarb/outputs/statarb_lgbm.txt \
    --entry-tau 0.6 \
    --max-open 100 \
    --output-dir data/paper_trading/my_custom_run
```

**Parameters:**
- `--model PATH` - Path to trained LightGBM model (required)
- `--entry-tau FLOAT` - Entry threshold for |prediction| (default: 0.5)
- `--max-open INT` - Maximum open positions (default: 50)
- `--check-interval INT` - Health check interval in seconds (default: 300)
- `--output-dir PATH` - Custom output directory

**How it works:**
1. Creates unique session directory: `data/paper_trading/3day_YYYYMMDD_HHMMSS/`
2. Starts collector: `python -m experiments.collect_statarb_data --forever --interval 60 --assets volatile`
3. Waits for collector to create data directory
4. Starts trader: `python -m experiments.paper_trade_lgbm --model <model> --run-dir <collector_run> --hours 120`
5. Loops every 10 seconds checking if processes are alive
6. Every 5 minutes: runs health check, saves dashboard
7. Auto-restarts if either process crashes (up to 10 times with backoff)
8. After 72 hours OR Ctrl+C: graceful shutdown

### 2. `monitor_paper_session.py` - Status Monitor

**What it does:**
- Reads session files (dashboard.json, summary.json)
- Displays formatted status
- Optional continuous refresh mode

**Usage:**
```bash
# Check latest session
python scripts/paper_trading_3day/monitor_paper_session.py --latest

# Monitor specific session
python scripts/paper_trading_3day/monitor_paper_session.py data/paper_trading/3day_YYYYMMDD_HHMMSS

# Continuous monitoring (like tail -f)
python scripts/paper_trading_3day/monitor_paper_session.py --latest --follow
```

**Parameters:**
- `SESSION_DIR` - Path to session directory (optional if using --latest)
- `--latest` - Monitor the most recent session
- `--follow, -f` - Continuously refresh status
- `--interval INT` - Refresh interval in seconds (default: 30)

### 3. `test_3day_setup.py` - Quick Test

**What it does:**
- Runs a 5-minute test of the full pipeline
- Verifies data collection works
- Verifies paper trader can start
- Checks for common issues

**Usage:**
```bash
python scripts/paper_trading_3day/test_3day_setup.py --model statarb/outputs/statarb_lgbm.txt
```

**When to use:**
- Before starting a full 3-day run
- To verify your environment is set up correctly
- After making changes to the code

### 4. `analyze_friction_realistic.py` - Post-Run Analysis

**What it does:**
- Matches each trade to actual market conditions at execution time
- Computes realistic costs: spread, slippage, fees, capital mobility penalties
- Compares original PnL vs realistic PnL after all friction

**Usage:**
```bash
# Basic analysis (uses $5k trade size for slippage)
python scripts/paper_trading_3day/analyze_friction_realistic.py data/paper_trading/3day_YYYYMMDD_HHMMSS

# Custom trade size
python scripts/paper_trading_3day/analyze_friction_realistic.py data/paper_trading/3day_YYYYMMDD_HHMMSS --trade-size-usd 10000

# Custom fees
python scripts/paper_trading_3day/analyze_friction_realistic.py data/paper_trading/3day_YYYYMMDD_HHMMSS --taker-fee-bps 5.0
```

**Parameters:**
- `SESSION_DIR` - Paper trading session directory (required)
- `--trade-size-usd FLOAT` - Trade size for slippage calculation (default: 5000)
- `--taker-fee-bps FLOAT` - Taker fee per leg in bps (default: 4.0)
- `--output PATH` - Output path (default: session_dir/friction_analysis.json)

**Output:**
- `friction_analysis.json` - Summary statistics
- `friction_analysis.csv` - Per-trade details
- Console report with PnL comparison

---

## Sleep Prevention

The system automatically prevents Windows sleep during the 3-day run:

**How it works:**
- Calls `SetThreadExecutionState` with `ES_SYSTEM_REQUIRED` flag
- Display can turn off, but system stays awake
- Restored to normal behavior on shutdown

**Manual sleep prevention (optional):**

Windows PowerShell (as Administrator):
```powershell
# Disable sleep/hibernate
powercfg /change standby-timeout-ac 0
powercfg /change hibernate-timeout-ac 0

# After 3 days, restore (e.g., 30 min)
powercfg /change standby-timeout-ac 30
powercfg /change hibernate-timeout-ac 180
```

---

## Troubleshooting

### Collector keeps restarting
**Symptoms:** Health check shows `restarts > 5`

**Check:**
```bash
tail -n 50 data/paper_trading/3day_*/collector.log
```

**Common causes:**
- Rate limiting: Increase `--interval` to 90 or 120 seconds
- Network connectivity issues
- Exchange API downtime

**Fix:**
Stop the session and restart with slower interval:
```bash
# Edit run_3day_paper_session.py line 53:
# Change: "--interval", "60",
# To:     "--interval", "90",
```

### Trader not starting
**Symptoms:** Dashboard shows `trader: running=false` after 10+ minutes

**Check:**
```bash
tail -n 50 data/paper_trading/3day_*/trader.log
```

**Common causes:**
- Collector hasn't produced enough data (needs 90+ snapshots)
- Model file missing or invalid
- Data directory permissions

**Fix:**
- Wait longer (90 snapshots at 60s = ~90 minutes)
- Check model path: `ls statarb/outputs/statarb_lgbm.txt`

### System still sleeping
**Symptoms:** Process stops updating during expected sleep time

**Check:**
```bash
# View current power settings
powercfg /query
```

**Fix:**
- Plug in laptop (power settings more aggressive on battery)
- Disable hibernation: `powercfg /hibernate off`
- Run in a cloud VM instead

### High memory usage
**Expected:** ~2-4 GB for the full system (collector + trader)

**If higher:**
- Collector keeps last 340 snapshots in memory (rolling window)
- Trader keeps 300 snapshots + model
- Log files grow ~100-500 KB/hour

**Monitor memory:**
```bash
# PowerShell
Get-Process python | Select-Object Name,Id,WorkingSet
```

---

## Best Practices

### Before Starting

1. **Test first:**
   ```bash
   python scripts/paper_trading_3day/test_3day_setup.py --model statarb/outputs/statarb_lgbm.txt
   ```

2. **Check disk space:**
   - Expect ~3-6 GB for 3 days of data
   - `df -h` (Linux) or `dir` (Windows)

3. **Close resource-heavy apps:**
   - Browser with many tabs
   - IDE/editor with heavy indexing
   - Games, video editing, etc.

4. **Ensure stable network:**
   - Wired connection recommended
   - Disable WiFi power saving

### During Run

1. **Monitor periodically:**
   - Check dashboard 2-3 times per day
   - Look for high restart counts
   - Verify trades are happening

2. **Don't modify files:**
   - Session directory is actively being written
   - Don't open JSONL files in Excel (locks file)
   - Read-only access is fine

3. **Keep laptop plugged in:**
   - Even with sleep prevention
   - Battery drain can cause unexpected shutdown

### After Completion

1. **Analyze friction:**
   ```bash
   python scripts/paper_trading_3day/analyze_friction_realistic.py data/paper_trading/3day_*/
   ```

2. **Calculate Sharpe:**
   ```bash
   python scripts/portfolio_sharpe_paper_session.py data/paper_trading/3day_*/ --max-open 50
   ```

3. **Archive data:**
   - Copy session to external drive/cloud
   - Compress if needed: `tar -czf session.tar.gz data/paper_trading/3day_*/`

---

## Advanced: Running in Background

### Windows (Hidden)

```powershell
# PowerShell (no window)
Start-Process python -ArgumentList "scripts\paper_trading_3day\run_3day_paper_session.py --model statarb\outputs\statarb_lgbm.txt" -WindowStyle Hidden

# Or pythonw.exe (no console)
pythonw scripts\paper_trading_3day\run_3day_paper_session.py --model statarb\outputs\statarb_lgbm.txt
```

### Linux/Mac

```bash
# nohup (survives logout)
nohup python scripts/paper_trading_3day/run_3day_paper_session.py --model statarb/outputs/statarb_lgbm.txt > session.out 2>&1 &

# screen (detachable)
screen -S paper_trading
python scripts/paper_trading_3day/run_3day_paper_session.py --model statarb/outputs/statarb_lgbm.txt
# Ctrl+A then D to detach
# screen -r paper_trading to reattach

# tmux (detachable)
tmux new -s paper_trading
python scripts/paper_trading_3day/run_3day_paper_session.py --model statarb/outputs/statarb_lgbm.txt
# Ctrl+B then D to detach
# tmux attach -t paper_trading to reattach
```

---

## FAQ

**Q: Can I pause and resume?**  
A: No, pausing isn't supported. The trader needs continuous data. If you stop, start fresh.

**Q: Can I run multiple sessions at once?**  
A: No, only one data collector can run at a time (enforced by lock file). You could run multiple traders on the same collector data.

**Q: What happens if my laptop dies?**  
A: Data is saved every 5 minutes. You'll lose the in-progress snapshot and any open positions. Start a fresh session.

**Q: Can I change parameters mid-run?**  
A: No, restart the session. Parameters are baked into session_config.json at startup.

**Q: How do I get email alerts?**  
A: Not built-in. You could write a wrapper script that checks dashboard.json and emails you via SMTP.

**Q: Can I run this on a server?**  
A: Yes! Use screen/tmux and ensure the Python environment is set up. No display server needed.

---

## Support

For issues:
1. Check logs: `collector.log` and `trader.log`
2. Check dashboard: `dashboard.json` for health status
3. Re-read this guide
4. Search existing issues in the repo
5. Create new issue with logs attached
