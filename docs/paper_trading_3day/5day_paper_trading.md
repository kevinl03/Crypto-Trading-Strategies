# 5-Day Continuous Paper Trading Session

This guide shows you how to run a continuous 5-day paper trading session with automated monitoring and health checks.

## Quick Start

### 1. Start the 5-day session

```bash
# Using your existing trained model
python scripts/paper_trading_3day/run_5day_paper_session.py --model statarb/outputs/statarb_lgbm.txt

# With custom parameters
python scripts/paper_trading_3day/run_5day_paper_session.py \
    --model statarb/outputs/statarb_lgbm.txt \
    --entry-tau 0.6 \
    --max-open 100
```

This will:
- Start a data collector that runs continuously (collects every 60 seconds)
- Start the LGBM paper trader configured for 120 hours (5 days)
- Auto-restart failed processes with exponential backoff
- Save all results to `data/paper_trading/5day_YYYYMMDD_HHMMSS/`

### 2. Monitor the session

In a separate terminal:

```bash
# One-time status check
python scripts/paper_trading_3day/monitor_paper_session.py --latest

# Continuous monitoring (refreshes every 30s)
python scripts/paper_trading_3day/monitor_paper_session.py --latest --follow

# Monitor specific session
python scripts/paper_trading_3day/monitor_paper_session.py data/paper_trading/5day_20260803_180000 --follow
```

### 3. Stop early (optional)

Press **Ctrl+C** in the main terminal to gracefully shut down:
- Trader will close any open positions
- Collector will stop data collection
- All state will be saved

## What Gets Created

```
data/paper_trading/5day_YYYYMMDD_HHMMSS/
├── session_config.json       # Session parameters
├── dashboard.json             # Latest health status
├── summary.json               # Trading statistics
├── signals.jsonl             # All predictions
├── trades.jsonl              # Closed trades
├── collector.log             # Data collector output
└── trader.log                # Paper trader output
```

## Features

### Automated Health Monitoring
- Checks every 5 minutes if processes are running
- Auto-restarts crashed processes (up to 10 times)
- Detects stale data feeds (>5 min without updates)
- Saves health snapshots to `dashboard.json`

### Graceful Shutdown
- Closes any open positions at current z-score
- Saves final summary with full statistics
- Flushes all logs and data files

### Progress Tracking
The monitor shows:
- **Progress bar**: Visual progress through the 5-day period
- **Uptime**: How long the session has been running
- **Remaining**: Time until completion
- **Live stats**: Predictions, trades, accuracy, PnL proxy
- **System health**: Process status, restarts, log sizes

## Parameters

### Main Script (`run_5day_paper_session.py`)

```
--model PATH            Path to trained LightGBM model (required)
--entry-tau FLOAT       Entry threshold for |prediction| (default: 0.5)
--max-open INT          Maximum open positions (default: 50)
--check-interval INT    Health check interval in seconds (default: 300)
--output-dir PATH       Custom output directory (optional)
```

### Monitor Script (`monitor_paper_session.py`)

```
SESSION_DIR             Path to session directory (optional if using --latest)
--latest                Monitor the most recent session
--follow, -f            Continuously refresh status
--interval INT          Refresh interval in seconds (default: 30)
```

## Expected Performance

Based on your existing runs:

| Metric | Expected Value |
|--------|---------------|
| Duration | 120 hours (5 days) |
| Snapshots | ~7,200 (60s intervals) |
| Predictions | ~150,000-200,000 |
| Trades | ~5,000-10,000 |
| Direction Accuracy | 55-65% |
| Mean PnL Proxy | +0.05 to +0.15 |

## Troubleshooting

### Collector keeps restarting
Check `collector.log` for errors. Common issues:
- Rate limiting (reduce `--interval` to 90 or 120 seconds)
- Network connectivity
- Exchange API downtime

### Trader not starting
Ensure:
- Collector has created at least 90 snapshots (warmup period)
- Model file exists and is valid
- Data directory is accessible

### High memory usage
For 5-day runs, consider:
- The collector automatically rotates files daily
- Trader keeps only recent data in memory (300 snapshots)
- Log files grow ~100-500 KB/hour

### System sleep/hibernation
The collector **prevents Windows sleep** automatically. If running on battery:
- Plug in the laptop
- Disable hibernation in power settings
- Or use a cloud VM instead

## Post-Session Analysis

After the session completes, analyze results:

```bash
# 1. Realistic friction analysis (NEW - incorporates actual market conditions)
python scripts/paper_trading_3day/analyze_friction_realistic.py data/paper_trading/5day_20260803_180000

# With custom trade size
python scripts/paper_trading_3day/analyze_friction_realistic.py data/paper_trading/5day_20260803_180000 --trade-size-usd 10000

# 2. Calculate portfolio Sharpe ratio
python scripts/portfolio_sharpe_paper_session.py data/paper_trading/5day_20260803_180000 --max-open 50

# 3. Plot trades
python scripts/plot_paper_trades.py data/paper_trading/5day_20260803_180000

# 4. Generate full report
python scripts/report_paper_session.py data/paper_trading/5day_20260803_180000
```

### Friction Analysis Features

The friction analyzer matches each trade to actual market conditions and computes:

**Execution Friction:**
- Bid-ask spread crossing costs (both entry and exit)
- Size-dependent slippage from order book depth
- Taker fees (4 bps per leg by default)

**Capital Mobility Friction:**
- Withdrawal/deposit blocked flags (when arbitrageurs can't move capital)
- Exchange downtime penalties
- Network transfer costs

**Output:**
- `friction_analysis.json` - Summary statistics
- `friction_analysis.csv` - Per-trade details for further analysis
- Comparison of original PnL vs realistic PnL after all friction

## Running in the Background (Advanced)

### Windows
```bash
# PowerShell
Start-Process python -ArgumentList "scripts/paper_trading_3day/run_5day_paper_session.py --model statarb/outputs/statarb_lgbm.txt" -WindowStyle Hidden

# Or use pythonw.exe (no console window)
pythonw scripts/paper_trading_3day/run_5day_paper_session.py --model statarb/outputs/statarb_lgbm.txt
```

### Linux/Mac
```bash
# Run in background with nohup
nohup python scripts/paper_trading_3day/run_5day_paper_session.py --model statarb/outputs/statarb_lgbm.txt > session.out 2>&1 &

# Or use screen/tmux for detachable sessions
screen -S paper_trading
python scripts/paper_trading_3day/run_5day_paper_session.py --model statarb/outputs/statarb_lgbm.txt
# Press Ctrl+A then D to detach
```

## Tips for Long Runs

1. **Use a stable network**: Wired connection recommended
2. **Monitor periodically**: Check status a few times per day
3. **Check logs if restarts occur**: `tail -n 100 data/paper_trading/5day_*/trader.log`
4. **Keep laptop plugged in**: Even though sleep is prevented, battery drain can cause shutdown
5. **Close resource-heavy apps**: Browser with many tabs, IDE, etc.

## Questions?

- Check existing paper trading runs in `data/paper_trading/`
- Review logs for error messages
- Compare with previous successful 8-hour runs
- Adjust `--entry-tau` and `--max-open` based on observed behavior
