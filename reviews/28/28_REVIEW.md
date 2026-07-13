# PR #28 — C++ Signal Engine for Hot-Path Acceleration

**Branch:** `feat/cpp-signal-engine` → `main`
**Author:** Kevin Litvin
**Reviewed:** 2026-06-10

**PR summary:** Adds a C++17/pybind11 extension module that replaces the
hot-path signal functions (OU estimation, z-score computation, batch backtest
loop) with native implementations. Achieves 160–209× speedup on batch
operations. Integrates into `paper_trader.py` and `backtest_historical.py`
with transparent Python fallback.

---

## Critical

### 1. OU backtest strategy: C++ re-estimates params per bar, Python estimates once

**File:** `experiments/backtest_historical.py` — `run_ou_strategy`
**File:** `cpp/signal_engine.cpp` — `backtest_engine`

The Python path calls `estimate_ou_params()` **once** on the first `warmup*2`
bars and uses those static `(mu, sigma)` for all subsequent z-scores:

```python
ou = estimate_ou_params(spread_bps[:warmup * 2] ...)
for i in range(warmup, len(spread_bps)):
    z = (s - ou.mu) / ou.sigma          # ← static params
```

The C++ `backtest_engine` with `strategy="ou"` calls `ou_zscore()` at every
bar, which calls `estimate_ou_params_impl()` on a **trailing window**:

```cpp
for (size_t i = window; i < n; ++i) {
    z = ou_zscore(data, i + 1, window, dt);  // ← re-estimates per bar
```

Three compounding divergences:
1. **Static vs. rolling estimation** — C++ tracks regime changes; Python doesn't.
2. **Mean-reversion guard** — C++ returns NaN when `b >= 0` (not mean-reverting),
   skipping that bar. Python has no such guard — it computes z-scores even when
   `theta <= 0`, generating signals on non-mean-reverting windows.
3. **Different estimation sample** — Python uses a prefix `[:warmup*2]`; C++ uses
   a trailing window `[i+1-window : i+1]`.

**Impact:** Different trade lists, P&L, and Sharpe ratios depending on whether
the C++ module is compiled. A user running `backtest_historical.py` with vs.
without the `.pyd`/`.so` gets different results with no warning.

<details>
<summary>Investigation trail</summary>

Verified by reading backtest_historical.py L233, L244 (Python path) and
signal_engine.cpp L296, L174-178, L86 (C++ path). Sub-agent confirmed all
three sub-claims. Phase 3 adversarial refutation: could not refute.
</details>

---

### 2. Rolling z-score window off-by-one between Python and C++

**File:** `experiments/backtest_historical.py` — `run_zscore_strategy`
**File:** `cpp/signal_engine.cpp` — `rolling_zscore`

The Python backtest computes z-score statistics on a window that **excludes**
the current bar:

```python
lookback = spread_bps[i - window:i]   # indices {i-window, ..., i-1}
z = (s - mu) / sigma                  # s = spread_bps[i], not in lookback
```

The C++ `rolling_zscore` (called via `backtest_engine`) computes on a window
that **includes** the current bar:

```cpp
size_t start = n - window;   // n = i+1, so indices {i+1-window, ..., i}
// bar i IS in the window used for mean/std
return (data[n - 1] - mean) / std;
```

Including the current bar biases z-scores toward zero by ~1.7–3.5% at
window=60. A bar at exactly `z = 2.0` (entry threshold) under the Python
convention would read ~1.93–1.97 in C++, potentially **missing the entry
signal entirely**.

Note: the C++ convention matches `paper_trader.py`'s `compute_zscore_rolling`
(which also includes the current bar). The inconsistency is in the backtest's
Python fallback, not the C++ code per se — but the MR silently changes the
backtest behavior when the C++ module is present.

<details>
<summary>Investigation trail</summary>

Traced Python slice `[i-window:i]` (backtest_historical.py L323) vs C++
`rolling_zscore(data, i+1, window)` (signal_engine.cpp L242, L134-138).
Confirmed window shift and inclusion of current bar.
</details>

---

## Moderate

### 3. No GIL release during batch C++ operations

**File:** `cpp/signal_engine.cpp` — pybind11 module definition

The pybind11 bindings do not use `py::call_guard<py::gil_scoped_release>()`.
For batch operations (`batch_ou_signals`, `backtest`) that run 8–10ms, the
GIL is held the entire time. This prevents other Python threads from running.

In the current codebase (single-threaded asyncio in paper_trader, single-
threaded backtest loop), this has no practical impact. But it would block
multi-threaded callers (e.g., a ThreadPoolExecutor running backtests in
parallel) and is easy to fix — the batch functions only touch the raw `double*`
pointer after calling `buf.request()`, so releasing the GIL is safe.

### 4. `estimate_ou_params` wrapper copies array on every call

**File:** `experiments/paper_trader.py` — `estimate_ou_params`, `compute_zscore_rolling`, `compute_ou_zscore`

Every C++ dispatch does `spreads.astype(np.float64)`, which creates a full
copy even if the array is already float64. In live trading (called on every
tick), this adds ~1–2μs of unnecessary allocation per call. Should be
`np.ascontiguousarray(spreads, dtype=np.float64)` or a dtype check first.

---

## Minor

### 5. No `__repr__` on `TradeResult`

**File:** `cpp/signal_engine.cpp` — pybind11 `TradeResult` class

The bound `TradeResult` struct exposes `.entry_idx`, `.pnl_gross`, etc. as
read-only properties but has no `__repr__`. Printing a trade returns
`<signal_engine.TradeResult object at 0x...>` which is unhelpful during
debugging.

### 6. `build/` in .gitignore may be too broad

**File:** `.gitignore`

`build/` is a common directory name. If any future project component uses a
`build/` folder (e.g., docs), it would be silently ignored. Consider
`build/lib.*` or a comment clarifying this is for setuptools.

---

## Rust Perspective — Where It Could Add Value

The C++ engine is solid for what it does — 200 lines of arithmetic on
contiguous `f64` arrays. A rewrite in Rust would produce **identical runtime
performance** (same LLVM backend, same auto-vectorization). The interesting
question is where Rust's type system and ecosystem would change the
architecture:

### Where Rust adds genuine value

| Area | C++ (current) | Rust alternative | Delta |
|------|---------------|-----------------|-------|
| **Build system** | `setup.py` with platform-detection if/else for MSVC/Clang/GCC flags | `cargo build --release` — uniform on all platforms, no flag management | Eliminates 15 lines of platform branching |
| **Thread safety** | GIL must be manually released; shared data needs manual reasoning | `Send + Sync` enforced at compile time; safe to parallelize batch operations | Enables fearless parallelism for grid search |
| **Memory safety** | Raw pointer arithmetic (`const double* data`, `size_t n`). Correct here, but buffer overrun bugs are silent | Slice bounds checked at compile time; `&[f64]` can't outlive its source | Catches the class of bugs that *this code doesn't have* but future edits could introduce |
| **Error handling** | `bool valid` flag + NaN sentinels, must remember to check | `Result<OUParams, EstimationError>` — compiler forces callers to handle failure | Eliminates the "forgot to check NaN" class of bugs |
| **Dependency management** | pybind11 via pip, then setuptools ext_modules | PyO3 + maturin: `pip install maturin && maturin develop` | Slightly simpler, but PyO3 numpy interop is more verbose |

### Where Rust does NOT help

- **Runtime speed**: Identical. Both compile to the same LLVM IR for this workload.
- **NumPy interop ergonomics**: pybind11's `py::array_t<double>` with `.request()` is cleaner than PyO3's `PyReadonlyArray1<f64>`.
- **Quant finance signal**: C++ is the lingua franca at HRT, Citadel, Jump, Two Sigma. Rust is growing but still niche in quant.

### Concrete Rust opportunity: parallel grid search

The biggest performance win Rust could unlock is parallelizing the parameter
grid search (5 entry-z × 5 exit-z × 5 windows × 50 pairs = 6,250 backtests).
Currently each runs sequentially. In Rust with Rayon:

```rust
use rayon::prelude::*;

param_grid.par_iter().map(|(entry_z, exit_z, window, pair)| {
    backtest_engine(&spreads[pair], *window, *entry_z, *exit_z, fee, max_hold)
}).collect::<Vec<_>>()
```

This is safe by construction — the compiler proves no data races. In C++ you'd
need to manually reason about thread safety or use OpenMP with careful scoping.

**Bottom line:** For 200 lines of arithmetic, C++ is the right choice — mature
tooling, quant-standard, and already working. Rust would shine if the engine
grows to include concurrent data pipelines, parallel grid search, or becomes a
standalone service rather than a Python extension.

---

## Summary

Two confirmed bugs change backtest behavior when the C++ module is present vs
absent:

1. **OU strategy** estimates params once (Python) vs per-bar (C++), producing
   different trade lists. Fix: either make C++ match Python (estimate once
   upfront), or update Python to match C++ (rolling re-estimation) — the
   latter is arguably better but changes existing results.

2. **Rolling z-score** includes current bar in C++ but not in Python backtest,
   causing ~1.7–3.5% z-score compression. Fix: shift the C++ window back by
   one index, or update the Python backtest to include the current bar
   (matching `paper_trader.py`).

Both are silent — no error, no warning — and would only surface when comparing
results with vs without the compiled module. The C++ implementations are
internally correct and well-optimized; the issue is semantic equivalence with
the Python paths they replace.
