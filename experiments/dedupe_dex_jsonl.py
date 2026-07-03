"""Deduplicate dex_pools.jsonl and dex_spreads.jsonl for a run directory.

Keeps the first row per unique key. Safe for snapshots 0-10 (no-op) and
snapshots 11+ where duplicate collectors doubled row counts.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys


def _validate_lines(path: str) -> tuple[int, int]:
    """Return (valid_lines, bad_lines)."""
    valid = bad = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                json.loads(line)
                valid += 1
            except json.JSONDecodeError:
                bad += 1
    return valid, bad


def dedupe_pools(path: str) -> tuple[int, int]:
    seen: set[tuple] = set()
    kept: list[str] = []
    dropped = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            raw = line.rstrip("\n")
            if not raw.strip():
                continue
            r = json.loads(raw)
            key = (r["snapshot_idx"], r["token"], r["chain"], r["dex"], r["pair_address"])
            if key in seen:
                dropped += 1
                continue
            seen.add(key)
            kept.append(raw)
    return len(kept), dropped


def dedupe_spreads(path: str) -> tuple[int, int]:
    seen: set[tuple] = set()
    kept: list[str] = []
    dropped = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            raw = line.rstrip("\n")
            if not raw.strip():
                continue
            r = json.loads(raw)
            key = (r["snapshot_idx"], r["token"])
            if key in seen:
                dropped += 1
                continue
            seen.add(key)
            kept.append(raw)
    return len(kept), dropped


def _atomic_write(path: str, lines: list[str]) -> None:
    tmp = path + ".dedupe.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")
        f.flush()
        os.fsync(f.fileno())
    shutil.move(tmp, path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Dedupe DEX JSONL run files")
    parser.add_argument("run_dir", help="e.g. data/dex/20260624_101824")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    pools = os.path.join(args.run_dir, "dex_pools.jsonl")
    spreads = os.path.join(args.run_dir, "dex_spreads.jsonl")
    for p in (pools, spreads):
        if not os.path.isfile(p):
            print(f"Missing: {p}", file=sys.stderr)
            sys.exit(1)

    for p in (pools, spreads):
        valid, bad = _validate_lines(p)
        print(f"{os.path.basename(p)}: {valid} valid lines, {bad} corrupt lines")
        if bad:
            print("Fix corrupt lines before deduping.", file=sys.stderr)
            sys.exit(1)

    # load + dedupe
    pool_kept, pool_drop = dedupe_pools(pools)
    spread_kept, spread_drop = dedupe_spreads(spreads)
    print(f"pools: keep {pool_kept}, drop {pool_drop}")
    print(f"spreads: keep {spread_kept}, drop {spread_drop}")

    if args.dry_run:
        return

    # re-read kept lines for atomic write
    for path, fn in ((pools, dedupe_pools), (spreads, dedupe_spreads)):
        seen: set[tuple] = set()
        kept: list[str] = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                raw = line.rstrip("\n")
                if not raw.strip():
                    continue
                r = json.loads(raw)
                if path.endswith("pools.jsonl"):
                    key = (r["snapshot_idx"], r["token"], r["chain"], r["dex"], r["pair_address"])
                else:
                    key = (r["snapshot_idx"], r["token"])
                if key in seen:
                    continue
                seen.add(key)
                kept.append(raw)
        bak = path + ".pre_dedupe.bak"
        if not os.path.exists(bak):
            shutil.copy2(path, bak)
        _atomic_write(path, kept)
        print(f"wrote {path} ({len(kept)} rows), backup at {bak}")


if __name__ == "__main__":
    main()
