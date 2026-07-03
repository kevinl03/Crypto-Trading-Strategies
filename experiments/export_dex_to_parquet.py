"""
Convert a DEX collector run (dex_pools.jsonl + dex_spreads.jsonl) to Parquet.

Output is separate from CEX stat-arb exports under data/exports/.

Usage:
    python -m experiments.export_dex_to_parquet \
        --run-dir data/dex/20260626_120000 \
        --out-dir data/exports/dex_96h_20260626
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pyarrow as pa
import pyarrow.parquet as pq

STREAMS = ("dex_pools", "dex_spreads")


def _iter_jsonl(path: Path) -> Iterable[Dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue


def _export_stream(
    run_id: str,
    run_dir: Path,
    out_dir: Path,
    stream: str,
    min_snapshot: Optional[int],
    max_snapshot: Optional[int],
    batch_rows: int,
    compression: str,
) -> Dict[str, int]:
    src = run_dir / f"{stream}.jsonl"
    if not src.exists():
        return {"rows": 0, "kept": 0, "bytes": 0}

    out_path = out_dir / f"{stream}.parquet"
    writer: Optional[pq.ParquetWriter] = None
    buf: List[Dict] = []
    total = kept = 0

    def _flush():
        nonlocal writer, buf
        if not buf:
            return
        table = pa.Table.from_pylist(buf)
        if writer is None:
            writer = pq.ParquetWriter(out_path, table.schema, compression=compression)
        writer.write_table(table)
        buf = []

    for rec in _iter_jsonl(src):
        total += 1
        si = rec.get("snapshot_idx", rec.get("snapshot"))
        if max_snapshot is not None and isinstance(si, (int, float)) and si > max_snapshot:
            continue
        if min_snapshot is not None and isinstance(si, (int, float)) and si < min_snapshot:
            continue
        row = {"run_id": run_id, **rec}
        kept += 1
        buf.append(row)
        if len(buf) >= batch_rows:
            _flush()
    _flush()
    if writer is not None:
        writer.close()

    size = out_path.stat().st_size if out_path.exists() else 0
    return {"rows": total, "kept": kept, "bytes": size}


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert DEX JSONL run to Parquet.")
    parser.add_argument("--run-dir", required=True, help="e.g. data/dex/20260626_120000")
    parser.add_argument("--out-dir", required=True, help="e.g. data/exports/dex_96h_20260626")
    parser.add_argument("--min-snapshot", type=int, default=None)
    parser.add_argument("--max-snapshot", type=int, default=None)
    parser.add_argument("--batch-rows", type=int, default=50000)
    parser.add_argument("--compression", default="zstd",
                        choices=["zstd", "snappy", "gzip", "none"])
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    if not run_dir.exists():
        raise SystemExit(f"[ERROR] Run dir not found: {run_dir}")
    run_id = run_dir.name
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    compression = "none" if args.compression == "none" else args.compression

    print(f"[START] run_id={run_id}")
    print(f"  run_dir={run_dir}")
    print(f"  out_dir={out_dir}")
    print(f"  snapshot_range=[{args.min_snapshot}, {args.max_snapshot}]")

    grand_kept = grand_bytes = 0
    for stream in STREAMS:
        stats = _export_stream(
            run_id=run_id,
            run_dir=run_dir,
            out_dir=out_dir,
            stream=stream,
            min_snapshot=args.min_snapshot,
            max_snapshot=args.max_snapshot,
            batch_rows=max(1, args.batch_rows),
            compression=compression,
        )
        grand_kept += stats["kept"]
        grand_bytes += stats["bytes"]
        mb = stats["bytes"] / 1e6
        print(f"  [{stream:<14}] kept {stats['kept']:>8} / {stats['rows']:>8} rows  ->  {mb:8.1f} MB")

    print(f"[DONE] kept_total={grand_kept}  parquet_total={grand_bytes/1e6:.1f} MB")


if __name__ == "__main__":
    main()
