"""Convert DEX depth collector JSONL to Parquet."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pyarrow as pa
import pyarrow.parquet as pq

STREAMS = ("dex_quotes", "dex_gas", "perp_funding")


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
        for k in ("out_amount", "in_amount", "dst_amount", "src_amount"):
            if k in row:
                row[k] = str(row[k])
        if "protocols" in row and not isinstance(row.get("protocols"), str):
            row["protocols"] = json.dumps(row["protocols"])
        buf.append(row)
        kept += 1
        if len(buf) >= batch_rows:
            _flush()
    _flush()
    if writer is not None:
        writer.close()

    size = out_path.stat().st_size if out_path.exists() else 0
    return {"rows": total, "kept": kept, "bytes": size}


def main() -> None:
    parser = argparse.ArgumentParser(description="Export DEX depth JSONL to Parquet")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--min-snapshot", type=int, default=None)
    parser.add_argument("--max-snapshot", type=int, default=None)
    parser.add_argument("--batch-rows", type=int, default=50000)
    parser.add_argument("--compression", default="zstd",
                        choices=["zstd", "snappy", "gzip", "none"])
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    run_id = run_dir.name
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    compression = "none" if args.compression == "none" else args.compression

    for stream in STREAMS:
        stats = _export_stream(
            run_id, run_dir, out_dir, stream,
            args.min_snapshot, args.max_snapshot,
            max(1, args.batch_rows), compression,
        )
        print(f"  [{stream}] kept {stats['kept']} -> {stats['bytes']/1e6:.2f} MB")


if __name__ == "__main__":
    main()
