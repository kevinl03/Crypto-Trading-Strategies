"""Create HF dataset repo and upload DEX Parquet export."""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import HfApi


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload DEX Parquet export to Hugging Face")
    parser.add_argument(
        "--export-dir",
        default="data/exports/dex_96h_20260626",
        help="Directory with README.md, dex_pools.parquet, dex_spreads.parquet",
    )
    parser.add_argument(
        "--repo-id",
        default="SFU-fintech-AI/statarb-crypto-dex",
    )
    parser.add_argument("--commit-message", default="Upload DEX parquet export")
    args = parser.parse_args()

    export_dir = Path(args.export_dir).resolve()
    if not export_dir.is_dir():
        raise SystemExit(f"Export dir not found: {export_dir}")
    for name in ("README.md", "dex_pools.parquet", "dex_spreads.parquet"):
        if not (export_dir / name).exists():
            raise SystemExit(f"Missing required file: {export_dir / name}")

    api = HfApi()
    user = api.whoami()
    print(f"Authenticated as {user.get('name')}")

    api.create_repo(
        repo_id=args.repo_id,
        repo_type="dataset",
        exist_ok=True,
    )
    print(f"Repo ready: https://huggingface.co/datasets/{args.repo_id}")

    api.upload_folder(
        folder_path=str(export_dir),
        repo_id=args.repo_id,
        repo_type="dataset",
        commit_message=args.commit_message,
    )
    print("Upload complete.")


if __name__ == "__main__":
    main()
