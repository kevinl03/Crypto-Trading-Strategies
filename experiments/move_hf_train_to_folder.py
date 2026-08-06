"""
Move Hub root train parquets into train/ via a Hugging Face PR (not main).

Uses server-side copy + delete so large files are not re-uploaded.

Auth: HF_TOKEN / HUGGING_FACE_HUB_TOKEN / huggingface_hub login.

Example:
    python -m experiments.move_hf_train_to_folder
    python -m experiments.move_hf_train_to_folder --dry-run
"""
from __future__ import annotations

import argparse
import os
import re
import time
from pathlib import Path

from huggingface_hub import (
    CommitOperationAdd,
    CommitOperationCopy,
    CommitOperationDelete,
    HfApi,
    hf_hub_download,
)
from huggingface_hub.utils import get_token

DEFAULT_REPO = "SFU-fintech-AI/statarb-crypto-research"


def retry(fn, label: str, tries: int = 8, sleep: int = 5):
    for i in range(1, tries + 1):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001
            print(f"[{label} {i}/{tries}] {type(e).__name__}: {e}", flush=True)
            time.sleep(sleep)
    raise SystemExit(f"GAVE_UP on {label}")

TRAIN_FILES = (
    "ohlcv.parquet",
    "ticker.parquet",
    "orderbook.parquet",
    "trades.parquet",
    "spread_matrix.parquet",
    "funding_rate.parquet",
    "open_interest.parquet",
    "withdrawal_status.parquet",
    "exchange_status.parquet",
)


def _token() -> str:
    env = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if env and env.strip():
        return env.strip()
    token = get_token()
    if token:
        return token
    raise RuntimeError("No HF token (set HF_TOKEN or run hf auth login)")


def _rewrite_readme(text: str) -> str:
    out = text
    for name in TRAIN_FILES:
        # Only rewrite bare root data_files entries, not test/... paths.
        out = re.sub(
            rf"(?m)^(\s*data_files:\s*){re.escape(name)}\s*$",
            rf"\1train/{name}",
            out,
        )
    out = out.replace(
        "## Train split (root-level Parquet files)",
        "## Train split (`train/` Parquet files)",
    )
    out = out.replace(
        "Train split (root-level",
        "Train split (`train/`",
    )
    return out


def build_operations(readme_text: str, remote_files: set[str]) -> list:
    ops: list = []
    for name in TRAIN_FILES:
        if name not in remote_files:
            raise SystemExit(f"Missing root train file on Hub: {name}")
        dest = f"train/{name}"
        if dest in remote_files:
            raise SystemExit(f"Destination already exists: {dest}")
        ops.append(CommitOperationCopy(src_path_in_repo=name, path_in_repo=dest))
        ops.append(CommitOperationDelete(path_in_repo=name))

    staging = Path(__file__).resolve().parent.parent / "data" / "hf_upload" / "_tmp_readme"
    staging.mkdir(parents=True, exist_ok=True)
    readme_path = staging / "README.md"
    readme_path.write_text(readme_text, encoding="utf-8")
    ops.append(
        CommitOperationAdd(path_in_repo="README.md", path_or_fileobj=str(readme_path))
    )
    return ops


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo-id", default=DEFAULT_REPO)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument(
        "--direct-to-main",
        action="store_true",
        help="Commit to dataset main (default: open a Hub PR)",
    )
    args = p.parse_args(argv)

    token = _token()
    api = HfApi(token=token)
    remote = set(
        retry(lambda: api.list_repo_files(args.repo_id, repo_type="dataset"), "list")
    )
    local_readme = retry(
        lambda: hf_hub_download(
            args.repo_id, "README.md", repo_type="dataset", token=token
        ),
        "readme",
    )
    new_readme = _rewrite_readme(Path(local_readme).read_text(encoding="utf-8"))
    ops = build_operations(new_readme, remote)

    print(f"repo={args.repo_id}")
    print(f"create_pr={not args.direct_to_main}")
    print(f"operations={len(ops)}")
    for op in ops:
        print(f"  {op}")

    if args.dry_run:
        print("DRY_RUN - no commit")
        return

    info = retry(
        lambda: api.create_commit(
            repo_id=args.repo_id,
            repo_type="dataset",
            operations=ops,
            commit_message="Move train split parquets from repo root into train/",
            commit_description=(
                "Relocate the Jun 13–16 train window into train/ to match test/ and "
                "validation/ layout. Dataset config names are unchanged; only "
                "data_files paths and the card wording are updated."
            ),
            create_pr=not args.direct_to_main,
        ),
        "create_commit",
        tries=12,
        sleep=10,
    )
    print("DONE")
    print(f"commit_url={getattr(info, 'commit_url', info)}")
    pr_url = getattr(info, "pr_url", None)
    if pr_url:
        print(f"pr_url={pr_url}")


if __name__ == "__main__":
    main()
