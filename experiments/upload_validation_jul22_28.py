"""
Upload Jul 22–28 long-run CEX parquets as a NEW root folder on HF.

Target (does NOT touch train root, test/, validation/, or validation_jul19-22/):

    validation_jul22-28/*.parquet

Same naming pattern as the existing validation_jul19-22/ folder.

Auth: HF_TOKEN, huggingface-cli login, or data/exports/.hf_token

Usage:
    python -m experiments.upload_validation_jul22_28
"""
from __future__ import annotations

import os
import shutil
import time
from pathlib import Path

from huggingface_hub import HfApi
from huggingface_hub.utils import get_token

REPO = "SFU-fintech-AI/statarb-crypto-research"
EXPORT = Path("data/exports/statarb_validation_20260722_20260728")
REMOTE_FOLDER = "validation_jul22-28"  # sibling of validation_jul19-22/
SIGNALS = [
    "ticker",
    "orderbook",
    "trades",
    "spread_matrix",
    "funding_rate",
    "open_interest",
    "withdrawal_status",
    "exchange_status",
    "long_short_ratio",
    "liquidations",
]


def _token() -> str:
    token = get_token()
    if token:
        return token
    # Prefer the historical exports token location used by prior upload scripts.
    repo_root = Path(__file__).resolve().parent.parent
    for candidate in (
        repo_root / "data" / "exports" / ".hf_token",
        Path(__file__).resolve().parent / ".hf_token",
    ):
        if candidate.exists():
            token = candidate.read_text(encoding="utf-8").strip()
            if token:
                os.environ["HF_TOKEN"] = token
                return token
    raise RuntimeError(
        "No HF token found (set HF_TOKEN, huggingface-cli login, "
        "or data/exports/.hf_token)"
    )


def retry(fn, label, tries=80, sleep=20):
    for i in range(1, tries + 1):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001
            print(f"[{label} attempt {i}] {type(e).__name__}: {e}", flush=True)
            time.sleep(sleep)
    raise SystemExit(f"GAVE_UP on {label}")


def _stage() -> Path:
    """
    Stage as <export>/validation_jul22-28/*.parquet so upload_folder
    preserves that path prefix at the repo root.
    """
    staged = EXPORT / REMOTE_FOLDER
    # remove old mistaken 'validation' stage dir if present
    old = EXPORT / "validation"
    if old.exists():
        shutil.rmtree(old)
    if staged.exists():
        shutil.rmtree(staged)
    staged.mkdir(parents=True)

    readme_src = EXPORT / "README.md"
    if readme_src.exists():
        shutil.copy2(readme_src, staged / "README.md")

    for signal in SIGNALS:
        src = EXPORT / f"{signal}.parquet"
        if not src.exists():
            raise SystemExit(f"Missing {src}")
        dst = staged / f"{signal}.parquet"
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)
    return staged


def main() -> None:
    _token()
    if not EXPORT.exists():
        raise SystemExit(f"Export folder missing: {EXPORT}")

    staged_parent = EXPORT  # contains validation_jul22-28/
    staged = _stage()
    print(f"staged={staged}", flush=True)

    api = HfApi(token=_token())
    # upload only the new root folder — never train/test/validation/
    retry(
        lambda: api.upload_folder(
            folder_path=str(staged_parent),
            repo_id=REPO,
            repo_type="dataset",
            path_in_repo="",  # repo root
            allow_patterns=[f"{REMOTE_FOLDER}/*"],
            commit_message=(
                "Add validation_jul22-28/ long-run CEX signals "
                "(incl. long_short_ratio + liquidations)"
            ),
        ),
        "upload_folder",
    )
    print("UPLOAD_DONE", flush=True)

    files = set(retry(lambda: api.list_repo_files(REPO, repo_type="dataset"), "list"))
    expected = {f"{REMOTE_FOLDER}/{s}.parquet" for s in SIGNALS}
    missing = sorted(expected - files)
    present = sorted(expected & files)
    print(f"present {len(present)}/{len(expected)}", flush=True)
    for f in present:
        print(f"  {f}", flush=True)
    # ensure we did NOT leave junk in validation/
    leftover = sorted(f for f in files if f.startswith("validation/"))
    if leftover:
        print("WARNING leftover under validation/:", leftover, flush=True)
    if missing:
        raise SystemExit(f"INCOMPLETE missing={missing}")
    print("ALL_DONE", flush=True)


if __name__ == "__main__":
    main()
