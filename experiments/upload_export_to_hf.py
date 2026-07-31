"""
Upload a local parquet export to Hugging Face as a named repo folder.

Follows the Hub upload guide:
  https://huggingface.co/docs/huggingface_hub/guides/upload

  - Prefer upload_folder() (not per-file upload_file loops)
  - Do not use deprecated upload_large_folder()
  - With hf_xet, interrupted uploads resume by re-running the same call
  - Sets HF_XET_HIGH_PERFORMANCE=1 by default

Safety (learned the hard way):
  - Never uses delete_patterns / delete_folder / delete_file
  - Refuses to upload into a remote path that already has matching
    parquet files unless --overwrite is passed
  - Refuses repo-root uploads unless --allow-root AND --overwrite
  - Always prefer a NEW --remote-folder for a new dataset split
    (e.g. validation_jul22-28), never clobber validation/ or train/

Auth: HF_TOKEN, `huggingface-cli login`, or data/exports/.hf_token

Examples:
    python -m experiments.upload_export_to_hf \\
        --export-dir data/exports/statarb_validation_20260722_20260728 \\
        --remote-folder validation_jul22-28

    python -m experiments.upload_export_to_hf \\
        --export-dir data/exports/my_new_run \\
        --remote-folder validation_aug01-07 \\
        --commit-message "Add August validation split"
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import time
from pathlib import Path

from huggingface_hub import HfApi
from huggingface_hub.utils import get_token

DEFAULT_REPO = "SFU-fintech-AI/statarb-crypto-research"
DEFAULT_PATTERNS = ["*.parquet", "README.md"]

# Historical splits we must never silently clobber.
PROTECTED_FOLDERS = frozenset(
    {
        "validation",
        "test",
        "validation_jul19-22",
        "validation_jul22-28",
    }
)


def _token() -> str:
    token = get_token()
    if token:
        return token
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


def _enable_xet_high_performance(enabled: bool) -> None:
    if not enabled:
        return
    os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")
    if importlib.util.find_spec("hf_xet") is None:
        print(
            "WARNING: hf_xet not installed; upload_folder falls back to "
            "legacy HTTP hashing. Install with: "
            'pip install -U "huggingface_hub[hf_xet]" '
            "(or upgrade huggingface_hub >= 0.32)",
            flush=True,
        )


def _normalize_remote_folder(raw: str) -> str:
    folder = raw.strip().strip("/\\")
    if not folder or folder in {".", ".."}:
        raise SystemExit(
            "Refusing empty/root --remote-folder. Pass an explicit folder "
            "name (e.g. validation_jul22-28). Use --allow-root to upload "
            "to the repo root."
        )
    if "\\" in folder:
        raise SystemExit("--remote-folder must use / separators, not \\")
    return folder


def _local_parquets(export_dir: Path) -> list[Path]:
    return sorted(export_dir.glob("*.parquet"))


def _remote_files_under(files: set[str], path_in_repo: str) -> set[str]:
    if not path_in_repo:
        # Repo root: only bare filenames (no '/'), typically train parquets.
        return {f for f in files if "/" not in f and f.endswith(".parquet")}
    prefix = path_in_repo.rstrip("/") + "/"
    return {f for f in files if f.startswith(prefix)}


def _conflicts(existing: set[str], expected: set[str]) -> list[str]:
    return sorted(existing & expected)


def retry(fn, label: str, tries: int, sleep: int):
    """Hub docs: expect failures on large streams; re-run the same call."""
    for i in range(1, tries + 1):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001
            print(f"[{label} attempt {i}/{tries}] {type(e).__name__}: {e}", flush=True)
            time.sleep(sleep)
    raise SystemExit(f"GAVE_UP on {label}")


def upload_export(
    *,
    export_dir: Path,
    remote_folder: str | None,
    repo_id: str,
    repo_type: str,
    allow_patterns: list[str],
    commit_message: str,
    tries: int,
    sleep: int,
    dry_run: bool,
    allow_root: bool,
    overwrite: bool,
) -> None:
    if not export_dir.is_dir():
        raise SystemExit(f"Export folder missing: {export_dir}")

    parquets = _local_parquets(export_dir)
    if not parquets:
        raise SystemExit(f"No *.parquet files in {export_dir}")

    if allow_root and remote_folder in (None, "", ".", "/"):
        if not overwrite:
            raise SystemExit(
                "Refusing repo-root upload without --overwrite "
                "(would risk clobbering train/*.parquet). "
                "Prefer a new --remote-folder instead."
            )
        path_in_repo = ""
        remote_prefix = ""
    else:
        if remote_folder is None:
            raise SystemExit("--remote-folder is required (or pass --allow-root)")
        path_in_repo = _normalize_remote_folder(remote_folder)
        remote_prefix = f"{path_in_repo}/"
        top = path_in_repo.split("/", 1)[0]
        if top in PROTECTED_FOLDERS and not overwrite:
            raise SystemExit(
                f"Refusing upload into protected folder '{top}/' without "
                f"--overwrite. Create a NEW folder name for new data "
                f"(e.g. validation_aug01-07), or pass --overwrite only if "
                f"you intentionally replace files in '{top}/'."
            )

    expected = {f"{remote_prefix}{p.name}" for p in parquets}
    print(f"export_dir={export_dir.resolve()}", flush=True)
    print(f"repo={repo_id} ({repo_type})", flush=True)
    print(f"path_in_repo={path_in_repo or '(repo root)'}", flush=True)
    print(f"overwrite={overwrite}", flush=True)
    print(f"files={len(parquets)} parquet(s)", flush=True)
    for p in parquets:
        print(f"  {remote_prefix}{p.name}  ({p.stat().st_size} bytes)", flush=True)

    token = _token() if not dry_run else (get_token() or "")
    # Always try to list for conflict checks when we have credentials.
    if not token and not dry_run:
        token = _token()

    def _list() -> set[str]:
        api_token = token or _token()
        return set(HfApi(token=api_token).list_repo_files(repo_id, repo_type=repo_type))

    try:
        remote_files = retry(_list, "list", tries=tries, sleep=sleep)
    except SystemExit:
        if dry_run:
            print(
                "DRY_RUN: could not list remote files; skipping conflict check",
                flush=True,
            )
            print("DRY_RUN - no upload", flush=True)
            return
        raise

    under = _remote_files_under(remote_files, path_in_repo)
    conflicts = _conflicts(under, expected)
    siblings = sorted(under - expected)

    if under:
        print(
            f"remote already has {len(under)} file(s) under "
            f"{path_in_repo or '(repo root)'}/",
            flush=True,
        )
    if siblings:
        print(
            f"note: {len(siblings)} remote file(s) in this folder are not in "
            f"this upload (left untouched; we never delete):",
            flush=True,
        )
        for f in siblings[:20]:
            print(f"  keep {f}", flush=True)
        if len(siblings) > 20:
            print(f"  ... +{len(siblings) - 20} more", flush=True)

    if conflicts and not overwrite:
        print("CONFLICT: these remote paths already exist:", flush=True)
        for f in conflicts:
            print(f"  {f}", flush=True)
        raise SystemExit(
            "Refusing to overwrite existing remote parquets. "
            "Pick a new --remote-folder for the new split, or pass "
            "--overwrite to replace ONLY the conflicting files "
            "(siblings in the folder are still never deleted)."
        )
    if conflicts and overwrite:
        print(
            f"WARNING: --overwrite will replace {len(conflicts)} existing "
            f"file(s); other files in the repo are untouched:",
            flush=True,
        )
        for f in conflicts:
            print(f"  replace {f}", flush=True)

    if dry_run:
        print("DRY_RUN - no upload", flush=True)
        return

    def _do_upload() -> None:
        # Fresh client each attempt: avoids 'client has been closed' after drops.
        # Intentionally never passes delete_patterns — a bad revert must not
        # wipe an entire folder (e.g. validation/).
        api = HfApi(token=token)
        api.upload_folder(
            folder_path=str(export_dir),
            path_in_repo=path_in_repo,
            repo_id=repo_id,
            repo_type=repo_type,
            allow_patterns=allow_patterns,
            commit_message=commit_message,
        )

    print(
        f"upload_folder "
        f"(HF_XET_HIGH_PERFORMANCE={os.environ.get('HF_XET_HIGH_PERFORMANCE')})",
        flush=True,
    )
    retry(_do_upload, "upload_folder", tries=tries, sleep=sleep)
    print("UPLOAD_DONE", flush=True)

    files = retry(_list, "list", tries=tries, sleep=sleep)
    missing = sorted(expected - files)
    present = sorted(expected & files)
    print(f"present {len(present)}/{len(expected)}", flush=True)
    for f in present:
        print(f"  {f}", flush=True)
    if missing:
        raise SystemExit(f"INCOMPLETE missing={missing}")
    print("ALL_DONE", flush=True)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Upload a local parquet export folder to Hugging Face under a "
            "new (or existing) remote folder name. Never deletes remote files."
        )
    )
    p.add_argument(
        "--export-dir",
        type=Path,
        required=True,
        help="Local folder containing *.parquet (e.g. data/exports/my_run)",
    )
    p.add_argument(
        "--remote-folder",
        type=str,
        default=None,
        help="Folder to create/update in the HF repo (e.g. validation_jul22-28)",
    )
    p.add_argument(
        "--repo-id",
        default=DEFAULT_REPO,
        help=f"HF dataset/model/space id (default: {DEFAULT_REPO})",
    )
    p.add_argument(
        "--repo-type",
        default="dataset",
        choices=["dataset", "model", "space"],
    )
    p.add_argument(
        "--allow-patterns",
        nargs="+",
        default=DEFAULT_PATTERNS,
        help="Glob patterns relative to export-dir (default: *.parquet README.md)",
    )
    p.add_argument(
        "--commit-message",
        default=None,
        help="Commit message (default: derived from remote folder name)",
    )
    p.add_argument("--tries", type=int, default=80, help="Retries for flaky uploads")
    p.add_argument("--sleep", type=int, default=20, help="Seconds between retries")
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="List files / conflicts that would be uploaded, then exit",
    )
    p.add_argument(
        "--allow-root",
        action="store_true",
        help="Allow uploading into the repo root (also requires --overwrite)",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Allow replacing remote files that already exist at the target "
            "paths. Does NOT delete other files in the folder or repo."
        ),
    )
    p.add_argument(
        "--no-xet-high-performance",
        action="store_true",
        help="Do not set HF_XET_HIGH_PERFORMANCE=1",
    )
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    _enable_xet_high_performance(not args.no_xet_high_performance)

    remote = args.remote_folder
    if args.commit_message:
        commit_message = args.commit_message
    elif remote:
        commit_message = f"Add/update {remote.strip().strip('/')} export"
    else:
        commit_message = "Add/update export at repo root"

    upload_export(
        export_dir=args.export_dir,
        remote_folder=remote,
        repo_id=args.repo_id,
        repo_type=args.repo_type,
        allow_patterns=list(args.allow_patterns),
        commit_message=commit_message,
        tries=args.tries,
        sleep=args.sleep,
        dry_run=args.dry_run,
        allow_root=args.allow_root,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
