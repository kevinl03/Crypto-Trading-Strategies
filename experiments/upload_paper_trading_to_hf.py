"""
Assemble the ~8h paper-trading sessions into data/hf_upload/paper_trading_8h/
and upload that folder to Hugging Face as a pull request (not a direct main commit).

Auth (first match wins):
  1. HF_TOKEN environment variable
  2. huggingface_hub cached login (hf auth / huggingface-cli login)
  3. data/exports/.hf_token

Examples:
    set HF_TOKEN=hf_...
    python -m experiments.upload_paper_trading_to_hf

    python -m experiments.upload_paper_trading_to_hf --dry-run
    python -m experiments.upload_paper_trading_to_hf --direct-to-main   # opt-in
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from pathlib import Path

from huggingface_hub import HfApi
from huggingface_hub.utils import get_token

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_REPO = "SFU-fintech-AI/statarb-crypto-research"
REMOTE_FOLDER = "paper_trading_8h"
STAGING = ROOT / "data" / "hf_upload" / "paper_trading_8h"

# Source sessions used in the paper (local repo paths).
SESSIONS: dict[str, Path] = {
    "july30_lgbm_8h": ROOT / "data" / "paper_trading" / "lgbm_8h_20260730",
    "july31_lgbm_8h": ROOT / "data" / "paper_trading" / "July31st_8_hr",
}

# Keep research artifacts; drop empty restart stubs and agent noise.
JULY31_FILES = (
    "config.json",
    "summary.json",
    "trades.jsonl",
    "signals.jsonl",
    "signals_001.jsonl",
    "metrics_report.csv",
    "health_latest.json",
    "portfolio_sharpe_report.json",
    "jsonl_writer_state.json",
)
JULY31_DIRS = (
    "baseline_strengthening",
    "mechanical_z_baseline",
)

LOCAL_PATH_KEYS = ("model", "run_dir", "session")


def _token() -> str:
    env = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if env and env.strip():
        return env.strip()
    token = get_token()
    if token:
        return token
    for candidate in (
        ROOT / "data" / "exports" / ".hf_token",
        ROOT / "experiments" / ".hf_token",
    ):
        if candidate.exists():
            text = candidate.read_text(encoding="utf-8").strip()
            if text:
                os.environ["HF_TOKEN"] = text
                return text
    raise RuntimeError(
        "No Hugging Face token found. Set HF_TOKEN (preferred), run "
        "`hf auth login`, or place a token in data/exports/.hf_token"
    )


def _scrub_local_paths(obj: object) -> object:
    """Replace absolute local paths with basename / relative placeholders."""
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if k in LOCAL_PATH_KEYS and isinstance(v, str) and (":\\" in v or v.startswith("/")):
                out[k] = Path(v).name
            else:
                out[k] = _scrub_local_paths(v)
        return out
    if isinstance(obj, list):
        return [_scrub_local_paths(x) for x in obj]
    return obj


def _copy_json_scrubbed(src: Path, dst: Path) -> None:
    data = json.loads(src.read_text(encoding="utf-8"))
    dst.write_text(
        json.dumps(_scrub_local_paths(data), indent=2) + "\n",
        encoding="utf-8",
    )


def _copy_tree_scrubbing_json(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True, exist_ok=True)
    for path in src.rglob("*"):
        rel = path.relative_to(src)
        target = dst / rel
        if path.is_dir():
            target.mkdir(parents=True, exist_ok=True)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix.lower() == ".json":
            _copy_json_scrubbed(path, target)
        else:
            shutil.copy2(path, target)


def assemble_staging() -> Path:
    """Build data/hf_upload/paper_trading_8h/{july30,july31}_lgbm_8h/."""
    readme = STAGING / "README.md"
    readme_text = readme.read_text(encoding="utf-8") if readme.exists() else None

    # Clear prior session folders; keep/restore README.
    STAGING.mkdir(parents=True, exist_ok=True)
    for child in list(STAGING.iterdir()):
        if child.name == "README.md":
            continue
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()

    # July 30: full session tree (small).
    src30 = SESSIONS["july30_lgbm_8h"]
    if not src30.is_dir():
        raise SystemExit(f"Missing session folder: {src30}")
    _copy_tree_scrubbing_json(src30, STAGING / "july30_lgbm_8h")

    # July 31: selected artifacts only.
    src31 = SESSIONS["july31_lgbm_8h"]
    if not src31.is_dir():
        raise SystemExit(f"Missing session folder: {src31}")
    dst31 = STAGING / "july31_lgbm_8h"
    dst31.mkdir(parents=True, exist_ok=True)
    for name in JULY31_FILES:
        src = src31 / name
        if not src.exists():
            print(f"WARNING: missing {src}", flush=True)
            continue
        if src.suffix.lower() == ".json":
            _copy_json_scrubbed(src, dst31 / name)
        else:
            shutil.copy2(src, dst31 / name)
    for name in JULY31_DIRS:
        src = src31 / name
        if src.is_dir():
            _copy_tree_scrubbing_json(src, dst31 / name)

    if readme_text is not None:
        readme.write_text(readme_text, encoding="utf-8")
    elif not readme.exists():
        raise SystemExit(f"Missing dataset README: {readme}")

    files = sorted(p for p in STAGING.rglob("*") if p.is_file())
    total = sum(p.stat().st_size for p in files)
    print(f"staging={STAGING}", flush=True)
    print(f"files={len(files)}  bytes={total}", flush=True)
    for p in files:
        print(f"  {p.relative_to(STAGING)}  ({p.stat().st_size})", flush=True)
    return STAGING


def retry(fn, label: str, tries: int, sleep: int):
    for i in range(1, tries + 1):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001
            print(f"[{label} attempt {i}/{tries}] {type(e).__name__}: {e}", flush=True)
            time.sleep(sleep)
    raise SystemExit(f"GAVE_UP on {label}")


def upload(
    *,
    repo_id: str,
    create_pr: bool,
    commit_message: str,
    dry_run: bool,
    tries: int,
    sleep: int,
) -> None:
    staging = assemble_staging()
    print(f"repo={repo_id} (dataset)", flush=True)
    print(f"path_in_repo={REMOTE_FOLDER}/", flush=True)
    print(f"create_pr={create_pr}", flush=True)

    if dry_run:
        print("DRY_RUN - assembled staging only; no upload", flush=True)
        return

    token = _token()
    # Surface which auth path we used without printing the secret.
    if os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"):
        print("auth=HF_TOKEN env", flush=True)
    else:
        print("auth=huggingface_hub cache / .hf_token file", flush=True)

    def _do_upload():
        api = HfApi(token=token)
        return api.upload_folder(
            folder_path=str(staging),
            path_in_repo=REMOTE_FOLDER,
            repo_id=repo_id,
            repo_type="dataset",
            commit_message=commit_message,
            create_pr=create_pr,
            # Never delete remote files.
            ignore_patterns=["**/.git/**", "**/__pycache__/**"],
        )

    info = retry(_do_upload, "upload_folder", tries=tries, sleep=sleep)
    print("UPLOAD_DONE", flush=True)
    print(f"commit_url={getattr(info, 'commit_url', None) or info}", flush=True)
    pr_url = getattr(info, "pr_url", None)
    if pr_url:
        print(f"pr_url={pr_url}", flush=True)
    elif create_pr:
        print(
            "PR opened on the Hub (see commit_url / dataset Discussions tab).",
            flush=True,
        )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Stage the Jul 30 + Jul 31 ~8h paper-trading sessions and upload "
            "them to Hugging Face under paper_trading_8h/ as a Hub PR."
        )
    )
    p.add_argument("--repo-id", default=DEFAULT_REPO)
    p.add_argument(
        "--direct-to-main",
        action="store_true",
        help="Commit straight to the dataset main branch (default: open a Hub PR)",
    )
    p.add_argument(
        "--commit-message",
        default=(
            "Add paper_trading_8h/ (Jul 30 + Jul 31 LightGBM live paper sessions)"
        ),
    )
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--tries", type=int, default=40)
    p.add_argument("--sleep", type=int, default=15)
    p.add_argument(
        "--assemble-only",
        action="store_true",
        help="Only rebuild the local staging folder",
    )
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.assemble_only:
        assemble_staging()
        return
    upload(
        repo_id=args.repo_id,
        create_pr=not args.direct_to_main,
        commit_message=args.commit_message,
        dry_run=args.dry_run,
        tries=args.tries,
        sleep=args.sleep,
    )


if __name__ == "__main__":
    main()
