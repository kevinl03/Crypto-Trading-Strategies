"""
Assemble the ~3-day (Aug 4–7) paper-trading campaign into
data/hf_upload/paper_trading_3day/ and upload to Hugging Face.

Default: open a Hub PR on SFU-fintech-AI/statarb-crypto-research
(same pattern as experiments/upload_paper_trading_to_hf.py).

Auth (first match wins):
  1. HF_TOKEN environment variable
  2. huggingface_hub cached login (hf auth / huggingface-cli login)
  3. data/exports/.hf_token

Examples:
    python -m experiments.upload_paper_trading_3day_to_hf --dry-run
    python -m experiments.upload_paper_trading_3day_to_hf
    python -m experiments.upload_paper_trading_3day_to_hf --direct-to-main
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from datetime import datetime
from pathlib import Path

from huggingface_hub import HfApi
from huggingface_hub.utils import get_token

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_REPO = "SFU-fintech-AI/statarb-crypto-research"
REMOTE_FOLDER = "paper_trading_3day"
STAGING = ROOT / "data" / "hf_upload" / "paper_trading_3day"
SESSION_SRC = ROOT / "data" / "paper_trading" / "5day_Aug4_2026"
SESSION_DST_NAME = "aug4_lgbm_3day"
METRICS_SRC = ROOT / "docs" / "72h_live_campaign_paper_metrics.md"

# Explicit allow-list (skip friction dumps, logs, pre_restart archives).
COPY_EXACT = (
    "summary.json",
    "config.json",
    "session_config.json",
    "dashboard.json",
    "portfolio_sharpe_report.json",
    "sim_persistence_hold_report.json",
    "jsonl_writer_state.json",
)
COPY_GLOBS = (
    "trades.jsonl",
    "trades_*.jsonl",
    "signals.jsonl",
    "signals_*.jsonl",
)

LOCAL_PATH_KEYS = ("model", "run_dir", "session", "session_dir", "model_path", "resume_run_dir")


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


def _first_jsonl_ts(path: Path) -> datetime | None:
    """Return timezone-aware datetime from the first JSONL record's ``ts``."""
    with path.open(encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line).get("ts")
            except json.JSONDecodeError:
                return None
            if not raw:
                return None
            return datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
    return None


def _dated_signals_name(src: Path) -> str:
    """e.g. signals.jsonl -> signals_aug04_0842Z.jsonl (month+day + HHMM UTC)."""
    ts = _first_jsonl_ts(src)
    if ts is None:
        # Fallback: keep original stem if unreadable
        return src.name
    # Military time = 24h HHMM; Z marks UTC (keep Z uppercase)
    stamp = ts.strftime("%b%d").lower() + ts.strftime("_%H%MZ")  # aug04_0842Z
    return f"signals_{stamp}.jsonl"


def assemble_staging() -> Path:
    readme = STAGING / "README.md"
    if not readme.exists():
        raise SystemExit(f"Missing dataset README: {readme}")
    readme_text = readme.read_text(encoding="utf-8")

    if not SESSION_SRC.is_dir():
        raise SystemExit(f"Missing session folder: {SESSION_SRC}")

    STAGING.mkdir(parents=True, exist_ok=True)
    for child in list(STAGING.iterdir()):
        if child.name == "README.md":
            continue
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()

    dst = STAGING / SESSION_DST_NAME
    dst.mkdir(parents=True, exist_ok=True)

    copied: list[Path] = []
    for name in COPY_EXACT:
        src = SESSION_SRC / name
        if not src.exists():
            print(f"WARNING: missing {src}", flush=True)
            continue
        if src.suffix.lower() == ".json":
            _copy_json_scrubbed(src, dst / name)
        else:
            shutil.copy2(src, dst / name)
        copied.append(dst / name)

    used_names: set[str] = set()
    for pattern in COPY_GLOBS:
        for src in sorted(SESSION_SRC.glob(pattern)):
            if not src.is_file():
                continue
            if src.name.startswith("signals"):
                name = _dated_signals_name(src)
                # Collision guard (same HHMM): append original shard suffix
                if name in used_names:
                    stem = Path(name).stem
                    name = f"{stem}_{src.stem}.jsonl"
                used_names.add(name)
            else:
                name = src.name
            shutil.copy2(src, dst / name)
            copied.append(dst / name)
            if name != src.name:
                print(f"  rename {src.name} -> {name}", flush=True)

    if METRICS_SRC.exists():
        shutil.copy2(METRICS_SRC, dst / "METRICS.md")
        copied.append(dst / "METRICS.md")
    else:
        print(f"WARNING: missing metrics writeup {METRICS_SRC}", flush=True)

    readme.write_text(readme_text, encoding="utf-8")

    files = sorted(p for p in STAGING.rglob("*") if p.is_file())
    total = sum(p.stat().st_size for p in files)
    print(f"staging={STAGING}", flush=True)
    print(f"files={len(files)}  bytes={total}  (~{total / 1e6:.1f} MB)", flush=True)
    for p in files:
        print(f"  {p.relative_to(STAGING)}  ({p.stat().st_size})", flush=True)
    if len(copied) < 5:
        raise SystemExit("Staging looks incomplete — aborting")
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
            "Stage the Aug 4–7 ~3-day LightGBM paper-trading campaign and "
            "upload it to Hugging Face under paper_trading_3day/."
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
            "Add paper_trading_3day/ (Aug 4–7 LightGBM live campaign, ~73h, 50.7k trades)"
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
