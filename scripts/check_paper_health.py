"""Quick health check for the live LGBM paper session. Exit 0=ok, 1=warn, 2=hard fail."""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(r"C:\Users\Kev\repos\stochastic-spread-modeling")
RUN = REPO / "data" / "statarb" / "20260801_025316"
OUT = REPO / "data" / "paper_trading" / "July31st_8_hr"
DEADLINE = datetime(2026, 8, 1, 14, 55, 33, tzinfo=timezone.utc)
WARMUP_SNAPS = 90


def main() -> int:
    now = datetime.now(timezone.utc)
    issues: list[str] = []
    hard = False

    spread_files = sorted((RUN / "spread_matrix").glob("*.jsonl")) if (RUN / "spread_matrix").exists() else []
    if not spread_files:
        issues.append("no spread_matrix jsonl")
        hard = True
    else:
        age = now.timestamp() - spread_files[-1].stat().st_mtime
        if age > 300:
            issues.append(f"spread stale age_s={age:.0f}")
            hard = True

    summary_path = OUT / "summary.json"
    summary = {}
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    else:
        issues.append("missing summary.json")
        hard = True

    n_snaps = int(summary.get("n_snaps") or 0)
    n_preds = int(summary.get("n_preds") or 0)
    n_closed = int(summary.get("n_closed") or 0)
    SESSION_START = datetime(2026, 8, 1, 2, 55, 33, tzinfo=timezone.utc)
    cfg_path = OUT / "config.json"
    if cfg_path.exists():
        try:
            started = json.loads(cfg_path.read_text(encoding="utf-8")).get("started_at")
            if started:
                SESSION_START = datetime.fromisoformat(started.replace("Z", "+00:00"))
                if SESSION_START.tzinfo is None:
                    SESSION_START = SESSION_START.replace(tzinfo=timezone.utc)
        except Exception:
            pass
    elapsed_min = (now - SESSION_START).total_seconds() / 60

    # Live collector with slow-every=1 runs ~100-120s/snap, not 60s.
    # Expect ~0.5 snaps/min; require predictions only once we actually have warmup snaps.
    expected_snaps = max(1, int(elapsed_min * 0.45))
    if n_snaps < max(3, expected_snaps // 2):
        issues.append(f"snap lag: have={n_snaps} expected~{expected_snaps}")
        hard = True

    if n_snaps >= WARMUP_SNAPS and n_preds == 0:
        issues.append(f"post-warmup no predictions (snaps={n_snaps})")
        hard = True
    elif elapsed_min >= 70 and n_snaps < WARMUP_SNAPS:
        issues.append(f"still below warmup after 70m: snaps={n_snaps}")
        hard = True

    if now > DEADLINE:
        issues.append("past deadline")

    report = {
        "ok": not hard and not issues,
        "hard": hard,
        "issues": issues,
        "elapsed_min": round(elapsed_min, 1),
        "remaining_min": round((DEADLINE - now).total_seconds() / 60, 1),
        "n_snaps": n_snaps,
        "n_preds": n_preds,
        "n_closed": n_closed,
        "dir_acc": summary.get("dir_acc"),
        "mean_pnl_proxy": summary.get("mean_pnl_proxy"),
        "updated_at": summary.get("updated_at"),
        "checked_at": now.isoformat(),
    }
    print(json.dumps(report, indent=2))
    (OUT / "health_latest.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    if hard:
        return 2
    if issues:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
