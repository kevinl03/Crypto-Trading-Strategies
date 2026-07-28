"""
Thin wrapper for the Jul 22–28 validation upload.

Prefer the generic CLI for new datasets (always a NEW folder name):

    python -m experiments.upload_export_to_hf \\
        --export-dir data/exports/<export> \\
        --remote-folder <new_folder_name>

This wrapper targets validation_jul22-28/ and will refuse to clobber it
unless you add --overwrite.
"""
from __future__ import annotations

from experiments.upload_export_to_hf import main as upload_main

EXPORT = "data/exports/statarb_validation_20260722_20260728"
REMOTE_FOLDER = "validation_jul22-28"


def main() -> None:
    upload_main(
        [
            "--export-dir",
            EXPORT,
            "--remote-folder",
            REMOTE_FOLDER,
            "--commit-message",
            "Add validation_jul22-28/ long-run CEX signals "
            "(incl. long_short_ratio + liquidations)",
        ]
    )


if __name__ == "__main__":
    main()
