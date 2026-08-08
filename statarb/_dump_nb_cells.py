"""Dump cex_gbm_new.ipynb cell sources for LOGO script extraction."""
from __future__ import annotations

import json
from pathlib import Path

nb_path = Path(__file__).with_name("cex_gbm_new.ipynb")
out_dir = Path(__file__).with_name("_nb_cell_dump")
out_dir.mkdir(exist_ok=True)

nb = json.loads(nb_path.read_text(encoding="utf-8"))
index_lines = []
for i, cell in enumerate(nb["cells"]):
    src = "".join(cell.get("source", []))
    first = src.strip().split("\n")[0][:100] if src.strip() else "(empty)"
    index_lines.append(f"{i:3d} {cell['cell_type'][:4]} {len(src):6d} | {first}")
    (out_dir / f"cell_{i:03d}.py").write_text(src, encoding="utf-8")

(out_dir / "INDEX.txt").write_text("\n".join(index_lines), encoding="utf-8")
print(f"Wrote {len(nb['cells'])} cells to {out_dir}")
print("\n".join(index_lines))
