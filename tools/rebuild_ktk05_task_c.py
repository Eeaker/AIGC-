"""Rebuild the KTK_05_140 advanced Task-C deliverable."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.task_c_bonus import run


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--root", type=Path,
                        default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    anchor = (args.root / "annotations" / "ktk05_task_c" /
              "bonus_ktk05_c_A0001_regions.json")
    if not anchor.exists():
        # Compatibility with packages created before annotations were grouped.
        anchor = args.root / "annotations" / "bonus_ktk05_c_A0001_regions.json"
    if not anchor.exists():
        raise FileNotFoundError("KTK_05 A1/A3/A5 region annotations are missing")
    output = args.root / "outputs" / "supplementary" / "ktk05"
    written = run(args.data_root, output, anchor_annotation=anchor)
    print(f"Wrote {len(written)} Task-C files under {output}")


if __name__ == "__main__":
    main()
