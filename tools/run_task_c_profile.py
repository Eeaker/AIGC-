from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.task_c import run


def main() -> None:
    parser = argparse.ArgumentParser(description="Render one explicit Task-C information profile")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--profile",
        choices=("automatic", "assisted", "dense_assisted", "oracle", "reviewed_assisted"),
        required=True,
    )
    args = parser.parse_args()
    paths = run(args.data_root, args.output, seed_profile=args.profile)
    print(f"rendered {len(paths)} Task-C frames with profile={args.profile}")


if __name__ == "__main__":
    main()
