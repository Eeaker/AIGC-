"""Reproduce the runnable parts of the submission and validate all formal outputs."""
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

from src import task_a, task_b, task_c


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--regenerate-b", action="store_true")
    ap.add_argument("--regenerate-b-shot-adapted", action="store_true")
    ap.add_argument("--regenerate-c", action="store_true")
    ap.add_argument("--regenerate-a-fast", action="store_true")
    ap.add_argument("--regenerate-a-from-checkpoints", action="store_true")
    args = ap.parse_args()
    root = Path(__file__).resolve().parent

    if args.regenerate_a_fast:
        # The packaged formal A is the documented DSV+LOFO result.  This flag
        # intentionally writes a separately named, auditable CPU baseline.
        task_a.run(args.data_root, root / "outputs" / "algorithm_only" / "task_a_fast_baseline",
                   preserve_green=False)
    if args.regenerate_a_from_checkpoints:
        subprocess.run([
            sys.executable, str(root / "tools" / "rebuild_task_a_from_checkpoints.py"),
            "--root", str(root), "--data-root", str(args.data_root),
            "--output", str(root / "outputs" / "reconstructed" / "task_a"),
        ], check=True)
    if args.regenerate_b:
        task_b.run(args.data_root, root / "outputs" / "official" / "task_b",
                   use_shot_cleanup=False)
    if args.regenerate_b_shot_adapted:
        task_b.run(args.data_root, root / "outputs" / "diagnostic" / "task_b_shot_adapted",
                   use_shot_cleanup=True)
    if args.regenerate_c:
        task_c.run(args.data_root, root / "outputs" / "official" / "task_c_assisted",
                   seed_profile="assisted")
        task_c.run(args.data_root, root / "outputs" / "algorithm_only" / "task_c_automatic",
                   seed_profile="automatic")

    subprocess.run([
        sys.executable, str(root / "tools" / "validate_submission.py"),
        "--data-root", str(args.data_root), "--root", str(root),
    ], check=True)


if __name__ == "__main__":
    main()
