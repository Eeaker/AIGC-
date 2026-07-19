"""Evaluate Task-C information boundaries without retaining duplicate TGAs."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.io_utils import imread
from src.metrics import region_color_metrics
from src import task_c


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--output", type=Path,
                    default=Path("outputs/c_profile_metrics.json"))
    args = ap.parse_args()
    shot = args.data_root / "KTK_04_246B"
    # All profiles use the same nine line frames. Cache dense flows by content
    # so the information-boundary audit measures labels, not repeated DIS work.
    uncached_flow = task_c._flow
    flow_cache = {}

    def cached_flow(source, target):
        key = (hashlib.blake2b(source, digest_size=8).digest(),
               hashlib.blake2b(target, digest_size=8).digest())
        if key not in flow_cache:
            flow_cache[key] = uncached_flow(source, target)
        return flow_cache[key]

    task_c._flow = cached_flow
    rows: dict[str, dict[str, object]] = {}
    for profile in ("automatic", "assisted", "dense_assisted", "oracle", "reviewed_assisted"):
        with tempfile.TemporaryDirectory(prefix=f"lingtu_c_{profile}_") as temp:
            paths = task_c.run(args.data_root, Path(temp), seed_profile=profile)
            frame_metrics = {}
            for path in paths:
                line = imread(shot / "源文件" / "上色" / path.name)
                ref = imread(shot / "成品" / "上色" / path.name)
                frame_metrics[path.name] = region_color_metrics(line, imread(path), ref)
            keys = ("area_precision", "area_coverage", "region_precision",
                    "region_coverage", "white_target_exact_fraction")
            mean = {
                key: sum(float(row[key]) for row in frame_metrics.values()) / len(frame_metrics)
                for key in keys
            }
            seed_count = len(json.loads((Path(temp) / "semantic_seeds.json").read_text("utf-8")))
            rows[profile] = {"seed_count": seed_count, "mean": mean,
                             "frames": frame_metrics}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({name: {"seed_count": row["seed_count"], "mean": row["mean"]}
                      for name, row in rows.items()}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
