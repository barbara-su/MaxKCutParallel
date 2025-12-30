#!/usr/bin/env python3
"""
For every (nested) subdirectory that contains at least one .json file,
compute the average of `best_score` over the JSON files in that directory,
and print the FULL directory path.

Usage:
  python avg_best_score_tree.py /path/to/root
"""

import sys
import json
from pathlib import Path

def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python avg_best_score_tree.py <root_dir>")
        sys.exit(1)

    root = Path(sys.argv[1]).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"Not a directory: {root}")

    dir_scores: dict[Path, list[float]] = {}

    for json_path in root.rglob("*.json"):
        parent = json_path.parent
        try:
            with json_path.open("r") as f:
                data = json.load(f)
            if "best_score" in data:
                dir_scores.setdefault(parent, []).append(float(data["best_score"]))
        except Exception as e:
            print(f"Skipping {json_path}: {e}")

    if not dir_scores:
        print("No JSON files with best_score found.")
        return

    for d in sorted(dir_scores.keys()):
        scores = dir_scores[d]
        avg = sum(scores) / len(scores)
        print(f"{str(d)}\tfiles={len(scores)}\tavg_best_score={avg}")

if __name__ == "__main__":
    main()
