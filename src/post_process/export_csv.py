#!/usr/bin/env python3
"""
Export a CSV of (json filename, score) for all .json files under a root directory,
sorted by filename in *natural* order (so Q_gset_2 < Q_gset_10).

Score extraction:
  - prefers "maxcut"
  - falls back to "best_score"

Usage:
  python export_scores_csv_natsort.py /path/to/root -o scores.csv
"""

import sys
import json
import csv
import re
import argparse
from pathlib import Path
from typing import Optional, Tuple, List, Union


_NAT_SPLIT = re.compile(r"(\d+)")


def natural_key(s: str) -> List[Union[int, str]]:
    """
    Split string into chunks of digits/non-digits so digits sort numerically.
    Example: "Q_gset_10.json" -> ["q_gset_", 10, ".json"]
    """
    parts = _NAT_SPLIT.split(s)
    key: List[Union[int, str]] = []
    for p in parts:
        if not p:
            continue
        if p.isdigit():
            key.append(int(p))
        else:
            key.append(p.lower())
    return key


def extract_score(data: object) -> Optional[float]:
    if not isinstance(data, dict):
        return None
    if "maxcut" in data:
        try:
            return float(data["maxcut"])
        except Exception:
            return None
    if "best_score" in data:
        try:
            return float(data["best_score"])
        except Exception:
            return None
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root_dir", help="Root directory to search recursively for .json files")
    parser.add_argument("-o", "--out", default="scores.csv", help="Output CSV path")
    parser.add_argument(
        "--abs-paths",
        action="store_true",
        help="Write absolute paths in CSV instead of paths relative to root",
    )
    args = parser.parse_args()

    root = Path(args.root_dir).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"Not a directory: {root}")

    rows: List[Tuple[str, float]] = []

    for json_path in root.rglob("*.json"):
        try:
            with json_path.open("r") as f:
                data = json.load(f)
        except Exception as e:
            print(f"Skipping {json_path}: {e}", file=sys.stderr)
            continue

        score = extract_score(data)
        if score is None:
            continue

        name = str(json_path if args.abs_paths else json_path.relative_to(root))
        rows.append((name, score))

    if not rows:
        print("No JSON files with a usable score (maxcut/best_score) found.", file=sys.stderr)
        sys.exit(2)

    # Natural sort by filename (and then full path if needed)
    rows.sort(key=lambda t: natural_key(t[0]))

    out_path = Path(args.out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file", "score"])
        w.writerows(rows)

    print(f"Wrote {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
