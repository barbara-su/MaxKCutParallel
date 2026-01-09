#!/usr/bin/env python3
"""
Given a directory, recursively find all .json files and print:

  <full_json_path> <tab> maxcut=<value>

- Prints one line per JSON file that contains a "maxcut" key.
- Skips unreadable / invalid JSON files with a warning to stderr.

Usage:
  python list_maxcut_jsons.py /path/to/root
"""

import sys
import json
from pathlib import Path


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python list_maxcut_jsons.py <root_dir>", file=sys.stderr)
        sys.exit(1)

    root = Path(sys.argv[1]).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"Not a directory: {root}")

    found_any = False

    for json_path in sorted(root.rglob("*.json")):
        try:
            with json_path.open("r") as f:
                data = json.load(f)
        except Exception as e:
            print(f"Skipping {json_path}: {e}", file=sys.stderr)
            continue

        if isinstance(data, dict) and "maxcut" in data:
            try:
                maxcut_val = float(data["maxcut"])
            except Exception as e:
                print(f"Skipping {json_path}: maxcut not numeric ({e})", file=sys.stderr)
                continue

            print(f"{str(json_path)}\tmaxcut={maxcut_val}")
            found_any = True
            
        elif isinstance(data, dict) and "best_score" in data:
            try:
                maxcut_val = float(data["best_score"])
            except Exception as e:
                print(f"Skipping {json_path}: best_score not numeric ({e})", file=sys.stderr)
                continue

            print(f"{str(json_path)}\tbest_score={maxcut_val}")
            found_any = True

    if not found_any:
        print("No JSON files with maxcut found.", file=sys.stderr)


if __name__ == "__main__":
    main()
