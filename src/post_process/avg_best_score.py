#!/usr/bin/env python3
import sys
import json
from pathlib import Path
from json import JSONDecodeError

def load_single_json_allow_trailing(path: Path) -> dict:
    """
    Load a JSON object. If the file has extra trailing data (common when two JSON
    objects are concatenated), parse the first JSON object and ignore the rest.
    """
    text = path.read_text()
    try:
        return json.loads(text)
    except JSONDecodeError as e:
        # Try to salvage "Extra data" by decoding the first object only.
        dec = json.JSONDecoder()
        obj, end = dec.raw_decode(text)
        # If you want to be strict, you can check that remaining non-whitespace is empty.
        # remaining = text[end:].strip()
        # if remaining:
        #     raise
        if not isinstance(obj, dict):
            raise
        return obj

def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python avg_best_score_tree.py <root_dir>")
        sys.exit(1)

    root = Path(sys.argv[1]).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"Not a directory: {root}")

    # Per directory: sum of scores, count of files contributing
    dir_sum: dict[Path, float] = {}
    dir_nfiles: dict[Path, int] = {}

    total_json = 0
    parsed_ok = 0
    skipped = 0
    missing_key = 0

    for json_path in root.rglob("*.json"):
        total_json += 1
        parent = json_path.parent
        try:
            data = load_single_json_allow_trailing(json_path)
            parsed_ok += 1

            if "best_score" in data:
                score = float(data["best_score"])
            elif "maxcut" in data:
                score = float(data["maxcut"])
            else:
                missing_key += 1
                continue

            dir_sum[parent] = dir_sum.get(parent, 0.0) + score
            dir_nfiles[parent] = dir_nfiles.get(parent, 0) + 1

        except Exception as e:
            skipped += 1
            print(f"Skipping {json_path}: {e}")

    print(f"total_json={total_json} parsed_ok={parsed_ok} skipped={skipped} missing_key={missing_key}")

    if not dir_nfiles:
        print("No JSON files with best_score/maxcut found.")
        return

    for d in sorted(dir_nfiles):
        n = dir_nfiles[d]
        avg = dir_sum[d] / n
        print(f"{d}\tfiles={n}\tavg_score={avg}")

if __name__ == "__main__":
    main()
