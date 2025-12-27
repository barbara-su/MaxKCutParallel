#!/usr/bin/env python3
import sys
import re
from pathlib import Path

if len(sys.argv) != 2:
    print("Usage: python rename_gset.py <target_dir>")
    sys.exit(1)

target_dir = Path(sys.argv[1]).expanduser().resolve()
if not target_dir.is_dir():
    raise ValueError(f"Not a directory: {target_dir}")

pattern = re.compile(r'^(?P<prefix>[QV])_gset(?P<gid>\d+)_.*\.npy$')

for p in target_dir.iterdir():
    if not p.is_file():
        continue

    m = pattern.match(p.name)
    if not m:
        continue

    prefix = m.group("prefix")   # Q or V
    gid = m.group("gid")          # gset id
    new_name = f"{prefix}_gset_{gid}.npy"
    new_path = target_dir / new_name

    print(f"{p.name} -> {new_name}")
    p.rename(new_path)
