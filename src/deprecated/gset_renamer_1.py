#!/usr/bin/env python3
import sys
import re
from pathlib import Path

# Rename:
#   Q_gset70.npy -> Q_gset_70.npy
#   V_gset7.npy  -> V_gset_7.npy
# Works for any *.npy in the target dir that matches [QV]_gset<number>.npy

if len(sys.argv) != 2:
    print("Usage: python add_underscore_gset.py <target_dir>")
    sys.exit(1)

d = Path(sys.argv[1]).expanduser().resolve()
if not d.is_dir():
    raise ValueError(f"Not a directory: {d}")

pat = re.compile(r"^(?P<prefix>[QV])_gset(?P<num>\d+)\.npy$")

for p in d.iterdir():
    if not p.is_file():
        continue
    m = pat.match(p.name)
    if not m:
        continue

    prefix = m.group("prefix")
    num = m.group("num")
    new_name = f"{prefix}_gset_{num}.npy"
    new_path = d / new_name

    if new_path.exists():
        raise FileExistsError(f"Target exists: {new_path}")

    print(f"{p.name} -> {new_name}")
    p.rename(new_path)
