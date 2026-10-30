#!/usr/bin/env python3
"""Package the chrome-extension/ directory into distributable zips.

Produces in --out:
    fraud-detector-v<version>.zip        - Load-unpacked / GitHub Release package
    fraud-detector-edge-v<version>.zip   - Microsoft Edge Add-ons package
                                            (identical contents; Edge requires the
                                            manifest at the zip root, which both have)
    SHA256SUMS.txt                        - sha256 of every produced zip

Both zips contain the extension files at the zip ROOT (no wrapping folder),
which is what Chrome "Load unpacked" (after unzip) and the Edge store expect.

Excluded: VCS files, node_modules, tests, source maps, OS junk.

Usage:
    python scripts/package_extension.py --src chrome-extension --out dist --version 1.2.0

Stdlib only.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import zipfile
from pathlib import Path

EXCLUDE_DIRS = {".git", "node_modules", "tests", "__pycache__"}
EXCLUDE_SUFFIXES = {".map"}
EXCLUDE_NAMES = {".DS_Store", "Thumbs.db", ".gitignore"}


def iter_files(src: Path):
    for path in sorted(src.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(src)
        if any(part in EXCLUDE_DIRS for part in rel.parts):
            continue
        if path.name in EXCLUDE_NAMES or path.suffix in EXCLUDE_SUFFIXES:
            continue
        yield path, rel


def write_zip(src: Path, dest: Path) -> int:
    count = 0
    with zipfile.ZipFile(dest, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for path, rel in iter_files(src):
            zf.write(path, arcname=str(rel).replace("\\", "/"))
            count += 1
    return count


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=Path, required=True, help="Extension directory")
    parser.add_argument("--out", type=Path, required=True, help="Output directory")
    parser.add_argument("--version", required=True, help="Version string, e.g. 1.2.0")
    args = parser.parse_args()

    if not (args.src / "manifest.json").exists():
        print(f"ERROR: {args.src}/manifest.json not found - is --src the extension root?")
        return 1

    model = args.src / "model" / "model.onnx"
    if not model.exists():
        print("WARNING: model/model.onnx is missing - the package will not run the real model.")

    args.out.mkdir(parents=True, exist_ok=True)
    tag = f"v{args.version}"
    targets = [
        args.out / f"fraud-detector-{tag}.zip",
        args.out / f"fraud-detector-edge-{tag}.zip",
    ]

    lines = []
    for dest in targets:
        count = write_zip(args.src, dest)
        digest = sha256(dest)
        size_mb = dest.stat().st_size / (1024 * 1024)
        print(f"{dest.name}: {count} files, {size_mb:.1f} MB, sha256 {digest}")
        lines.append(f"{digest}  {dest.name}")

    sums = args.out / "SHA256SUMS.txt"
    sums.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {sums}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
