#!/usr/bin/env python3
"""Validate a Chrome/Edge extension manifest.json.

Usage:
    python scripts/validate_manifest.py chrome-extension/manifest.json
    python scripts/validate_manifest.py chrome-extension/manifest.json --expect-version 1.2.0
    python scripts/validate_manifest.py chrome-extension/manifest.json --edge

Checks (errors fail the build):
  - file is valid JSON, manifest_version == 3
  - name / version / description present; version is numeric dotted (x.y.z)
  - version matches --expect-version when given
  - every file the manifest references (service worker, content scripts,
    popup, options page, icons, web-accessible resources) exists on disk

Warnings (printed, non-fatal):
  - broad host_permissions, missing icons, and - with --edge - anything
    known to trip up Microsoft Edge Add-ons review (e.g. a Chrome-only
    `key` field, which Edge ignores and reviewers flag).

Stdlib only, so it runs in CI with no installs.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

VERSION_RE = re.compile(r"^\d+(\.\d+){0,3}$")


def collect_referenced_files(manifest: dict) -> list[str]:
    refs: list[str] = []

    background = manifest.get("background", {})
    if isinstance(background, dict) and background.get("service_worker"):
        refs.append(background["service_worker"])

    action = manifest.get("action", {})
    if isinstance(action, dict) and action.get("default_popup"):
        refs.append(action["default_popup"])

    for script in manifest.get("content_scripts", []) or []:
        for key in ("js", "css"):
            refs.extend(script.get(key, []) or [])

    if manifest.get("options_page"):
        refs.append(manifest["options_page"])
    options_ui = manifest.get("options_ui", {})
    if isinstance(options_ui, dict) and options_ui.get("page"):
        refs.append(options_ui["page"])

    icons = manifest.get("icons", {})
    if isinstance(icons, dict):
        refs.extend(icons.values())
    if isinstance(action, dict) and isinstance(action.get("default_icon"), dict):
        refs.extend(action["default_icon"].values())
    elif isinstance(action, dict) and action.get("default_icon"):
        refs.append(action["default_icon"])

    for war in manifest.get("web_accessible_resources", []) or []:
        if isinstance(war, dict):
            refs.extend(war.get("resources", []) or [])
        elif isinstance(war, str):
            refs.append(war)

    side_panel = manifest.get("side_panel", {})
    if isinstance(side_panel, dict) and side_panel.get("default_path"):
        refs.append(side_panel["default_path"])

    # Wildcard entries (e.g. "model/*.wasm") cannot be checked literally.
    return [r for r in refs if "*" not in r]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path, help="Path to manifest.json")
    parser.add_argument("--expect-version", default=None,
                        help="Fail unless manifest version equals this")
    parser.add_argument("--edge", action="store_true",
                        help="Add Microsoft Edge Add-ons specific checks")
    args = parser.parse_args()

    errors: list[str] = []
    warnings: list[str] = []

    if not args.manifest.exists():
        print(f"ERROR: manifest not found: {args.manifest}")
        return 1

    try:
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        print(f"ERROR: manifest is not valid JSON: {exc}")
        return 1

    root = args.manifest.parent

    if manifest.get("manifest_version") != 3:
        errors.append(f"manifest_version must be 3, got {manifest.get('manifest_version')!r}")

    for field in ("name", "version", "description"):
        if not manifest.get(field):
            errors.append(f"missing required field: {field}")

    version = str(manifest.get("version", ""))
    if version and not VERSION_RE.match(version):
        errors.append(f"version {version!r} is not a numeric dotted version (x.y.z)")
    if args.expect_version and version != args.expect_version:
        errors.append(
            f"version mismatch: manifest has {version}, expected {args.expect_version}"
        )

    for ref in collect_referenced_files(manifest):
        if not (root / ref).exists():
            errors.append(f"referenced file missing on disk: {ref}")

    host_permissions = manifest.get("host_permissions", []) or []
    if any(h in ("<all_urls>", "*://*/*") for h in host_permissions):
        warnings.append(
            "host_permissions includes a broad pattern - store reviewers will ask why; "
            "keep it only if the floating button truly needs every page."
        )
    if not manifest.get("icons"):
        warnings.append("no icons declared - stores display a generic puzzle-piece icon.")

    if args.edge:
        if "key" in manifest:
            warnings.append(
                "manifest contains a Chrome-only `key` field - Edge ignores it and "
                "reviewers may flag it; remove it from the Edge package if present."
            )
        if "update_url" in manifest:
            warnings.append(
                "manifest contains `update_url` - remove it for store submission; "
                "the store manages updates."
            )

    for warning in warnings:
        print(f"WARNING: {warning}")
    for error in errors:
        print(f"ERROR: {error}")

    if errors:
        print(f"Manifest validation FAILED ({len(errors)} error(s)).")
        return 1
    print(f"Manifest OK: {manifest.get('name')} v{version} (MV3)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
