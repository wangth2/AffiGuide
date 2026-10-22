#!/usr/bin/env python3
"""Fetch or explicitly install exact, SHA256-pinned AffiGuide model assets."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from diffab.utils.model_assets import ModelAssetError, ensure_model, load_manifest, verify_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models', nargs='+', choices=('DiffAb', 'D', 'G'), default=('DiffAb', 'D', 'G'))
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1], help='Install/check repository-relative manifest paths under this directory.')
    parser.add_argument('--offline', action='store_true', help='Forbid all network calls; existing matching files or explicit --source-dir only.')
    parser.add_argument('--source-dir', type=Path, help='Explicit local frozen asset directory; no hidden cache is searched.')
    parser.add_argument('--check-only', action='store_true', help='Verify files without downloads or writes.')
    parser.add_argument('--manifest', type=Path, default=Path(__file__).resolve().parents[1] / 'MODEL_MANIFEST.json')
    args = parser.parse_args()
    rows = []
    for name in args.models:
        try:
            if not args.check_only:
                ensure_model(name, root=args.root, offline=args.offline, manifest_path=args.manifest, source_dir=args.source_dir)
            row = verify_model(name, root=args.root, manifest_path=args.manifest)
        except (ModelAssetError, OSError, ValueError) as exc:
            row = {'model': name, 'status': 'ERROR', 'error': str(exc)}
        rows.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
    return int(any(row['status'] != 'VERIFIED' for row in rows))


if __name__ == '__main__':
    sys.exit(main())
