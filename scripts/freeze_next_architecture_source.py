#!/usr/bin/env python3
"""Freeze a reviewable code-only candidate; never deploy or start a service."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

from v44_preflight import source_inventory


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', type=Path, default=Path('/home/dzliu/pzliu/sap-rl'))
    parser.add_argument('--output-directory', type=Path, required=True)
    args = parser.parse_args()
    args.output_directory = args.output_directory.resolve()
    args.output_directory.mkdir(parents=True, exist_ok=False)
    source = args.output_directory/'source'; source.mkdir()
    for directory in ('sapai/sapai', 'sapai-gym/sapai_gym1', 'sapai-gym/scripts', 'sapai-gym/tests'):
        for path in sorted((args.project/directory).rglob('*.py')):
            if '__pycache__' in path.parts:
                continue
            target = source/path.relative_to(args.project)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
    for name in ('sapai/setup.py', 'sapai-gym/setup.py'):
        path = args.project/name
        if path.exists():
            shutil.copy2(path, source/name)
    # guide_teacher imports this file for its provenance/version even when
    # no guide policy is being trained or evaluated.
    asset_names = ('sapai-gym/TURTLE_P1_GUIDE.md',)
    assets = {}
    for name in asset_names:
        path = args.project/name
        shutil.copy2(path, source/name)
        assets[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    inventory = source_inventory(source)
    (args.output_directory/'source-manifest.json').write_text(json.dumps(inventory, indent=2)+'\n')
    (args.output_directory/'assets-manifest.json').write_text(json.dumps(assets, indent=2)+'\n')
    bundle_hash = hashlib.sha256(json.dumps({'code': inventory['sha256'], 'assets': assets}, sort_keys=True).encode()).hexdigest()
    (args.output_directory/'status.json').write_text(json.dumps({
        'purpose': 'next-architecture review candidate; not the active entropy runtime',
        'source': str(source), 'sourceInventorySHA256': inventory['sha256'], 'files': len(inventory['files']),
        'sourceBundleSHA256': bundle_hash, 'assetsManifest': 'assets-manifest.json',
        'image': 'sha256:3ed8782e02a5d46e400bc39a5a77654c26e8a86b1d210c160e5965b765955ce1',
        'deployed': False, 'queued': False, 'gpuValidated': False, 'qualityValidated': False,
        'strengthValidated': False, 'requiresEntropyResultsAndExplicitResourceBoundary': True,
        'policyCausalContextTransportComplete': False}, indent=2)+'\n')
    print(json.dumps({'source': str(source), 'sha256': inventory['sha256'], 'files': len(inventory['files']), 'deployed': False, 'queued': False}))


if __name__ == '__main__':
    main()
