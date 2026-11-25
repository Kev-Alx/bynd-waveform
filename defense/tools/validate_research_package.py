"""Validate the research deliverable, without model inference or network requests."""
from __future__ import annotations

import ast
import hashlib
import json
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import unquote, urlparse

ROOT = Path('/workspace/defense')
REPORT = ROOT / 'evidence/package_validation.json'
MANIFEST = ROOT / 'evidence/artifact_manifest.json'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    errors = []
    required = ['HANDOFF.md', '00_RESEARCH_BRIEF.md', '01_ORIGINAL_METHOD.md',
                '02_REPOSITORY_AUDIT.md', '03_LITERATURE_REVIEW.md',
                '04_DEFENSE_DESIGNS.md', '05_EXPERIMENTAL_PROTOCOL.md',
                '06_PHASE_PLAN.md', 'literature_sources.json',
                'evidence/local_audit.json', 'tools/audit_local_evidence.py']
    for relative in required:
        if not (ROOT / relative).is_file():
            errors.append(f'Missing required file: {relative}')
    files = sorted(p for p in ROOT.rglob('*') if p.is_file())
    for path in files:
        if not path.resolve().is_relative_to(ROOT.resolve()):
            errors.append(f'Artifact resolves outside root: {path}')
        if '__pycache__' in path.parts:
            errors.append(f'Unexpected bytecode cache: {path}')
        if path.suffix == '.py':
            ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
    json_count = 0
    for path in files:
        if path.suffix == '.json':
            json.loads(path.read_text(encoding='utf-8'))
            json_count += 1
    checked_links = 0
    for path in ROOT.glob('*.md'):
        for target in re.findall(r'\[[^\]]*\]\(([^)]+)\)', path.read_text(encoding='utf-8')):
            target = target.strip('<>')
            if urlparse(target).scheme or target.startswith('#'):
                continue
            target = unquote(target.split('#', 1)[0])
            target = re.sub(r':\d+$', '', target)
            dest = Path(target) if target.startswith('/') else path.parent / target
            if dest not in (REPORT, MANIFEST) and not dest.exists():
                errors.append(f'Broken local link in {path.name}: {target}')
            checked_links += 1
    registry = json.loads((ROOT / 'literature_sources.json').read_text())
    ids = [s['id'] for s in registry['sources']]
    if len(ids) != len(set(ids)):
        errors.append('Duplicate source IDs')
    for source in registry['sources']:
        parsed = urlparse(source['url'])
        if parsed.scheme != 'https' or not parsed.netloc or not source.get('checked'):
            errors.append(f'Incomplete primary-source entry: {source["id"]}')
    audit = json.loads((ROOT / 'evidence/local_audit.json').read_text())
    unchanged = 0
    for raw, item in audit['input_files'].items():
        path = Path(raw)
        if not path.is_file() or digest(path) != item['sha256']:
            errors.append(f'Audited input changed since snapshot: {raw}')
        else:
            unchanged += 1
    artifact_rows = [{'path': str(p.relative_to(ROOT)), 'bytes': p.stat().st_size,
                      'sha256': digest(p)} for p in files if p not in (REPORT, MANIFEST)]
    MANIFEST.write_text(json.dumps({'root': str(ROOT), 'files': artifact_rows,
                                   'excludes': ['evidence/artifact_manifest.json',
                                                'evidence/package_validation.json']}, indent=2) + '\n')
    result = {
        'created_at_taipei': datetime.now(timezone(timedelta(hours=8))).isoformat(),
        'status': 'pass' if not errors else 'fail', 'errors': errors,
        'artifact_files_hashed': len(artifact_rows), 'json_files_parsed': json_count,
        'local_links_checked': checked_links, 'primary_source_entries': len(ids),
        'audited_inputs_unchanged': unchanged,
        'scope': 'Containment, required artifacts, Python syntax, JSON, local links, provenance and unchanged input hashes.',
        'not_performed': ['Neural inference', 'Defense implementation/training',
                          'Model test suite', 'Fresh audio/lexical audit',
                          'New online verification of every cited URL'],
    }
    REPORT.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
