"""Validate this planning package without executing research experiments.

Run with /opt/venv/bin/python /workspace/improvement/validate_package.py.
Writes only artifact_validation.json beside this file.
"""
import ast
import csv
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
required = [
    'README.md', '01_original_method_and_baseline.md',
    '02_research_directions.md', '03_literature_review.md',
    '04_experimental_plan.md', 'experiment_matrix.csv',
    'data_and_evaluation_constraints.md', 'baseline_evidence.json',
    'audit_saved_baselines.py', 'literature_sources.json', 'request.txt',
]
errors = []
for name in required:
    p = ROOT / name
    if not p.is_file() or not p.stat().st_size:
        errors.append(f'Missing or empty artifact: {name}')
if errors:
    raise SystemExit('\n'.join(errors))

for p in ROOT.glob('*.py'):
    ast.parse(p.read_text(), filename=str(p))

baseline = json.loads((ROOT / 'baseline_evidence.json').read_text())
if baseline['status'] != 'pass' or baseline['issues']:
    errors.append('Saved-evidence audit did not pass')
if len(baseline['runs']) != 10 or sum(r['independent_edit_distance_rows'] for r in baseline['runs']) != 6570:
    errors.append('Baseline audit count differs from narrative')
if any(r['cohort_id_counts']['prospective48'] for r in baseline['score_file_exposure']):
    errors.append('Prospective48 appears in scanned score history')

sources = json.loads((ROOT / 'literature_sources.json').read_text())['sources']
if len({s['id'] for s in sources}) != len(sources):
    errors.append('Duplicate literature source IDs')
for s in sources:
    if not all(s.get(k) for k in ('id', 'title', 'primary_url', 'verified_on')):
        errors.append(f'Incomplete literature record: {s.get("id")}')
    if not s.get('primary_url', '').startswith('https://'):
        errors.append(f'Non-HTTPS primary source: {s.get("id")}')

with (ROOT / 'experiment_matrix.csv').open(newline='') as f:
    reader = csv.DictReader(f)
    columns = reader.fieldnames or []
    experiments = list(reader)
if not experiments:
    errors.append('No experiments in matrix')
required_columns = {
    'id', 'status', 'hypothesis', 'methodological_change', 'architecture',
    'learning_paradigm', 'data_requirements', 'preprocessing',
    'training_procedure', 'loss_or_objective', 'metrics',
    'compute_memory_estimate', 'runtime_estimate',
    'support_criterion', 'reject_or_stop_criterion',
}
if not required_columns.issubset(columns):
    errors.append('Matrix is missing required experiment-specification columns')
for n, row in enumerate(experiments, start=2):
    if None in row or any(v is None or not v.strip() for v in row.values()):
        errors.append(f'Missing/extra matrix field on CSV line {n}')
id_column = next((c for c in columns if c.lower() in ('id', 'experiment_id')), None)
if id_column:
    if len({r[id_column] for r in experiments}) != len(experiments):
        errors.append('Duplicate experiment IDs')
status_column = next((c for c in columns if c.lower() == 'status'), None)
if not status_column or any(r[status_column].lower() != 'proposed' for r in experiments):
    errors.append('Matrix must mark all experiments proposed')

checked_local_links = []
for p in ROOT.glob('*.md'):
    for target in re.findall(r'\[[^\]]*\]\(([^)]+)\)', p.read_text()):
        target = target.strip().strip('<>')
        if target.startswith(('http://', 'https://', 'mailto:', '#', 'codex:')):
            continue
        target = re.sub(r':\d+$', '', target.split('#', 1)[0])
        link_path = Path(target)
        if not link_path.is_absolute():
            link_path = p.parent / link_path
        # The validation receipt is the output of this script.
        if link_path == ROOT / 'artifact_validation.json':
            continue
        if not link_path.exists():
            errors.append(f'Broken local link in {p.name}: {target}')
        checked_local_links.append({'document': p.name, 'target': target})

# Check preserved source/config inputs already hashed by the baseline audit.
# This does not repeat the much larger audio rehash audit.
preserved_inputs = []
for name, expected in baseline['file_sha256'].items():
    p = Path(name)
    if p.suffix not in ('.py', '.md', '.yaml', '.json', '.pdf') or '/reproduction-runs/' in name:
        continue
    actual = hashlib.sha256(p.read_bytes()).hexdigest()
    if actual != expected:
        errors.append(f'Existing source/config changed after baseline audit: {name}')
    preserved_inputs.append(name)

manifest = {
    p.name: {'bytes': p.stat().st_size, 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}
    for p in sorted(ROOT.iterdir())
    if p.is_file() and p.name != 'artifact_validation.json'
}
report = {
    'checked_at_utc': datetime.now(timezone.utc).isoformat(),
    'status': 'pass' if not errors else 'fail',
    'scope': 'Syntax, JSON/CSV structure, nonempty experiment fields, proposed statuses, local links, and preserved input hashes; no models executed or external links re-fetched.',
    'errors': errors,
    'experiment_count': len(experiments),
    'matrix_columns': columns,
    'literature_source_count': len(sources),
    'baseline_runs': len(baseline['runs']),
    'baseline_score_rows': sum(r['independent_edit_distance_rows'] for r in baseline['runs']),
    'local_links_checked': len(checked_local_links),
    'preserved_source_inputs_checked': preserved_inputs,
    'artifacts': manifest,
}
(ROOT / 'artifact_validation.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps({k: report[k] for k in ('status', 'errors', 'experiment_count', 'literature_source_count', 'baseline_score_rows', 'local_links_checked')}, indent=2))
raise SystemExit(bool(errors))
