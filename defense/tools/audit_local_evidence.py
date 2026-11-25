"""Read-only research inventory; writes only beneath /workspace/defense.

No neural inference, attack generation, training, downloads, or lexical rescoring.
Reserved cohorts are inspected only for metadata/file availability, never predictions.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path('/workspace')
OUT = ROOT / 'defense'


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def command(args):
    result = subprocess.run(args, cwd=ROOT, capture_output=True, text=True)
    return {'returncode': result.returncode, 'stdout': result.stdout.strip(),
            'stderr': result.stderr.strip()}


def main():
    OUT.mkdir(exist_ok=True)
    evidence = OUT / 'evidence'
    evidence.mkdir(exist_ok=True)
    source_names = [
        'BeyondWaveform.pdf', 'paper/BeyondWaveform.md', 'AGENTS.md',
        'docs/agents/domain.md', 'docs/agents/issue-tracker.md',
        'BeyondWaveform_Reproduction_Spec.md', 'docs/STRICT_PARITY_AUDIT.md',
        'docs/DEFENSE_RESEARCH.md', 'docs/THREAD_SUMMARY_AND_PLAN.md',
        'beyondwaveform/attacks.py', 'beyondwaveform/losses.py',
        'beyondwaveform/defenses.py', 'beyondwaveform/models/feature.py',
        'beyondwaveform/models/whisper.py', 'beyondwaveform/models/targets.py',
        'beyondwaveform/config.py', 'beyondwaveform/training.py',
        'beyondwaveform/evaluation.py', 'configs/english_selected.yaml',
        'scripts/run_defense_suite.py',
        'reproduction-runs/english-defenses-999/results.json',
        'reproduction-runs/english-defenses-999/plan.json',
        'reproduction-runs/paper-quality-research/paper-setting-test-999/evaluation/summary.json',
        'improvement/data_and_evaluation_constraints.md',
        'improvement/08_pilot48_results.md', 'improvement/09_refinement_protocol.md',
        'improvement/10_refinement_results.md', 'improvement/baseline_evidence.json',
    ]
    sources = {str(ROOT / name): {'sha256': sha256(ROOT / name),
                                  'bytes': (ROOT / name).stat().st_size}
               for name in source_names if (ROOT / name).is_file()}
    legacy = json.loads((ROOT / 'improvement/baseline_evidence.json').read_text())
    cohort_paths = {
        name: value['path'] for name, value in legacy.get('manifests', {}).items()
        if isinstance(value, dict) and value.get('path')
    }
    cohort_paths.update({
        'attack_train_1h': str(ROOT / 'improvement/execution/data/train-1h.jsonl'),
        'attack_train_10h': str(ROOT / 'improvement/execution/data/train-10h.jsonl'),
        'other_project_final_reserve200': str(ROOT / 'improvement/execution/data/final-reserve-200.jsonl'),
        'exposed_attack_development48': str(ROOT / 'improvement/execution48/development-48.jsonl'),
        'exposed_attack_validation76': str(ROOT / 'improvement/execution_refine/validation-76.jsonl'),
    })
    cohorts, rowsets = {}, {}
    for name, raw_path in cohort_paths.items():
        path = Path(raw_path)
        if not path.is_file():
            cohorts[name] = {'path': raw_path, 'exists': False}
            continue
        rows = [json.loads(line) for line in path.open(encoding='utf-8') if line.strip()]
        rowsets[name] = rows
        durations = [r['duration'] for r in rows if 'duration' in r]
        cohorts[name] = {
            'path': raw_path, 'exists': True, 'sha256': sha256(path),
            'count': len(rows), 'speakers': len({str(r.get('speaker')) for r in rows}),
            'roles': sorted({r.get('role', 'unknown') for r in rows}),
            'splits': sorted({r.get('split', 'unknown') for r in rows}),
            'duration_seconds_sum': sum(durations),
            'source_audio_files_present': sum(Path(r['audio']).is_file() for r in rows),
            'audit_scope': 'manifest metadata and file existence; source audio not rehashed',
        }
        sources[str(path)] = {'sha256': sha256(path), 'bytes': path.stat().st_size}
    overlaps = []
    names = list(rowsets)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            rr, ss = rowsets[a], rowsets[b]
            ids = len({r['id'] for r in rr} & {r['id'] for r in ss})
            audio = len({r['audio_sha256'] for r in rr if r.get('audio_sha256')}
                        & {r['audio_sha256'] for r in ss if r.get('audio_sha256')})
            speakers = len({str(r['speaker']) for r in rr if r.get('speaker')}
                           & {str(r['speaker']) for r in ss if r.get('speaker')})
            overlaps.append({'a': a, 'b': b, 'id_overlap': ids,
                             'recorded_audio_sha256_overlap': audio,
                             'speaker_overlap': speakers})
    defense = json.loads((ROOT / 'reproduction-runs/english-defenses-999/results.json').read_text())
    results = []
    for row in defense['rows']:
        variants = {}
        for variant, v in row['summary'].items():
            recomputed = 100 * sum(v[k] for k in ('substitutions', 'deletions', 'insertions')) / v['reference_units']
            assert abs(recomputed - v['error_percent']) < 1e-10
            variants[variant] = {**v, 'recomputed_from_saved_counts': recomputed}
        results.append({'condition': row['name'], 'variants': variants})
    paper_control = json.loads((ROOT / 'reproduction-runs/paper-quality-research/paper-setting-test-999/evaluation/summary.json').read_text())
    for v in paper_control['summary'].values():
        assert abs(100 * sum(v[k] for k in ('substitutions', 'deletions', 'insertions'))
                   / v['reference_units'] - v['error_percent']) < 1e-10
    status_path = ROOT / 'improvement/execution_refine/progress.json'
    status = json.loads(status_path.read_text())
    pid = status.get('supervisor_pid')
    report = {
        'created_at_taipei': datetime.now(timezone(timedelta(hours=8))).isoformat(),
        'artifact_root': str(OUT),
        'scope': 'Research-only saved-artifact inspection; no new model results.',
        'git_head': command(['git', 'rev-parse', 'HEAD']),
        'git_status': command(['git', 'status', '--short', '--branch']),
        'gpu_snapshot': command(['nvidia-smi', '--query-gpu=name,memory.total,memory.used,utilization.gpu', '--format=csv,noheader']),
        'input_files': sources,
        'cohorts': cohorts, 'cohort_overlaps': overlaps,
        'saved_defense_results': results,
        'saved_paper_setting_test': paper_control['summary'],
        'existing_attack_job_observation': {
            'source': str(status_path), 'status': status.get('status'),
            'phase': status.get('phase'), 'supervisor_pid': pid,
            'pid_exists_at_snapshot': bool(pid and Path(f'/proc/{pid}').is_dir()),
            'note': 'Point-in-time observation only; job not modified or adopted.',
        },
        'limits': [
            'No fresh ASR/MOS inference, audio rehash, lexical alignment audit, or training.',
            'Cohort inventory covers explicitly listed known manifests, not every historical cache.',
            'Overlap hashes are recorded manifest hashes, not newly decoded PCM hashes.',
            'Existing reserves belong to prior work and are not allocated to this defense task.',
            'Metadata availability and independence from defense training do not imply pretrained-model non-exposure.',
        ],
    }
    dest = evidence / 'local_audit.json'
    dest.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(json.dumps({'output': str(dest), 'input_files_hashed': len(sources),
                      'cohorts': {n: {'count': c.get('count'), 'speakers': c.get('speakers'),
                                     'source_audio_files_present': c.get('source_audio_files_present')}
                                  for n, c in cohorts.items()},
                      'existing_job': report['existing_attack_job_observation']}, indent=2))


if __name__ == '__main__':
    main()
