"""Independently audit a completed corrected-CER stage from saved transcripts."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import unicodedata

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reproduction-runs/mandarin-cer-corrected'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalized(text):
    text = unicodedata.normalize('NFKC', text).lower()
    return ''.join(c for c in text if not c.isspace() and not unicodedata.category(c).startswith('P'))


def distance(a, b):
    previous = list(range(len(b)+1))
    for i, x in enumerate(a, 1):
        current = [i]
        for j, y in enumerate(b, 1):
            current.append(min(current[-1]+1, previous[j]+1, previous[j-1]+(x!=y)))
        previous = current
    return previous[-1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['pilot-clean', 'pilot', 'test-clean', 'test'], required=True)
    parser.add_argument('--candidate', choices=['selected', 'paper', 'paper-lr025'], default='selected')
    args = parser.parse_args()
    folder = OUT / args.stage / args.candidate
    result = json.loads((folder/'results.json').read_text())
    assert result['status'] == 'complete'
    plan = json.loads((OUT/'plan.json').read_text())
    assert sha(OUT/'plan.json') == result['plan_sha256']
    if args.candidate not in plan['config_sha256']:
        extension = json.loads((OUT/'additional-candidates.json').read_text())
        assert extension['base_plan_sha256'] == sha(OUT/'plan.json')
        assert result['candidate_extension_sha256'] == sha(OUT/'additional-candidates.json')
        plan['config_sha256'][args.candidate] = extension['candidates'][args.candidate]['config_sha256']
    assert result['config_sha256'] == plan['config_sha256'][args.candidate]
    assert result['target']['sha256'] == plan['checkpoint_sha256']
    assert result['target']['decoding']['without_timestamps'] is True
    assert result['target']['language'] == 'zh'
    manifest = OUT / ('test-5000.jsonl' if args.stage.startswith('test') else 'pilot-48.jsonl')
    assert sha(manifest) == plan['test_manifest_sha256' if args.stage.startswith('test') else 'pilot_manifest_sha256']
    expected = {r['id']:r for r in map(json.loads, manifest.read_text().splitlines())}
    rows = list(map(json.loads, (folder/'utterances.jsonl').read_text().splitlines()))
    variants = ['clean'] if args.stage.endswith('clean') else ['clean','reconstruction','adversarial']
    pairs = {(r['id'],r['variant']) for r in rows}
    assert len(rows) == len(pairs) == len(expected)*len(variants)
    assert pairs == {(uid,v) for uid in expected for v in variants}
    truth = dict(line.split(maxsplit=1) for line in (ROOT/'data/data_aishell/transcript/aishell_transcript_v0.8.txt').read_text().splitlines() if line.strip())
    totals = {v:Counter() for v in variants}
    for r in rows:
        assert r['reference'] == expected[r['id']]['text']
        ref,hyp = normalized(r['reference']),normalized(r['hypothesis'])
        assert ref == normalized(truth[r['id']]) == r['reference_normalized']
        assert hyp == r['hypothesis_normalized']
        assert len(ref) == r['reference_units']
        assert distance(ref,hyp) == r['substitutions']+r['deletions']+r['insertions']
        totals[r['variant']].update({k:r[k] for k in ['substitutions','deletions','insertions','reference_units']})
    for v,counts in totals.items():
        counts['count'] = len(expected)
        counts['error_percent'] = 100*(counts['substitutions']+counts['deletions']+counts['insertions'])/counts['reference_units']
        assert dict(counts) == result['summary'][v]
    generation_runs = []
    original_plan=json.loads((ROOT/'reproduction-runs/mandarin-cer-evaluation/plan.json').read_text())
    if args.stage == 'pilot':
        generation_runs = [folder/'generation']
    elif args.stage == 'test':
        generation_runs = ([Path(s['run']) for s in plan['saved_test_generation']]
                           if args.candidate == 'selected' else [folder/'generation'])
    cfg = json.loads((OUT/f'{args.candidate}-config.json').read_text())
    generated_ids = set()
    trace_checks = 0
    for run in generation_runs:
        metadata = json.loads((run/'run.json').read_text())
        assert metadata['status'] == 'complete'
        assert sha(run/'generated.jsonl') == metadata['generated_sha256']
        assert metadata['surrogate']['sha256'] == plan['checkpoint_sha256']
        assert metadata['features']['encoder_sha256'] == original_plan['artifact_hashes'][str(ROOT/'artifacts/checkpoints/WavLM-Large.pt')]
        assert metadata['features']['vocoder_sha256'] == original_plan['artifact_hashes'][str(ROOT/'artifacts/checkpoints/g_02500000.pt')]
        prior_config = json.loads(json.dumps(metadata['config']))
        expected_config = json.loads(json.dumps(cfg))
        # Decoder correction does not change the differentiable NLL used to
        # generate the earlier selected-setting test WAVs.
        prior_config['surrogate']['decoding'].pop('without_timestamps',None)
        expected_config['surrogate']['decoding'].pop('without_timestamps',None)
        assert prior_config == expected_config
        records = list(map(json.loads,(run/'generated.jsonl').read_text().splitlines()))
        assert len(records) == metadata['count']
        for record in records:
            assert record['id'] not in generated_ids
            generated_ids.add(record['id'])
            assert record['text'] == expected[record['id']]['text']
            assert record['input_audio_sha256'] == expected[record['id']]['audio_sha256']
            assert len(record['trace']) == cfg['attack']['steps']
            for step in record['trace']:
                assert math.isfinite(step['loss_before_update'])
                assert step['radius_after_update'] <= cfg['attack']['rho']+1e-6
                trace_checks += 1
            scores = {r['variant']:r for r in rows if r['id']==record['id']}
            assert all(scores[v]['audio_sha256'] == record['audio'][v]['sha256'] for v in variants)
    if generation_runs:
        assert generated_ids == set(expected)
    comparison = {}
    for v in variants:
        p = plan['paper'].get('clean_cer' if v=='clean' else 'attack_cer' if v=='adversarial' else '')
        if p is not None:
            delta = totals[v]['error_percent']-p
            comparison[v] = {'paper_cer':p,'our_cer':totals[v]['error_percent'],
                             'difference_pp':delta,'within_declared_tolerance':abs(delta)<=plan['comparison_tolerance_pp'][v]}
    audit = {'status':'passed','count':len(expected),'independent_distance_checks':len(rows),
             'generation_runs_audited':len(generation_runs),'finite_projected_attack_steps_checked':trace_checks,
             'all_references_match_official_transcript':True,'summary_sha256':sha(folder/'results.json'),
             'utterances_sha256':sha(folder/'utterances.jsonl'),'paper_comparison':comparison,
             'scope':'Independent audit of saved transcripts, IDs, aggregates, official references and frozen protocol; does not rerun ASR',
             'checked_at':datetime.now(timezone.utc).isoformat()}
    (folder/'independent-audit.json').write_text(json.dumps(audit,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps(audit,indent=2,ensure_ascii=False))


if __name__ == '__main__':
    main()
