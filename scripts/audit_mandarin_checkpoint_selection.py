"""Independent character-distance audit of validation-only checkpoint scores."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import unicodedata

ROOT=Path(__file__).resolve().parents[1]


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def normalize(text):
    text=unicodedata.normalize('NFKC',text).lower()
    return ''.join(c for c in text if not c.isspace() and not unicodedata.category(c).startswith('P'))


def distance(a,b):
    previous=list(range(len(b)+1))
    for i,x in enumerate(a,1):
        current=[i]
        for j,y in enumerate(b,1):
            current.append(min(current[-1]+1,previous[j]+1,previous[j-1]+(x!=y)))
        previous=current
    return previous[-1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=ROOT/'reproduction-runs/mandarin-cer-checkpoint-selection')
    parser.add_argument('--stage',choices=['pilot','validation'],default='validation')
    parser.add_argument('--candidate',default='all')
    args=parser.parse_args();root=args.root.resolve();plan=json.loads((root/'plan.json').read_text())
    manifest=root/('pilot-48.jsonl' if args.stage=='pilot' else 'validation-1435.jsonl')
    assert sha(manifest)==plan['pilot_sha256' if args.stage=='pilot' else 'validation_sha256']
    expected={r['id']:r for r in map(json.loads,manifest.read_text().splitlines())}
    assert all(r['role']=='validation' and r['split']!='test' for r in expected.values())
    truth=dict(line.split(maxsplit=1) for line in (ROOT/'data/data_aishell/transcript/aishell_transcript_v0.8.txt').read_text().splitlines() if line.strip())
    candidates=list(plan['targets']) if args.candidate=='all' else [args.candidate]
    audits={}
    for name in candidates:
        folder=root/args.stage/name;result=json.loads((folder/'results.json').read_text())
        assert result['status']=='complete' and result['count']==len(expected)
        assert result['plan_sha256']==sha(root/'plan.json')
        assert result['target']['sha256']==plan['targets'][name]['checkpoint_sha256']
        assert result['target']['language']=='zh' and result['target']['decoding']['without_timestamps'] is True
        rows=list(map(json.loads,(folder/'utterances.jsonl').read_text().splitlines()))
        assert len(rows)==len({r['id'] for r in rows})==len(expected)
        assert {r['id'] for r in rows}==set(expected)
        counts=Counter()
        for r in rows:
            original=expected[r['id']]
            assert r['reference']==original['text'] and r['audio_sha256']==original['audio_sha256']
            ref,hyp=normalize(r['reference']),normalize(r['hypothesis'])
            assert ref==normalize(truth[r['id']])==r['reference_normalized']
            assert hyp==r['hypothesis_normalized'] and len(ref)==r['reference_units']
            assert distance(ref,hyp)==r['substitutions']+r['deletions']+r['insertions']
            counts.update({k:r[k] for k in ['substitutions','deletions','insertions','reference_units']})
        counts['count']=len(rows)
        counts['error_percent']=100*(counts['substitutions']+counts['deletions']+counts['insertions'])/counts['reference_units']
        assert dict(counts)==result['summary']['clean']
        audit={'status':'passed','candidate':name,'count':len(rows),'independent_character_distance_checks':len(rows),
               'all_references_match_official_transcripts':True,'summary':dict(counts),
               'results_sha256':sha(folder/'results.json'),'utterances_sha256':sha(folder/'utterances.jsonl')}
        (folder/'independent-audit.json').write_text(json.dumps(audit,indent=2)+'\n');audits[name]=audit
    if args.candidate=='all' and args.stage=='validation':
        result=json.loads((root/'validation-results.json').read_text())
        winner=min(list(plan['targets']),key=lambda n:audits[n]['summary']['error_percent'])
        assert result['selection']==winner
    print(json.dumps(audits,indent=2))


if __name__=='__main__':main()
