"""Aggregate real saved results and independently verify lexical counts/coverage."""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
from beyondwaveform.evaluation import aggregate_errors,error_counts

from .common import ROOT,atomic_json,digest,read_json,read_jsonl,stamp


def paired_interval(a,b,iterations=10000):
    keys=lambda r:(r['id'],r['variant'])
    amap={keys(r):r for r in a};bmap={keys(r):r for r in b}
    if set(amap)!=set(bmap):raise ValueError('Paired interval requires identical coverage')
    speakers=sorted({r['speaker'] for r in a})
    totals=[]
    for speaker in speakers:
        ra=[r for r in a if r['speaker']==speaker]
        rb=[bmap[keys(r)] for r in ra]
        na=sum(sum(r[k] for k in ['substitutions','deletions','insertions']) for r in ra)
        nb=sum(sum(r[k] for k in ['substitutions','deletions','insertions']) for r in rb)
        den=sum(r['reference_units'] for r in ra)
        totals.append([na,nb,den])
    totals=np.array(totals,dtype=float)
    rng=np.random.default_rng(20261022)
    indices=rng.integers(0,len(speakers),size=(iterations,len(speakers)))
    samples=totals[indices].sum(1)
    values=100*(samples[:,0]-samples[:,1])/samples[:,2]
    return {'difference_pp':100*(totals[:,0].sum()-totals[:,1].sum())/totals[:,2].sum(),
            'interval_95':np.quantile(values,[.025,.975]).tolist(),'speakers':len(speakers),
            'bootstrap_iterations':iterations,'scope':'descriptive exposed-development interval; single training seed'}


def audit_records(records,hypothesis_key='hypothesis'):
    for row in records:
        expected=error_counts(row['reference'],row[hypothesis_key])
        for key in ['substitutions','deletions','insertions','reference_units']:
            if expected[key]!=row[key]:raise ValueError(f'Saved lexical counts differ: {row["id"]}/{key}')


def report():
    folder=ROOT/'runs/pilot-v1'
    plan=read_json(folder/'plan.json')
    inference=read_jsonl(folder/'inference48/records.jsonl')
    rows=read_jsonl(ROOT/'data/development48.jsonl')
    expected={(r['id'],v) for r in rows for v in ['clean','reconstruction','adversarial']}
    if {(r['id'],r['variant']) for r in inference}!=expected or len(inference)!=len(expected):
        raise ValueError('Incomplete/duplicate inference coverage')
    groups=defaultdict(list)
    phone_counts=defaultdict(list)
    rescored=0
    for row in inference:
        for name,hypothesis in row['hypotheses'].items():
            record={'id':row['id'],'speaker':row['speaker'],'variant':row['variant'],'reference':row['reference'],
                    'hypothesis':hypothesis,**row['scores'][name]}
            audit_records([record]);rescored+=1;groups[name].append(record)
        for name in ['oracle_candidates_diagnostic_only','oracle_beam_diagnostic_only']:
            groups[name].append({'id':row['id'],'speaker':row['speaker'],'variant':row['variant'],**row['scores'][name]})
        if row.get('phone_per'):phone_counts[row['variant']].append(row['phone_per'])
    learned={}
    for condition in plan['training']['conditions']:
        training=read_json(folder/f'train-{condition}/training.json')
        if training['status']!='complete' or digest(folder/f'train-{condition}/adapters.pt')!=training['adapter_sha256']:
            raise ValueError('Incomplete/tampered learned adapter')
        records=read_jsonl(folder/f'train-{condition}/evaluation/records.jsonl')
        if {(r['id'],r['variant']) for r in records}!=expected or len(records)!=len(expected):
            raise ValueError('Incomplete learned-model evaluation coverage')
        audit_records(records);rescored+=len(records);groups[condition]=records;learned[condition]=training
    results={}
    for condition,records in groups.items():
        results[condition]={}
        for variant in ['clean','reconstruction','adversarial']:
            subset=[r for r in records if r['variant']==variant]
            results[condition][variant]={'count':len(subset),**aggregate_errors(subset)}
    intervals={}
    for name in ['phone_expansion','phone_noexp','local8s','clean_ft','reconstruction_ft','union_at']:
        baseline='base_beam8' if name.startswith('phone') else 'base_beam5'
        intervals[name]={}
        for variant in ['clean','adversarial']:
            intervals[name][variant]=paired_interval([r for r in groups[name] if r['variant']==variant],
                                                     [r for r in groups[baseline] if r['variant']==variant])
    adaptive=read_json(folder/'adaptive/summary.json')
    if adaptive['status']!='complete':raise ValueError('Adaptive stage incomplete')
    delta_checks=0
    import torch
    for p in (folder/'adaptive').glob('*/*/restart-*/step-*.json'):
        record=read_json(p)
        if digest(record['delta_path'])!=record['delta_sha256']:raise ValueError('Changed adaptive delta')
        if digest(record['q_path'])!=record['q_sha256']:raise ValueError('Changed adaptive clean features')
        q=torch.load(record['q_path'],weights_only=True,map_location='cpu')
        delta=torch.load(record['delta_path'],weights_only=True,map_location='cpu')
        radius=float(delta.norm()/(q.norm()+1e-8))
        if radius>.100001 or abs(radius-record['actual_relative_radius'])>1e-6:
            raise ValueError('Independent adaptive-radius verification failed')
        if digest(record['audio']['path'])!=record['audio']['sha256']:raise ValueError('Changed adaptive waveform')
        audit_records([record]);delta_checks+=1
    per={v:{'scorable_utterances':len(rr),**aggregate_errors(rr)} for v,rr in phone_counts.items()}
    gates={}
    for name in ['phone_expansion','local8s','union_at']:
        baseline='base_beam8' if name.startswith('phone') else 'base_beam5'
        clean_regret=results[name]['clean']['error_percent']-results[baseline]['clean']['error_percent']
        gain=results[baseline]['adversarial']['error_percent']-results[name]['adversarial']['error_percent']
        gates[name]={'clean_regret_pp':clean_regret,'nonadaptive_gain_pp':gain,
                     'clean_gate_pass':clean_regret<=1.,'nonadaptive_10pp_screen_pass':gain>=10.,
                     'promotion':'No final promotion: direct adaptive/generalization/human-validity conditions also required'}
    package={'status':'complete','created_at':stamp(),'cohort':'48 exposed dev-clean utterances / 36 speakers',
             'results':results,'phone_error':per,'descriptive_speaker_intervals':intervals,
             'training':learned,'adaptive':adaptive,'gates':gates,
             'independent_saved_artifact_audit':{'lexical_rows_rescored':rescored,'adaptive_delta_radius_checks':delta_checks,
                 'coverage_complete':True,'score_is_fresh_neural_reinference':False,
                 'explanation':'Model hypotheses were newly generated earlier in this queue; this separate audit recomputes saved counts/hashes/radii.'},
             'limits':['Exposed development only','Training uses 80 clips / ~10 minutes, one seed / 128 outer updates',
                       'Nonadaptive48 reuses historical paper-budget FV; current-model adaptive check only three clips',
                       'No new final/confirmation reserve, held-out generator, human study or certification',
                       'Phone lexicon expansion approximates a greedy phone path, not exact full-lattice decoding',
                       'No full TransferBreaker/AudioPure or equal-resource second-ASR ensemble comparison yet']}
    atomic_json(ROOT/'reports/pilot-v1-results.json',package)
    lines=['# First defense implementation and development results','',f"Completed {package['created_at']}.",'',
           'All values below are real newly executed development measurements. They are not final-test or certified robustness results.','',
           '| Condition | Clean WER % | Reconstruction WER % | Fixed FV attack WER % |','|---|---:|---:|---:|']
    order=['base_beam5','base_beam8','ar_rescore_expansion_control','phone_noexp','phone_expansion','local8s',
           'clean_ft','reconstruction_ft','union_at','oracle_beam_diagnostic_only','oracle_candidates_diagnostic_only']
    for name in order:
        rr=results[name]
        lines.append(f"| {name} | {rr['clean']['error_percent']:.3f} | {rr['reconstruction']['error_percent']:.3f} | {rr['adversarial']['error_percent']:.3f} |")
    lines+=['','Oracle rows use reference transcripts for analysis only and are not deployable defenses.','',
            '## Direct adaptive FV audit','',
            'Three duration-stratified development clips; each learned recognizer and the base are reattacked directly. Entries retain the worst WER per utterance across three restarts and all observed endpoints up to the stated budget.','',
            '| Condition / budget | Utterances | WER % |','|---|---:|---:|']
    for name,value in adaptive['results'].items():lines.append(f"| {name} | {value['count']} | {value['error_percent']:.3f} |")
    lines+=['','The smaller phone-joint check targets both ASR and phone-CTC, then evaluates actual hard reconciliation. It is a weaker one-restart/50-step check, not equivalent to the recognizer audit.','',
            '## Training and interpretation','']
    for name,value in learned.items():
        lines.append(f"- {name}: {value['updates']} updates, {value['trainable_parameters']:,} trainable parameters, {value['seconds']:.1f} seconds, family counts {value['family_counts']}.")
    lines+=['','All three prototypes are implemented. Passing a non-adaptive screen does not establish robustness; the direct adaptive results, clean cost and mechanism controls must be considered together.','',
            'The machine-readable result includes paired speaker bootstrap intervals, phone error, gates, source provenance and the separate saved-artifact audit.','',
            '## Limits','']+[f'- {v}.' for v in package['limits']]
    lines+=['','## Artifacts','',
            '- [Machine-readable results](pilot-v1-results.json)',
            '- [Registered plan](/workspace/defense/runs/pilot-v1/plan.json)',
            '- [Inference records](/workspace/defense/runs/pilot-v1/inference48/records.jsonl)',
            '- [Live/completed job state](/workspace/defense/JOB_STATUS.md)',
            '- [Handoff](/workspace/defense/HANDOFF.md)']
    (ROOT/'reports/07_IMPLEMENTATION_RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return {'report':str(ROOT/'reports/07_IMPLEMENTATION_RESULTS.md'),'gates':gates,'adaptive':adaptive['results']}
