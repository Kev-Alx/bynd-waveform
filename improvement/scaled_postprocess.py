"""Frozen source/quality decisions; no reserved-target feedback for selection."""
import argparse
from datetime import datetime, timezone, timedelta
import hashlib
import json
from pathlib import Path
import numpy as np

OUT=Path('/workspace/improvement');RUN=OUT/'execution_scaled'


def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def floors(s):
    q=s['quality'];return q['dnsmos']>=3 and q['nisqa']>=3.75 and q['utmos']>=3.5
def summary(phase):
    folder=RUN/phase
    if read(folder/'audit.json')['status']!='pass':raise ValueError('Independent phase audit must pass')
    return read(folder/'source-summary.json')['summary']


def decoder():
    s=summary('qualification');big=s['bigvgan_reconstruction'];old=s['fv_reconstruction'];clean=s['clean']
    passed=floors(big) and big['error_percent']<=clean['error_percent']+2 and big['quality']['utmos']>=old['quality']['utmos']-.15
    value={'status':'pass' if passed else 'negative','source_wer_increase':big['error_percent']-clean['error_percent'],
        'utmos_delta_vs_fv_reconstruction':big['quality']['utmos']-old['quality']['utmos'],
        'quality_floors_pass':floors(big),'scope':'48-point reconstruction qualification; no attack effect inferred'}
    path=RUN/'decoder-decision.json'
    if path.exists():raise FileExistsError('Preserve first decoder decision')
    path.write_text(json.dumps(value,indent=2)+'\n');path.chmod(0o444);report();print(json.dumps(value,indent=2))


def select():
    plan=read(RUN/'plan.json');s=summary('fvpilot');entries=[]
    for variant,value in s.items():
        if variant.startswith(('single_','ensemble_raw_','ensemble_eot_')):
            method,endpoint=variant.rsplit('_',1)
            entries.append({'name':variant,'method':method,'step':int(endpoint),'phase':'fvpilot','score':value})
    if read(RUN/'decoder-decision.json')['status']=='pass':
        for variant,value in summary('melpilot').items():
            if variant.startswith('mel_eot_'):
                entries.append({'name':variant,'method':'mel_eot','step':int(variant.rsplit('_',1)[1]),'phase':'melpilot','score':value})
    admissible=[e for e in entries if floors(e['score'])]
    key=lambda e:(-e['score']['error_percent'],-e['score']['quality']['utmos'],e['step'],e['name'])
    useful=sorted([e for e in admissible if e['score']['error_percent']>75],key=key)
    baseline=[e for e in admissible if e['method']=='single']
    baseline.append({'name':'fv_quality50','method':'quality','step':50,'phase':'fvpilot','score':s['fv_quality50']})
    baseline=[e for e in baseline if floors(e['score'])]
    if not baseline:raise ValueError('No quality-admissible original control')
    control=sorted(baseline,key=key)[0]
    candidate=useful[0] if useful else None
    # If the best result is itself the longer original baseline, retain the
    # frozen LR.02 original as comparator and label the outcome a known baseline.
    if candidate and candidate['name']==control['name']:
        control=next(e for e in baseline if e['method']=='quality')
    status='validate' if candidate else 'stop'
    value={'status':status,'created_at_utc':datetime.now(timezone.utc).isoformat(),'candidate':candidate,'control':control,
        'candidate_is_known_baseline':bool(candidate and candidate['method']=='single'),
        'methods':{} if not candidate else {'candidate':{k:candidate[k] for k in ('method','step')},
            'control':{k:control[k] for k in ('method','step')}},
        'quality_admissible_points':[{'name':e['name'],'wer':e['score']['error_percent']} for e in admissible],
        'reason':'Complete-cohort sourceWER strictly above75 and unchanged quality floors; no target outcomes used',
        'primary_source_only':True,'target_models_evaluated':[]}
    path=RUN/'selection.json'
    if path.exists():raise FileExistsError('Preserve source-only checkpoint selection')
    path.write_text(json.dumps(value,indent=2)+'\n');path.chmod(0o444)
    bound=[RUN/'plan.json',path,RUN/'decoder-decision.json',Path(plan['cohorts']['validation']['manifest'])]
    for phase in ('qualification','fvpilot','melpilot'):
        if (RUN/phase/'source-summary.json').exists():bound.extend([RUN/phase/'source-summary.json',RUN/phase/'audit.json'])
    freeze={**value,'bindings':{str(p):sha(p) for p in bound}}
    path=RUN/'validation-freeze.json';path.write_text(json.dumps(freeze,indent=2)+'\n');path.chmod(0o444)
    report();print(json.dumps(value,indent=2))


def validation():
    import scaled_scope as scope
    freeze=scope.validation_freeze();s=summary('validation');candidate=s['candidate']
    passed=floors(candidate) and candidate['error_percent']>75
    value={'status':'evaluate_transfer' if passed else 'stop','created_at_utc':datetime.now(timezone.utc).isoformat(),
        'candidate':freeze['candidate'],'control':freeze['control'],'candidate_source_wer':candidate['error_percent'],
        'candidate_quality':candidate['quality'],'quality_floors_pass':floors(candidate),
        'reselected':False,'reason':'Frozen76-point source validation must again exceed75WER and meet all quality floors'}
    path=RUN/'transfer-freeze.json'
    if path.exists():raise FileExistsError('Preserve validation decision')
    bound=[RUN/'validation-freeze.json',RUN/'validation/source-summary.json',RUN/'validation/audit.json']
    value['bindings']={str(p):sha(p) for p in bound}
    path.write_text(json.dumps(value,indent=2)+'\n');path.chmod(0o444);report();print(json.dumps(value,indent=2))


def report():
    stamp=datetime.now(timezone(timedelta(hours=8))).isoformat(timespec='seconds')
    lines=['# Larger-model research results','',f'Updated **{stamp} (Asia/Taipei)**.','',
        'Goal: sourceWER>75% with unchanged automatic quality floors, followed by measured held-out transfer '
        'and defense persistence. These are separate endpoints. No human-intelligibility or novelty claim is made.','']
    for phase in ('qualification','fvpilot','melpilot','validation'):
        path=RUN/phase/'source-summary.json'
        if not path.exists():continue
        lines += [f'## {phase}','','| Condition | N | SourceWER% | DNSMOS | NISQA | UTMOS |','|---|---:|---:|---:|---:|---:|']
        for variant,s in read(path)['summary'].items():
            q=s['quality'];lines.append(f"| {variant} | {s['count']} | {s['error_percent']:.4f} | {q['dnsmos']:.4f} | {q['nisqa']:.4f} | {q['utmos']:.4f} |")
        lines += ['']
    if (RUN/'selection.json').exists():
        value=read(RUN/'selection.json');lines += [f"Source-only advancement: **{value['status']}**. "
            f"Candidate: **{value['candidate']['name'] if value['candidate'] else 'none'}**. "
            f"Known-baseline candidate: **{value['candidate_is_known_baseline']}**.",'']
    if (RUN/'transfer-freeze.json').exists():lines += [f"Transfer-stage decision: **{read(RUN/'transfer-freeze.json')['status']}**.",'']
    if (RUN/'transfer/results.json').exists():
        value=read(RUN/'transfer/results.json');lines += ['## Held-out transfer and preprocessing','',
            'Full target/condition scores and paired descriptive intervals are in '
            '[the transfer receipt](execution_scaled/transfer/results.json).', '',
            f"Measured preservation screen: **{value.get('preservation_status','incomplete')}**. "
            'Known-transform adaptation and withheld-parameter tests are reported separately.','']
    lines += ['The76-point validation is exposed development with shared speakers. Confirmation48/final200 remain unused. '
        'LPF and WaveGuard resampling cover named preprocessing procedures; missing training-defense weights, '
        'sentence-compatible AudioPure, and full WaveGuard detection are not measured. A blinded human study remains required.','',
        '[Protocol and goals](14_scaled_research_protocol.md) · [Live status](JOB_STATUS.md) · [Changes explained](11_changes_and_goals.md)','']
    (OUT/'15_scaled_results.md').write_text('\n'.join(lines))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['decoder','select','validation','report'])
    stage=p.parse_args().stage
    {'decoder':decoder,'select':select,'validation':validation,'report':report}[stage]()
