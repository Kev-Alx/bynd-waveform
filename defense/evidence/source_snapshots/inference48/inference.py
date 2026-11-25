"""Paired saved-audio defense pilots; deployment rules never receive references."""
from __future__ import annotations

import time
from collections import defaultdict
from pathlib import Path

import torch
from beyondwaveform.evaluation import error_counts,aggregate_errors
from beyondwaveform.io import load_audio

from .common import ROOT,atomic_json,digest,read_json,read_jsonl,stamp,verify_audio_asset,write_jsonl
from .ctc import greedy_ctc
from .phone import PhoneRecognizer,PronunciationLexicon,PhoneWordDecoder
from .whisper_adapter import load_base,beam_candidates,candidate_nll,local_decode


def phone_reconcile(base_text,candidates,phone_lp,lexicon,eta=1.,allow_expansion=True):
    """Candidate scores contain no ground truth or original clean counterpart."""
    base_prons,base_unknown=lexicon.variants(base_text)
    if not base_prons:return base_text,{'fallback':'base OOV/number or empty text','unknown':base_unknown}
    pool=[c for c in candidates if allow_expansion or c['origin']=='whisper_beam']
    scored=[]
    for c in pool:
        acoustic,unknown=lexicon.score(phone_lp,c['text'])
        if acoustic is None or not torch.isfinite(acoustic):continue
        scored.append({**c,'phone_score':float(acoustic),'joint_score':-c['asr_nll']+eta*float(acoustic)})
    if not scored:return base_text,{'fallback':'no scorable candidates'}
    chosen=max(scored,key=lambda c:c['joint_score'])
    return chosen['text'],{'selected':chosen,'scores':scored,'fallback':None}


def summarize(records):
    groups=defaultdict(list)
    for r in records:
        for condition,counts in r['scores'].items():groups[(r['variant'],condition)].append(counts)
    return {f'{variant}/{condition}':{'count':len(rows),**aggregate_errors(rows)}
            for (variant,condition),rows in groups.items()}


def infer(manifest,output,temporal=True,expansion=True):
    output=Path(output)
    rows=read_jsonl(manifest)
    used=[Path(__file__),Path(__file__).with_name('common.py'),Path(__file__).with_name('phone.py'),
          Path(__file__).with_name('ctc.py'),Path(__file__).with_name('whisper_adapter.py'),
          Path(__file__).with_name('temporal.py')]
    identity={'manifest':str(manifest),'manifest_sha256':digest(manifest),
              'source_hashes':{str(p):digest(p) for p in used},'temporal':temporal,'expansion':expansion,
              'kind':'exposed-development non-adaptive saved FV attack',
              'reference_access':'Only scoring/PER/oracle diagnostics; not candidate generation or deployed decisions'}
    meta=output/'identity.json'
    if meta.exists() and read_json(meta)!=identity:raise ValueError('Existing run identity differs')
    atomic_json(meta,identity)
    base,base_meta=load_base()
    phone=PhoneRecognizer()
    lexicon=PronunciationLexicon(phone.labels)
    word_decoder=PhoneWordDecoder(lexicon,read_json(ROOT/'data/training_word_counts.json'))
    atomic_json(output/'models.json',{'base':base_meta,'phone':phone.provenance})
    records=[]
    started=time.time()
    total=3*len(rows)
    for row in rows:
        for variant in ['clean','reconstruction','adversarial']:
            dest=output/'records'/f'{row["id"]}--{variant}.json'
            asset=row['saved_audio'][variant];verify_audio_asset(asset)
            if dest.exists():
                prior=read_json(dest)
                if prior['audio_sha256']!=asset['sha256']:raise ValueError('Changed saved audio')
                records.append(prior);continue
            x=load_audio(asset['path']).cuda()
            begin=time.time()
            with torch.no_grad():
                native=base.transcribe(x)
                text,candidates,features=beam_candidates(base,x,beam=8)
                lp=phone.lexical_log_probs(x)
                phone_ids=greedy_ctc(lp)
                for c in candidates:c['origin']='whisper_beam'
                if expansion:
                    for c in word_decoder.decode(phone_ids):
                        if c['text'] not in {s['text'] for s in candidates}:
                            candidates.append({**c,'origin':'phone_path_lexicon'})
                for c in candidates:c['asr_nll']=float(candidate_nll(base,features,c['text']))
                noexp,detail_noexp=phone_reconcile(text,candidates,lp,lexicon,allow_expansion=False)
                expanded,detail=phone_reconcile(text,candidates,lp,lexicon,allow_expansion=True)
                asr_only=min(candidates,key=lambda c:c['asr_nll'])['text'] if candidates else text
                hypotheses={'base_beam5':native,'base_beam8':text,'ar_rescore_expansion_control':asr_only,
                            'phone_noexp':noexp,'phone_expansion':expanded}
                local_detail=None
                if temporal:
                    local,local_detail=local_decode(base,x)
                    hypotheses['local8s']=local
                target_phones,unknown=lexicon.encode(row['text'])
            scores={condition:error_counts(row['text'],hyp) for condition,hyp in hypotheses.items()}
            # References enter only below, for post-hoc diagnostic ceilings and PER.
            oracle=min((error_counts(row['text'],c['text']) for c in candidates),
                       key=lambda v:sum(v[k] for k in ['substitutions','deletions','insertions']),
                       default=error_counts(row['text'],text))
            beam_oracle=min((error_counts(row['text'],c['text']) for c in candidates if c['origin']=='whisper_beam'),
                       key=lambda v:sum(v[k] for k in ['substitutions','deletions','insertions']),
                       default=error_counts(row['text'],text))
            scores['oracle_candidates_diagnostic_only']=oracle
            scores['oracle_beam_diagnostic_only']=beam_oracle
            phone_per=None
            if target_phones:
                import jiwer
                alignment=jiwer.process_words(' '.join(map(str,target_phones)),' '.join(map(str,phone_ids)))
                phone_per={'substitutions':alignment.substitutions,'deletions':alignment.deletions,
                           'insertions':alignment.insertions,'reference_units':len(target_phones)}
            record={'id':row['id'],'speaker':row['speaker'],'variant':variant,'reference':row['text'],
                    'audio_sha256':asset['sha256'],'hypotheses':hypotheses,'scores':scores,
                    'candidates':candidates,'phone_ids':phone_ids,'phone_per':phone_per,'reference_oov':unknown,
                    'phone_reconciliation':detail,'phone_noexp_reconciliation':detail_noexp,
                    'local':local_detail,'seconds':time.time()-begin,
                    'peak_cuda_bytes':torch.cuda.max_memory_allocated()}
            atomic_json(dest,record);records.append(record)
            atomic_json(output/'progress.json',{'status':'running','completed':len(records),'total':total,
                        'last_id':row['id'],'last_variant':variant,'updated_at':stamp()})
            print(f'infer {len(records)}/{total} {row["id"]} {variant} {record["seconds"]:.2f}s',flush=True)
    summary={'status':'complete','scope':'Non-adaptive exposed-development pilot; no human content validation',
             'results':summarize(records),'seconds':time.time()-started,'records':len(records)}
    write_jsonl(output/'records.jsonl',records);atomic_json(output/'summary.json',summary)
    atomic_json(output/'progress.json',{'status':'complete','completed':len(records),'total':total,'updated_at':stamp()})
    return summary
