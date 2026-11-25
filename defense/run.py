#!/usr/bin/env python3
"""Entry point for isolated, resumable development experiments."""
from pathlib import Path
import os
import sys

ROOT=Path(__file__).resolve().parent
sys.dont_write_bytecode=True
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'runtime'),str(ROOT.parent)]
from defense_lab.common import configure_runtime,initialize_torch,ROOT,atomic_json,stamp
configure_runtime()


def main():
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=['register','infer','train','adaptive','report'])
    parser.add_argument('--manifest',default=str(ROOT/'data/development48.jsonl'))
    parser.add_argument('--output',default=str(ROOT/'runs/pilot-v1/inference'))
    parser.add_argument('--condition',default='union_at')
    parser.add_argument('--no-temporal',action='store_true')
    parser.add_argument('--no-expansion',action='store_true')
    args=parser.parse_args()
    initialize_torch()
    if args.stage!='register':
        from defense_lab.common import read_json,digest
        plan=read_json(ROOT/'runs/pilot-v1/plan.json')
        for path,expected in {**plan['inputs'],**plan['protected_manifests']}.items():
            if digest(path)!=expected:raise ValueError(f'Changed registered input: {path}')
    if args.stage=='register':
        from defense_lab.register import register
        print(register());return
    if args.stage=='infer':
        from defense_lab.inference import infer
        print(infer(Path(args.manifest),Path(args.output),not args.no_temporal,not args.no_expansion));return
    if args.stage=='train':
        from defense_lab.training import train
        print(train(args.condition));return
    if args.stage=='adaptive':
        from defense_lab.adaptive import audit
        print(audit());return
    if args.stage=='report':
        from defense_lab.reporting import report
        print(report());return


if __name__=='__main__':
    try:main()
    except BaseException as error:
        import traceback
        if not isinstance(error,SystemExit) or error.code:
            atomic_json(ROOT/'runs/last_failure.json',{'at':stamp(),'argv':sys.argv,'error':repr(error),
                                                     'traceback':traceback.format_exc()})
        raise
