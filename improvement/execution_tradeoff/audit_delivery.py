"""Final evidence audit; no model inference or cohort exposure."""
import pathlib,json,hashlib,urllib.request,sys
P=pathlib.Path('/workspace/improvement');D=P/'execution_tradeoff';sys.path[:0]=[str(P),str(D),'/workspace','/workspace/scripts']
import execution_runtime as rt

def read(p):return json.loads(p.read_text())
def main():
 plan=read(D/'plan.json');assert all(rt.digest(p)==h for p,h in plan['bindings'].items()),'New plan binding changed';rt.check(read(rt.RUN/'plan.json'))
 import scaled_scope
 scaled_scope.check()
 result=read(D/'results.json');score=read(D/'pilot/source-summary.json');generated=rt.read_jsonl(D/'pilot/generated.jsonl');scores=[json.loads(line) for line in (D/'pilot/source-scores.jsonl').read_text().splitlines()];assert len(generated)==48 and len(scores)==144 and result['count']==48
 assert len({(r['id'],r['variant']) for r in scores})==144;assert set(s['variant'] for s in scores)=={'raw50','stft_identity','anchor25'};cohort={r['id']:r for r in rt.read_jsonl(P/'execution48/development-48.jsonl')};assert all(r['text']==cohort[r['id']]['text'] for r in generated);assert all(r['reference']==cohort[r['id']]['text'] for r in scores)
 for v in ['raw50','stft_identity','anchor25']:
  rr=[r for r in scores if r['variant']==v];assert len(rr)==48 and sum(r['reference_units'] for r in rr)==994
  s=score['summary'][v];assert abs(100*sum(r[k] for r in rr for k in ['substitutions','deletions','insertions'])/994-s['error_percent'])<1e-10
  assert all(abs(sum(r['quality'][k] for r in rr)/48-s['quality'][k])<1e-7 for k in ['dnsmos','nisqa','utmos'])
 data=read(P/'audio_browser/src/data.json');assert data['buildStatus']=='complete' and len(data['queries']['runs']['rows'])==70 and len(data['queries']['recordings']['rows'])==2808
 assert read(D/'browser_receipt.json')['status']=='pass';assert read(D/'interface_build_receipt.json')['missing_assets']==0
 manifest=read(P/'audio_browser/dist/data-app-build.json');assert rt.digest(P/'audio_browser/dist/index.html')==manifest['html']['sha256'];assert rt.digest(P/'audio_browser/dist'/manifest['snapshot']['path'])==manifest['snapshot']['sha256']
 old=read(D/'served_before/data-app-build.json');assert manifest['runtimeSha256']==old['runtimeSha256'],'Protected runtime changed'
 assert urllib.request.urlopen('http://127.0.0.1:8888/',timeout=5).status==200;server=read(P/'audio_browser_server_status.json');assert server['status']=='running' and server['port']==8888
 receipt={'status':'pass','plan_bindings_preserved':True,'base_and_expanded_frozen_bindings_preserved':True,'cohort48_and_994_words_per_condition':True,'scores144_reconciled':True,'catalogue_conditions':70,'catalogue_recordings':2808,'protected_runtime_unchanged':True,'browser_QA_pass':True,'port8888_HTTP200':True,'server_pid':server['pid'],'actual_heldout_targets_evaluated':result['target_models_evaluated'],'note':'No unseen, confirmation/final or human performance claims; forecasts are conditional'};(D/'final_audit.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt,indent=2))
if __name__=='__main__':main()
