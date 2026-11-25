"""Build off-line, stage immutable assets, then replace index atomically on NFS."""
import pathlib,shutil,os,subprocess,json,ctypes,time,hashlib
P=pathlib.Path('/workspace/improvement');D=P/'execution_tradeoff';APP=P/'audio_browser';STAGE=D/'interface_build'
def main():
 if STAGE.exists():raise FileExistsError('Preserve earlier staged build; version the staging path')
 def ignore(directory,names):return [n for n in names if n in ['dist','node_modules','.git','audio','.data-app-offline','.openai']]
 shutil.copytree(APP,STAGE,ignore=ignore)
 subprocess.run(['/usr/bin/node','/home/codex/.codex/plugins/cache/openai-curated-remote/data-analytics/1.0.11/scripts/data-app.mjs','build','--project-dir',str(STAGE),'--separate-data'],check=True)
 new=STAGE/'dist';(new/'audio').mkdir(exist_ok=True);assets=list((APP/'src/content/assets/audio').glob('*.wav'))
 for source in assets:
  target=new/'audio'/source.name
  try:os.link(source,target)
  except OSError:shutil.copyfile(source,target)
 shutil.copyfile(P/'all_results.csv',new/'all_results.csv')
 for name in ['latency_per_recording.csv','runtime_catalogue.csv','results.json']:
  if (D/name).exists():shutil.copyfile(D/name,new/name)
 data=json.loads((APP/'src/data.json').read_text());expected={r['audio'] for r in data['queries']['recordings']['rows']};assert all((new/path).is_file() for path in expected)
 # NFS here rejects directory RENAME_EXCHANGE (EINVAL). Keep all old immutable
 # assets/sidecars and stage new files before one atomic index replacement.
 current=APP/'dist';backup=D/'served_before';backup.mkdir()
 for source in current.iterdir():
  if source.is_file():shutil.copyfile(source,backup/source.name)
 def publish_file(source,target):
  target.parent.mkdir(parents=True,exist_ok=True)
  if target.exists() and hashlib.sha256(source.read_bytes()).digest()==hashlib.sha256(target.read_bytes()).digest():return
  tmp=target.with_name('.tradeoff-stage-'+target.name)
  if tmp.exists():raise FileExistsError('Preserve incomplete staging file')
  try:os.link(source,tmp)
  except OSError:shutil.copyfile(source,tmp)
  os.replace(tmp,target)
 for source in new.rglob('*'):
  if source.is_file() and source.name!='index.html':publish_file(source,current/source.relative_to(new))
 assert all((current/path).is_file() for path in expected)
 publish_file(new/'index.html',current/'index.html')
 receipt={'status':'complete','build_status':data['buildStatus'],'atomic_index_replace':True,'atomic_exchange':False,'filesystem':'NFS; directory exchange unsupported, staged immutable assets and retained old sidecars','audio_assets':len(assets),'referenced_assets':len(expected),'missing_assets':0,'prior_served_metadata_preserved':str(backup),'staged_build_preserved':str(new),'source_snapshot_sha256':hashlib.sha256((APP/'src/data.json').read_bytes()).hexdigest()};(D/'interface_build_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))
if __name__=='__main__':main()
