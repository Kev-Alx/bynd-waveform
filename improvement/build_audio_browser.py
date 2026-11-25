"""Build the reviewed app and stage every original WAV for static serving."""
import json
import os
import shutil
import subprocess
from datetime import datetime,timezone
from pathlib import Path

ROOT=Path('/workspace/improvement');APP=ROOT/'audio_browser'
PLUGIN=Path('/home/codex/.codex/plugins/cache/openai-curated-remote/data-analytics/1.0.11')

def main():
    subprocess.run(['/usr/bin/node',str(PLUGIN/'scripts/data-app.mjs'),'build','--project-dir',str(APP),'--separate-data'],check=True)
    dist=APP/'dist';dst=dist/'audio';dst.mkdir(exist_ok=True)
    assets=list((APP/'src/content/assets/audio').glob('*.wav'))
    for source in assets:
        target=dst/source.name
        if not target.exists():
            try:os.link(source,target)
            except OSError:shutil.copyfile(source,target)
        if target.stat().st_size!=source.stat().st_size:raise ValueError(f'Incomplete staged audio: {source.name}')
    shutil.copyfile(ROOT/'all_results.csv',dist/'all_results.csv')
    data=json.loads((APP/'src/data.json').read_text())
    expected={r['audio'] for r in data['queries']['recordings']['rows']}
    missing=[s for s in expected if not (dist/s).is_file()]
    if missing:raise ValueError(f'{len(missing)} missing playback assets')
    receipt={'status':'complete','built_at_utc':datetime.now(timezone.utc).isoformat(),'audio_assets':len(assets),'referenced_assets':len(expected),'missing_assets':len(missing),'build_status':data['buildStatus']}
    (ROOT/'audio_browser_build_receipt.json').write_text(json.dumps(receipt,indent=2))
    print(json.dumps(receipt),flush=True)

if __name__=='__main__':main()
