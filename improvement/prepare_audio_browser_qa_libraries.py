"""Resolve browser libraries into an isolated directory, without an OS install."""
import json
import os
import subprocess
from datetime import datetime,timezone
from pathlib import Path

ROOT=Path('/workspace/improvement');QA=ROOT/'audio_browser_qa'
def state(**kwargs):
    p=ROOT/'audio_browser_qa_libraries_status.json';p.write_text(json.dumps({'updated_at_utc':datetime.now(timezone.utc).isoformat(),**kwargs},indent=2))
def main():
    for p in ['apt/state/lists/partial','apt/cache/archives/partial','debs','libs']:(QA/p).mkdir(parents=True,exist_ok=True)
    args=['apt-get','-o','Dir::Etc::parts=/nonexistent','-o',f'Dir::State={QA}/apt/state','-o',f'Dir::Cache={QA}/apt/cache','-o','Debug::NoLocking=1']
    state(status='running',phase='reading package indexes')
    subprocess.run([*args,'update'],check=True)
    state(status='running',phase='downloading browser libraries')
    subprocess.run([*args,'download','libnspr4','libnss3','libatk1.0-0','libatk-bridge2.0-0','libatspi2.0-0','libxcomposite1','libxdamage1'],cwd=QA/'debs',check=True)
    for p in (QA/'debs').glob('*.deb'):subprocess.run(['dpkg-deb','-x',str(p),str(QA/'libs')],check=True)
    state(status='complete')
if __name__=='__main__':
    try:main()
    except BaseException as e:state(status='failed',error=repr(e));raise
