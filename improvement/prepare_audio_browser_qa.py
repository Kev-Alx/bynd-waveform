"""Install isolated browser QA tools; logs and downloads stay in improvement."""
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path('/workspace/improvement')
QA = ROOT/'audio_browser_qa'


def state(**kwargs):
    (ROOT/'audio_browser_qa_status.json').write_text(json.dumps({'updated_at_utc':datetime.now(timezone.utc).isoformat(),**kwargs},indent=2))


def main():
    QA.mkdir(exist_ok=True)
    env=dict(os.environ, npm_config_cache=str(QA/'npm-cache'), PLAYWRIGHT_BROWSERS_PATH=str(QA/'browsers'))
    commands=[['npm','install','--prefix',str(QA),'--save-exact','playwright@1.56.1'],
              ['node',str(QA/'node_modules/playwright/cli.js'),'install','chromium','--only-shell']]
    for i,cmd in enumerate(commands,1):
        state(status='running',stage=i,total=len(commands));subprocess.run(cmd,env=env,check=True)
    state(status='complete')


if __name__=='__main__':
    try:main()
    except BaseException as e:state(status='failed',error=repr(e));raise
