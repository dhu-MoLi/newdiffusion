from pathlib import Path
import json
import os
import subprocess
import sys
import time
import hashlib

tree = Path(__file__).resolve().parents[2]
root = Path('/home/dell/2T/lm/new_projects/category_anomaly_diffusion/runs/v06_r/visa/chewinggum/20261012-strong-r')
status_path = root / 'training_group.json'
assert json.loads((root / 'smoke_checks.json').read_text())['passed']
frozen = json.loads((root / 'inputs.json').read_text())['formal_a_h_and_composition_sha256']
for relative, expected in frozen.items():
    assert hashlib.sha256((tree / relative).read_bytes()).hexdigest() == expected, relative
if status_path.exists():
    raise FileExistsError(status_path)
head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=tree, text=True).strip()
status = {'schema_version': 'v06_r_training_group_v1', 'state': 'starting', 'arms': {}, 'started_at_unix': time.time(), 'experiment_git_head': head}

def update():
    temporary = status_path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(status, indent=2) + '\n', encoding='utf-8')
    temporary.replace(status_path)

update()
environment = dict(os.environ, PYTHONPATH='src:.', CUBLAS_WORKSPACE_CONFIG=':4096:8')
for arm in ('latent_control', 'rgb', 'rgb_edge'):
    destination = root / 'training' / arm
    status.update(state='running', current_arm=arm)
    status['arms'][arm] = {'state': 'running', 'started_at_unix': time.time(), 'output': str(destination)}
    update()
    command = [sys.executable, '-u', str(tree / 'tools/v06_r/run_experiment.py'),
               '--config', str(tree / f'configs/experiments/v06_r/chewinggum_{arm}_20261012.yaml'),
               '--output', str(destination)]
    print(json.dumps({'group': 'v06_r', 'event': 'starting_arm', 'arm': arm}), flush=True)
    process = subprocess.run(command, cwd=tree, env=environment)
    if process.returncode:
        status.update(state='failed', failed_arm=arm, exit_code=process.returncode)
        status['arms'][arm]['state'] = 'failed'
        update()
        sys.exit(process.returncode)
    summary = json.loads((destination / 'summary.json').read_text(encoding='utf-8'))
    assert summary['complete'] and summary['completed_steps'] == 1200 and not summary['smoke']
    status['arms'][arm].update(state='complete', completed_at_unix=time.time(), checkpoint=summary['checkpoint'], completed_steps=1200)
    update()
status.update(state='complete', completed_at_unix=time.time())
update()
print(json.dumps({'group': 'v06_r', 'event': 'all_arms_complete', 'steps_per_arm': 1200}), flush=True)
