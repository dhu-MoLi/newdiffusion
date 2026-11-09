from pathlib import Path
import json
import torch
from cadiff.p2_a1.config import P2A1Config
from cadiff.v06_c.can_exact_mask_r import load_can_r_modules
from cadiff.v06_c.visa_exact_mask_r import CHECKPOINT_SCHEMA

root = Path('/home/dell/2T/lm/new_projects/category_anomaly_diffusion/runs/v06_r/visa/chewinggum/20261012-strong-r')
smoke = root / 'smoke'
names = ('latent_control-t100-resume', 'latent_control-t100-uninterrupted', 'rgb-t100', 'rgb_edge-t100', 'rgb-t900', 'rgb_edge-t900')
result = {'schema_version': 'v06_r_smoke_checks_v1', 'passed': False, 'runs': {}}
for name in names:
    folder = smoke / name
    summary = json.loads((folder / 'summary.json').read_text(encoding='utf-8'))
    assert summary['complete'] and summary['smoke'] and summary['last']['finite'] and summary['last']['grad_l2'] > 0
    checkpoint = Path(summary['checkpoint'])
    metadata = json.loads((checkpoint / 'metadata.json').read_text(encoding='utf-8'))
    raw = dict(metadata['config'])
    for field in ('defect_types', 'eval_timesteps'):
        raw[field] = tuple(raw[field])
    config = P2A1Config(**raw)
    _, conditioner, adapter, _ = load_can_r_modules(checkpoint, config=config, device=torch.device('cpu'), expected_schema=CHECKPOINT_SCHEMA)
    assert all(torch.isfinite(p).all() for p in (*conditioner.parameters(), *adapter.parameters()))
    rows = [json.loads(line) for line in (folder / 'training.jsonl').read_text(encoding='utf-8').splitlines()]
    if name.endswith('t900'):
        assert all(row['decode_count'] == 0 and row['rgb_gt'] == 0 and row['edge_gt'] == 0 for row in rows)
    if name in ('rgb-t100', 'rgb_edge-t100'):
        assert all(row['decode_count'] == 1 and row['rgb_gt'] > 0 for row in rows)
    result['runs'][name] = {'steps': summary['completed_steps'], 'reload': True, 'all_finite': all(row['finite'] for row in rows), 'last_gradient_norm': summary['last']['grad_l2']}
    del conditioner, adapter

def checkpoint(name):
    return json.loads((smoke / name / 'checkpoint-step-000004/metadata.json').read_text(encoding='utf-8'))
first, second = checkpoint('latent_control-t100-resume'), checkpoint('latent_control-t100-uninterrupted')
for filename in ('adapter.safetensors', 'conditioner.safetensors'):
    assert first['files'][filename]['sha256'] == second['files'][filename]['sha256'], filename
result['control_gpu_resume_bitwise_equal_to_uninterrupted'] = True
result['passed'] = True
(root / 'smoke_checks.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
print(json.dumps(result))
