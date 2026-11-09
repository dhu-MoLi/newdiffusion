from dataclasses import replace
from pathlib import Path
import argparse
import json
import yaml
from cadiff.v06_c.visa_exact_mask_r import visa_p2_a1_config
from cadiff.v06_r.config import RExperimentConfig
from cadiff.v06_r.runner import train_r_experiment

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--smoke', action='store_true')
parser.add_argument('--steps', type=int)
parser.add_argument('--forced-timestep', type=int)
parser.add_argument('--resume', type=Path)
parser.add_argument('--stop-after', type=int)
args = parser.parse_args()
raw = yaml.safe_load(args.config.read_text(encoding='utf-8'))
inputs = json.loads(Path(raw['inputs']).read_text(encoding='utf-8'))
config = replace(visa_p2_a1_config(inputs['category'], tuple(inputs['defect_types'])),
                 steps=args.steps or raw['steps'], seed=raw['seed'],
                 context_fill_seed=raw['context_fill_seed'], noise_seed=raw['noise_seed'])
loss = RExperimentConfig(arm=raw['arm'], minimum_snr=raw['minimum_snr'],
                        rgb_weight=raw['rgb_weight'], edge_weight=raw['edge_weight'],
                        save_every=raw['save_every'])
result = train_r_experiment(paths_yaml=Path(raw['paths_yaml']),
    manifest=Path(inputs['training_manifest']), dataset_root=Path(inputs['dataset_root']),
    output=args.output, config=config, experiment=loss, device='cuda:0',
    smoke=args.smoke, forced_timestep=args.forced_timestep, resume=args.resume,
    stop_after=args.stop_after)
print(json.dumps(result, ensure_ascii=False, sort_keys=True))
