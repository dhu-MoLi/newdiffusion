"""Fixed historical output engineering checks, not a semantic mask metric."""
import collections
import datetime
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image

ROOT = Path('/home/dell/2T/lm/new_projects/category_anomaly_diffusion')
WORK = ROOT / '.worktrees/visa-r-baseline-20261012'
AUDIT = WORK / 'runs/v06_audit/visa/20261015-split-mask-coverage'
OUT = AUDIT / 'existing_r_output_check'


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def main():
    audit_status = json.loads((AUDIT / 'status.json').read_text())
    if audit_status['state'] not in ('complete', 'completed'):
        raise RuntimeError('Full audit must complete before this dependent check')
    groups = collections.defaultdict(list)
    index = AUDIT / 'attempt_index.jsonl'
    for line in index.open():
        row = json.loads(line)
        groups[(row['category'], row['type_slug'])].append(row)
    chosen = [r for key in sorted(groups) for r in sorted(groups[key], key=lambda x: (x['index'], x['job_id']))[:2]]
    assert len(groups) == 58 and len(chosen) == 116
    identity_source = AUDIT / 'identity_contract.jsonl'
    canonical_normals = {}
    for line in identity_source.open():
        r = json.loads(line)
        if r['role'] == 'normal_train':
            canonical_normals[(r['category'], r['source_id'])] = r
    OUT.mkdir(exist_ok=False)
    started = now()
    write('protocol.json', {
        'registered_utc': started, 'parent_audit': str(AUDIT), 'parent_attempt_index_sha256': sha(index),
        'identity_contract_sha256': sha(identity_source),
        'runner_sha256': sha(__file__), 'selection': 'first two indices per supported type, from preregistered first-five list; include failed and under64, never replace',
        'sample_count': len(chosen), 'supported_type_count': len(groups), 'gpu': False,
        'scope': 'Stage2a fixed-identity engineering precheck of existing normal-conditioned outputs; no generation or training',
        'measures': ['exact H-exterior RGB equality', 'exact zero change throughout H', 'descriptive H-interior absolute RGB difference'],
        'limitations': ['not independent visible-defect G annotation', 'not a type or realism pass', 'historical pooled R training contracts differ', 'no inference about all attempts from these fixed samples'],
        'fixed_attempt_keys': [r['attempt_key'] for r in chosen],
    })
    write('status.json', {'state': 'running', 'started_utc': started})
    results = []
    for row in chosen:
        d = {k: row[k] for k in ('attempt_key', 'category', 'type_slug', 'index', 'job_id', 'numeric_success', 'record_failed', 'violations')}
        d['human_semantic_status'] = 'unreviewed'
        folder = Path(row['run_path']) / 'samples' / row['job_id']
        paths = {key: folder / filename for key, filename in [('normal', 'normal.png'), ('generated', 'final_y.png'), ('mask', 'm_final.png')]}
        d['source_paths'] = {key: str(path) for key, path in paths.items()}
        if not row['numeric_success']:
            d['check_state'] = 'ineligible_historical_attempt_retained'
            results.append(d)
            continue
        arrays = {}
        for key, path in paths.items():
            with Image.open(path) as im:
                arrays[key] = np.asarray(im.convert('L' if key == 'mask' else 'RGB')).copy()
        n, y, h = arrays['normal'], arrays['generated'], arrays['mask'] > 0
        if n.shape != (512, 512, 3) or y.shape != n.shape or h.shape != (512, 512):
            d['check_state'] = 'geometry_error'
        else:
            delta = np.abs(y.astype(np.int16) - n.astype(np.int16))
            changed = np.any(delta > 0, axis=2)
            canonical = canonical_normals[(row['category'], row['target_id'])]
            with Image.open(canonical['source_path']) as original:
                rgb = original.convert('RGB')
                width, height = rgb.size
                scale = min(512 / width, 512 / height)
                rw, rh = max(1, round(width * scale)), max(1, round(height * scale))
                left, top = (512 - rw) // 2, (512 - rh) // 2
                canvas = Image.new('RGB', (512, 512), 'black')
                canvas.paste(rgb.resize((rw, rh), Image.Resampling.LANCZOS), (left, top))
                canonical_array = np.asarray(canvas)
            valid = np.zeros((512, 512), dtype=bool)
            valid[top:top + rh, left:left + rw] = True
            d.update(check_state='checked', source_sha256={k: sha(p) for k, p in paths.items()},
                     canonical_normal_path=canonical['source_path'], canonical_normal_sha256=sha(canonical['source_path']),
                     canonical_source_sha_matches=(sha(canonical['source_path']) == canonical['image_sha256']),
                     sample_normal_matches_canonical512=bool(np.array_equal(n, canonical_array)),
                     mask_pixels=int(h.sum()), outside_changed_pixels=int((changed & ~h).sum()),
                     outside_changed_valid_pixels=int((changed & ~h & valid).sum()),
                     inside_changed_pixels=int((changed & h).sum()),
                     inside_changed_fraction=float(changed[h].mean()),
                     inside_mae_rgb_0_255=float(delta[h].mean()),
                     exact_unchanged_inside=bool(not changed[h].any()))
        results.append(d)
    with (OUT / 'samples.jsonl').open('x', encoding='utf-8') as f:
        for d in results:
            f.write(json.dumps(d, ensure_ascii=False) + '\n')
    checked = [d for d in results if d['check_state'] == 'checked']
    summary = {'state': 'complete', 'started_utc': started, 'completed_utc': now(),
               'fixed_attempts': len(results), 'retained_ineligible': sum(d['check_state'] == 'ineligible_historical_attempt_retained' for d in results),
               'checked_legal_outputs': len(checked), 'geometry_errors': sum(d['check_state'] == 'geometry_error' for d in results),
               'outside_change_violations': [d['attempt_key'] for d in checked if d['outside_changed_pixels']],
               'canonical_normal_mismatch': [d['attempt_key'] for d in checked if not d['sample_normal_matches_canonical512'] or not d['canonical_source_sha_matches']],
               'exact_no_change_inside': [d['attempt_key'] for d in checked if d['exact_unchanged_inside']],
               'semantic_conclusion': 'Unknown; byte changes do not establish defect visibility, requested type, realism or matching label',
               'samples_sha256': sha(OUT / 'samples.jsonl'), 'protocol_sha256': sha(OUT / 'protocol.json')}
    write('summary.json', summary)
    write('status.json', summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
