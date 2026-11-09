"""Read-only source audit; writes only a new supplemental result directory."""
import collections
import concurrent.futures
import datetime
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path('/home/dell/2T/lm/new_projects/category_anomaly_diffusion')
WORK = ROOT / '.worktrees/visa-r-baseline-20261012'
RUN = WORK / 'runs/v06_audit/visa/20261015-split-mask-coverage'
OUT = RUN / 'supplemental_identity_review'


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(name, value):
    path = OUT / name
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temp.replace(path)


def decode(row):
    with Image.open(row['source_path']) as im:
        a = np.asarray(im.convert('RGB'))
        digest = hashlib.sha256(str(a.shape).encode() + a.tobytes()).hexdigest()
    return dict(row, decoded_rgb_sha256=digest)


def conflict(roles):
    return (('normal_train' in roles and 'normal_test' in roles)
            or ('anomaly_support' in roles and 'anomaly_heldout' in roles)
            or (any(x.startswith('normal') for x in roles)
                and any(x.startswith('anomaly') for x in roles)))


def main():
    OUT.mkdir(exist_ok=False)
    started = now()
    source = RUN / 'identity_contract.jsonl'
    rows = [json.loads(line) for line in source.read_text().splitlines() if line.strip()]
    write('protocol.json', {
        'registered_utc': started, 'source': str(source), 'source_sha256': sha(source),
        'purpose': 'Complete decoded-normal role-duplicate check and inspect historical R training manifests',
        'gpu': False, 'workers': 2, 'read_only_sources': True,
        'decoded_digest': 'SHA256(str(numpy RGB shape).encode + RGB uint8 bytes)',
        'renderer_scope': 'Named historical R metadata/manifests; generation-to-checkpoint lineage is not proven by matching directory names',
        'does_not_test': ['near duplicates', 'semantic mask validity', 'new generation performance'],
    })
    write('status.json', {'state': 'running', 'started_utc': started, 'phase': 'decode_normals'})
    normals = [r for r in rows if r['role'].startswith('normal')]
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        decoded = list(pool.map(decode, normals))
    normal_map = {r['image_relpath']: r for r in decoded}
    complete = [normal_map.get(r['image_relpath'], r) for r in rows]
    with (OUT / 'normal_decoded_sha256.jsonl').open('x', encoding='utf-8') as f:
        for r in decoded:
            f.write(json.dumps({k: r[k] for k in ('category', 'image_relpath', 'role', 'image_sha256', 'decoded_rgb_sha256')}) + '\n')
    modes = {}
    for mode in ('magic', 'seas'):
        groups = collections.defaultdict(list)
        for r in complete:
            role = r.get('seas_role', r['role']) if mode == 'seas' else r['role']
            groups[r['decoded_rgb_sha256']].append({'image_relpath': r['image_relpath'], 'role': role})
        duplicates = [v for v in groups.values() if len(v) > 1]
        conflicts = [v for v in duplicates if conflict({x['role'] for x in v})]
        modes[mode] = {'duplicate_groups': duplicates, 'cross_role_conflicts': conflicts,
                       'cross_role_conflict_count': len(conflicts)}
    write('decoded_role_checks.json', {'scope': 'all 10821 official images, exact decoded RGB equality', 'normal_count': len(decoded), 'total_count': len(complete), 'protocols': modes})
    write('status.json', {'state': 'running', 'started_utc': started, 'phase': 'historical_renderer_manifests'})
    identities = {r['image_relpath']: r for r in rows}
    r_scopes = []
    for cat in sorted({r['category'] for r in rows}):
        tag = '20260922-seas' if cat in ('candle', 'capsules', 'cashew', 'chewinggum') else '20260927-magic'
        rdir = ROOT / f'runs/v06_b/visa/{cat}/exact-mask-r/seed-{tag}'
        manifest = ROOT / f'runs/v06_b/visa/{cat}/splits/{tag}/manifest.jsonl'
        paths = [rdir / 'summary.json', rdir / 'checkpoint/metadata.json', manifest]
        record = {'category': cat, 'tag': tag, 'generation_checkpoint_link': 'not independently recorded in generation records; metadata comparison only',
                  'evidence': [{'path': str(p), 'sha256': sha(p)} for p in paths]}
        summary = json.loads(paths[0].read_text())
        metadata = json.loads(paths[1].read_text())
        registered = [json.loads(l) for l in manifest.read_text().splitlines() if l.strip()]
        counts = collections.Counter()
        violations = []
        combined_train = []
        normal_train = []
        for r in registered:
            current = identities.get(r['image_relpath'])
            if current is None:
                violations.append({'kind': 'unknown_source', 'path': r['image_relpath']})
                continue
            if r.get('image_sha256') != current['image_sha256']:
                violations.append({'kind': 'image_sha_mismatch', 'path': r['image_relpath']})
            if r.get('mask_relpath') and r.get('mask_sha256') != current.get('native_mask', {}).get('file_sha256'):
                violations.append({'kind': 'mask_sha_mismatch', 'path': r['image_relpath']})
            for role in r.get('roles', []):
                counts[role] += 1
            if 'fewshot_train' in r.get('roles', []):
                if current['role'] == 'anomaly_heldout':
                    violations.append({'kind': 'registered_training_uses_magic_heldout', 'path': r['image_relpath']})
                if r['defect_type'] == 'combined':
                    combined_train.append(r['image_relpath'])
            if any('normal' in role for role in r.get('roles', [])):
                normal_train.append(r['image_relpath'])
                if current['role'] == 'normal_test':
                    violations.append({'kind': 'registered_normal_train_uses_test', 'path': r['image_relpath']})
        record.update(split_id=metadata.get('split_id'), steps=summary.get('steps'),
                      defect_types=metadata['config']['defect_types'], role_counts=dict(counts),
                      combined_training_count=len(combined_train), combined_training_ids=combined_train,
                      violations=violations, pure_single_label_training_contract=(not combined_train and not violations))
        r_scopes.append(record)
    write('historical_renderer_scope.json', {'scope': 'registered historical R training identity manifests and checkpoint metadata; no model weight loading', 'categories': r_scopes,
        'interpretation': 'Combined training outside the MAGIC table is extra training scope, not automatically support-test leakage. Historical pools are not a uniformly clean single-label training comparison.',
        'limits': ['does not prove exact checkpoint lineage for historical generation', 'does not replay optimizer execution or test inference']})
    result = {'state': 'complete', 'started_utc': started, 'completed_utc': now(),
              'decoded_normal_count': len(decoded), 'all_image_count': len(complete),
              'decoded_cross_role_conflicts': {k: v['cross_role_conflict_count'] for k, v in modes.items()},
              'historical_renderer_categories': len(r_scopes),
              'registered_manifest_violations': sum(len(r['violations']) for r in r_scopes),
              'extra_combined_training': {r['category']: r['combined_training_count'] for r in r_scopes if r['combined_training_count']},
              'output_sha256': {p.name: sha(p) for p in OUT.iterdir() if p.suffix in ('.json', '.jsonl') and p.name != 'status.json'}}
    write('status.json', result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
