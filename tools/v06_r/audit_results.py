from collections import Counter
from pathlib import Path
import hashlib
import json
import math

ROOT = Path('/home/dell/2T/lm/new_projects/category_anomaly_diffusion/runs/v06_r/visa/chewinggum/20261012-strong-r')
ARMS = ('latent_control', 'rgb', 'rgb_edge')
TRACE_KEYS = ('step', 'sample_id', 'defect_type', 'timestep', 'context_seed', 'context_sha256')
IDENTITY_KEYS = ('p2_config', 'experiment', 'train', 'manifest_sha256', 'smoke', 'forced_timestep', 'paths_sha256')
SHARED_IDENTITY_KEYS = ('p2_config', 'train', 'manifest_sha256', 'smoke', 'forced_timestep', 'paths_sha256')
EXPECTED_EXPERIMENTS = {
    'latent_control': {'arm': 'latent_control', 'minimum_snr': 0.1, 'rgb_weight': 0.0, 'edge_weight': 0.0, 'save_every': 300, 'schema_version': 'v06_r_loss_ablation_v1'},
    'rgb': {'arm': 'rgb', 'minimum_snr': 0.1, 'rgb_weight': 1.0, 'edge_weight': 0.0, 'save_every': 300, 'schema_version': 'v06_r_loss_ablation_v1'},
    'rgb_edge': {'arm': 'rgb_edge', 'minimum_snr': 0.1, 'rgb_weight': 1.0, 'edge_weight': 0.5, 'save_every': 300, 'schema_version': 'v06_r_loss_ablation_v1'},
}


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def read_jsonl(path):
    rows = []
    with Path(path).open(encoding='utf-8') as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise AssertionError(f'invalid JSON at {path}:{line_number}: {exc}') from exc
    return rows


def finite(value):
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(finite(item) for item in value.values())
    if isinstance(value, list):
        return all(finite(item) for item in value)
    return True


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def declared_segments(arm, state):
    recovery = state.get('recovery')
    if not recovery:
        return [{'output': state['output'], 'first_step': 1, 'last_step': 1200,
                 'name': 'primary', 'resume_from': None}]
    segments = recovery.get('segments')
    assert isinstance(segments, list) and len(segments) >= 2, (arm, 'missing recovery segments')
    assert Path(recovery['original_output']) == Path(segments[0]['output']), (arm, 'original output mismatch')
    assert Path(state['output']) == Path(segments[-1]['output']), (arm, 'group output is not final recovery output')
    assert recovery['resume_step'] == segments[0]['last_step'], (arm, 'resume step differs from prior segment')
    assert segments[1]['first_step'] == recovery['resume_step'] + 1, (arm, 'recovery segment does not start after checkpoint')
    normalized = []
    for index, item in enumerate(segments):
        segment = dict(item)
        assert segment['first_step'] <= segment['last_step'], (arm, 'empty segment')
        if index:
            assert segment['first_step'] == segments[index - 1]['last_step'] + 1, (arm, 'segment step ranges are not contiguous')
        segment.setdefault('name', f'segment_{index + 1}')
        segment.setdefault('resume_from', recovery.get('resume_from') if index else None)
        normalized.append(segment)
    return normalized


def load_segment(arm, segment):
    output = Path(segment['output'])
    manifest = read_json(output / 'run_manifest.json')
    raw_rows = read_jsonl(output / 'training.jsonl')
    first, last = segment['first_step'], segment['last_step']
    used = [row for row in raw_rows if first <= row.get('step', -1) <= last]
    excluded = [row for row in raw_rows if not first <= row.get('step', -1) <= last]
    expected_steps = list(range(first, last + 1))
    assert [row.get('step') for row in used] == expected_steps, (arm, segment.get('name'), 'step sequence mismatch')
    assert all(row.get('finite') is True and finite(row) for row in used), (arm, segment.get('name'), 'non-finite training record')
    assert all(row.get('applied_optimizer_steps') == row.get('step') and row.get('grad_l2', 0) > 0 for row in used), (arm, segment.get('name'), 'optimizer-step integrity failure')
    info = {'name': segment.get('name'), 'output': str(output), 'first_step': first, 'last_step': last,
            'raw_log_records': len(raw_rows), 'used_records': len(used), 'excluded_uncheckpointed_records': len(excluded),
            'resume_from': segment.get('resume_from')}
    return used, manifest, info, excluded


def main():
    inputs = read_json(ROOT / 'inputs.json')
    tree = Path(inputs['worktree'])
    for relative, digest in inputs['formal_a_h_and_composition_sha256'].items():
        assert sha256_file(tree / relative) == digest, relative

    group = read_json(ROOT / 'training_group.json')
    assert group.get('state') == 'complete', f"training group is not complete: {group.get('state')}"
    summaries, manifests, traces, segments_info = {}, {}, {}, {}
    recovery_audit = {}
    for arm in ARMS:
        state = group['arms'][arm]
        segments = declared_segments(arm, state)
        stitched, segment_infos, manifests_for_arm = [], [], []
        excluded_by_segment = []
        for segment in segments:
            rows, manifest, info, excluded = load_segment(arm, segment)
            stitched.extend(rows)
            segment_infos.append(info)
            manifests_for_arm.append(manifest)
            excluded_by_segment.append(excluded)
        assert len(stitched) == 1200 and [r['step'] for r in stitched] == list(range(1, 1201)), (arm, 'stitched trace is not 1200 steps')
        assert all(row['applied_optimizer_steps'] == row['step'] and row['finite'] for row in stitched)

        baseline_manifest = manifests_for_arm[0]
        for manifest in manifests_for_arm[1:]:
            for field in IDENTITY_KEYS:
                assert manifest[field] == baseline_manifest[field], (arm, field, 'segment identity mismatch')

        if arm == 'rgb_edge' and len(segments) > 1:
            recovery = state['recovery']
            resume_path = Path(recovery['resume_from'])
            resume_meta = read_json(resume_path / 'metadata.json')
            recovery_manifest = manifests_for_arm[-1]
            assert recovery_manifest.get('resume_from') == str(resume_path), 'recovered run manifest resume_from mismatch'
            assert resume_meta['completed_steps'] == recovery['resume_step'], 'resume checkpoint step mismatch'
            expected_identity = {key: recovery_manifest[key] for key in IDENTITY_KEYS}
            assert resume_meta['identity'] == expected_identity, 'resume checkpoint training identity mismatch'
            for filename, record in resume_meta['files'].items():
                assert sha256_file(resume_path / filename) == record['sha256'], ('resume checkpoint hash mismatch', filename)
            old_uncheckpointed = {row['step']: row for row in excluded_by_segment[0]}
            resumed_rows = {row['step']: row for row in stitched if recovery['resume_step'] < row['step'] <= recovery['original_last_logged_step']}
            overlap_steps = list(range(recovery['resume_step'] + 1, recovery['original_last_logged_step'] + 1))
            assert all(step in old_uncheckpointed and step in resumed_rows for step in overlap_steps), 'missing old/recovered overlap rows'
            assert all(old_uncheckpointed[step] == resumed_rows[step] for step in overlap_steps), 'recovery overlap rows differ from discarded old segment'
            recovery_audit = {'resume_step': recovery['resume_step'], 'resume_from': str(resume_path),
                              'old_last_logged_step': recovery['original_last_logged_step'],
                              'overlap_steps_verified': len(overlap_steps), 'overlap_records_identical': True}

        traces[arm] = stitched
        manifests[arm] = baseline_manifest
        segments_info[arm] = segment_infos
        final_output = Path(state['output'])
        summary = read_json(final_output / 'summary.json')
        assert summary['complete'] and summary['completed_steps'] == 1200 and not summary['smoke'], (arm, 'incomplete summary')
        assert summary['no_test_loading'] and summary['frozen_vae_gradient_free'] and summary['frozen_unet_unchanged_by_optimizer']
        assert summary['vae_decode_dtype'] == 'torch.float32' and summary['fewshot_train'] == 18
        checkpoint = Path(summary['checkpoint'])
        metadata = read_json(checkpoint / 'metadata.json')
        assert metadata['completed_steps'] == 1200 and metadata['experiment']['arm'] == arm
        assert metadata['identity'] == {key: baseline_manifest[key] for key in IDENTITY_KEYS}
        for filename, record in metadata['files'].items():
            assert sha256_file(checkpoint / filename) == record['sha256'], (arm, filename, 'final checkpoint hash mismatch')
        summaries[arm] = summary

    baseline = manifests[ARMS[0]]
    for arm in ARMS:
        assert manifests[arm]['experiment'] == EXPECTED_EXPERIMENTS[arm], (arm, 'experiment config mismatch')
    for arm in ARMS[1:]:
        for field in SHARED_IDENTITY_KEYS:
            assert manifests[arm][field] == baseline[field], (arm, field)
        for step, (left, right) in enumerate(zip(traces[ARMS[0]], traces[arm]), 1):
            for field in TRACE_KEYS:
                assert left[field] == right[field], (arm, step, field, left[field], right[field])
    train = baseline['train']
    assert len(train) == 18 and all(row.get('role') == 'fewshot_train' for row in train)
    assert not any('test' in str(row).lower() or 'open_dev' in str(row).lower() for row in train)

    result = {'schema_version': 'v06_r_training_audit_v2', 'all_three_complete': True,
              'steps_per_arm': 1200, 'paired_random_trace_exact': True,
              'formal_a_h_and_composition_unchanged': True, 'training_samples': 18,
              'uses_heldout_anomalies': False, 'training_seed_count': 1,
              'visual_success': None, 'arms': {}}
    for arm in ARMS:
        data = traces[arm]
        by_type = {}
        for label in inputs['defect_types']:
            rows = [row for row in data if row['defect_type'] == label]
            active = [row for row in rows if row['decode_count'] > 0]
            by_type[label] = {'updates': len(rows), 'appearance_active_updates': len(active),
                              'active_updates_without_edge_interior': sum(row['edge_pixels'] == 0 for row in active),
                              'gated_decode_count': sum(row['decode_count'] for row in rows),
                              'edge_valid_empty_updates': sum(row['edge_pixels'] == 0 for row in rows)}
        result['arms'][arm] = {'checkpoint': summaries[arm]['checkpoint'],
                               'gpu_peak_allocated_gib': summaries[arm]['gpu_peak_allocated_bytes'] / 1024**3,
                               'amp_overflow_backoffs': sum(row['amp_overflow_skips'] for row in data),
                               'all_updates_finite': True, 'type_counts': dict(Counter(row['defect_type'] for row in data)),
                               'appearance_coverage': by_type, 'segments': segments_info[arm],
                               'final_training_loss': data[-1]['total']}
    result['recovery'] = recovery_audit
    audit_path = ROOT / 'training_audit.json'
    with audit_path.open('x', encoding='utf-8') as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
        handle.write('\n')
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
