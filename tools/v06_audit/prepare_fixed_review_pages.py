"""Contact sheets of existing evidence for research precheck, not a report."""
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path('/home/dell/2T/lm/new_projects/category_anomaly_diffusion')
RUN = ROOT / '.worktrees/visa-r-baseline-20261012/runs/v06_audit/visa/20261015-split-mask-coverage'
OUT = RUN / 'existing_r_output_check/fixed_visual_precheck'
SIZE = 224


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_rgb(path):
    with Image.open(path) as im:
        return im.convert('RGB').copy()


def bbox(mask, pad=32):
    yy, xx = np.where(mask)
    if not len(xx):
        return (0, 0, 512, 512)
    return max(0, int(xx.min()) - pad), max(0, int(yy.min()) - pad), min(512, int(xx.max()) + pad + 1), min(512, int(yy.max()) + pad + 1)


def fit(image):
    image = image.copy()
    image.thumbnail((SIZE, SIZE), Image.Resampling.NEAREST)
    tile = Image.new('RGB', (SIZE, SIZE), '#eeeeee')
    tile.paste(image, ((SIZE - image.width) // 2, (SIZE - image.height) // 2))
    return tile


def main():
    OUT.mkdir(exist_ok=False)
    samples = [json.loads(l) for l in (RUN / 'existing_r_output_check/samples.jsonl').open()]
    wanted = {s['attempt_key']: s for s in samples}
    first = {}
    for line in (RUN / 'attempt_index.jsonl').open():
        row = json.loads(line)
        key = (row['category'], row['type_slug'])
        if row['attempt_key'] in wanted and (key not in first or row['index'] < first[key]['index']):
            first[key] = row
    chosen = [first[k] for k in sorted(first)]
    assert len(chosen) == 58
    ids = {}
    for line in (RUN / 'identity_contract.jsonl').open():
        row = json.loads(line)
        if row['role'] == 'anomaly_support':
            ids[(row['category'], row['source_id'])] = row
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 15)
    manifest = {'selection': 'first index per supported type from the already locked 116 slots; all 58 including failures, no substitution',
                'kind': 'research inspection cache, not teacher report or human evaluation',
                'display': 'native512 full panels and exact H-bbox+32 crops; nearest display scaling; reference uses real training source GT-bbox+32',
                'columns': ['real support crop', 'normal full', 'generated full', 'H full', 'normal H-ROI', 'generated H-ROI'],
                'human_semantic_status': 'unreviewed', 'pages': []}
    (OUT / 'selection_before_render.json').write_text(json.dumps({'attempt_keys': [r['attempt_key'] for r in chosen], **{k:v for k,v in manifest.items() if k != 'pages'}}, indent=2), encoding='utf-8')
    for start in range(0, len(chosen), 6):
        chunk = chosen[start:start + 6]
        page = Image.new('RGB', (SIZE * 6, 60 + len(chunk) * (SIZE + 48)), 'white')
        draw = ImageDraw.Draw(page)
        draw.text((5, 2), 'Fixed historical outputs: AI precheck only. Failed rows retained. No new generation.', font=font, fill='black')
        for c, label in enumerate(manifest['columns']):
            draw.text((c * SIZE + 5, 30), label, font=font, fill='black')
        evidence = []
        for ri, row in enumerate(chunk):
            top = 60 + ri * (SIZE + 48)
            flag = 'FAILED: no published output' if row['record_failed'] else ('numeric legal; semantic unknown' if row['numeric_success'] else 'historical H below current gate')
            draw.text((5, top), f"{row['category']} / {row['type_slug']} | index={row['index']} | {flag}", font=font, fill='#a00000' if not row['numeric_success'] else 'black')
            draw.text((5, top + 20), row['job_id'], font=font, fill='#555555')
            folder = Path(row['run_path']) / 'samples' / row['job_id']
            paths = {x: folder / f'{x}.png' for x in ('normal', 'final_y', 'm_final')}
            normal = load_rgb(paths['normal'])
            generated = load_rgb(paths['final_y'])
            with Image.open(paths['m_final']) as im:
                mask = np.asarray(im.convert('L')) > 0
            box = bbox(mask)
            reference = Image.new('RGB', (SIZE, SIZE), '#dddddd')
            src = ids.get((row['category'], row.get('source_id')))
            if src:
                rgb = load_rgb(Path(src['source_path']))
                with Image.open(ROOT / 'data/VisA' / src['mask_relpath']) as im:
                    m = Image.fromarray((np.asarray(im.convert('L')) > 0).astype('uint8') * 255).resize(rgb.size, Image.Resampling.NEAREST)
                rw, rh = src['valid_rectangle512'][2] - src['valid_rectangle512'][0], src['valid_rectangle512'][3] - src['valid_rectangle512'][1]
                pos = tuple(src['valid_rectangle512'][:2])
                can = Image.new('RGB', (512, 512), 'black'); can.paste(rgb.resize((rw, rh), Image.Resampling.LANCZOS), pos)
                mc = Image.new('L', (512, 512), 0); mc.paste(m.resize((rw, rh), Image.Resampling.NEAREST), pos)
                reference = can.crop(bbox(np.asarray(mc) > 0))
            mask_image = Image.fromarray(mask.astype('uint8') * 255).convert('RGB')
            tiles = [reference, normal, generated, mask_image, normal.crop(box), generated.crop(box)]
            for ci, tile in enumerate(tiles):
                page.paste(fit(tile), (ci * SIZE, top + 48))
            evidence.append({'attempt_key': row['attempt_key'], 'numeric_success': row['numeric_success'], 'failed': row['record_failed'], 'h_roi': list(box),
                             'source_paths': {k:str(v) for k,v in paths.items()}, 'source_sha256': {k:digest(v) for k,v in paths.items()},
                             'real_support_path': src['source_path'] if src else None})
        name = f'page-{start // 6 + 1:02d}.png'; page.save(OUT / name)
        manifest['pages'].append({'file': name, 'sha256': digest(OUT / name), 'samples': evidence})
    (OUT / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'pages': len(manifest['pages']), 'samples': len(chosen), 'output': str(OUT), 'manifest_sha256': digest(OUT / 'manifest.json')}))


if __name__ == '__main__':
    main()
