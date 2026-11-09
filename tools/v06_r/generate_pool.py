"""Frozen matched 400-record pool, durable per-record commits, no candidate selection."""
from __future__ import annotations
import argparse
from dataclasses import asdict
import gc
import json
from pathlib import Path
import re
import uuid
import numpy as np
from PIL import Image
import torch
from cadiff.config import load_config
from cadiff.v06_b.renderer import compose_causal
from cadiff.v06_c.visa_exact_mask_r import load_visa_renderer,CHECKPOINT_SCHEMA
from cadiff.v06_c.can_exact_mask_r import load_can_r_modules
from cadiff.p2_a0.model import P2InfillNoisePredictor
from cadiff.v06_r.downstream import ARMS,digest,read_json,new_json,validate_protocol
from tools.v06_r.render_panel import checked_inputs,metadata_config

def check_identity_manifest(manifest):
    if manifest.get('schema_version')!='v06_r_matched_pool_v1': raise ValueError('unknown pool identity schema')
    rows=manifest['rows']; types=manifest['defect_types']
    if len(rows)!=400 or len({r['identity'] for r in rows})!=400 or len(types)!=4 or 'combined' in types:
        raise ValueError('pool must freeze four types and 400 distinct identities')
    if any(sum(r['defect_type']==t for r in rows)!=100 for t in types): raise ValueError('pool needs 100 per type')
    for row in rows:
        if not re.fullmatch(r'[A-Za-z0-9_-]+',row['identity']): raise ValueError('unsafe identity')
        if row['normal_role']!='amazon_1cls_train' or row['uses_test_mask'] is not False:
            raise ValueError('pool uses non-training assets')
        normal,mask=checked_inputs(row)
        rect=row['valid_rectangle']
        if len(rect)!=4 or any(type(v)!=int for v in rect) or not (0<=rect[0]<rect[2]<=512 and 0<=rect[1]<rect[3]<=512): raise ValueError('invalid valid rectangle')
        valid=np.zeros((512,512),bool); valid[rect[1]:rect[3],rect[0]:rect[2]]=True
        if np.any(mask&~valid): raise ValueError('H mask leaves frozen valid image domain')
        if not row.get('normal_native_path') or not Path(row['normal_native_path']).is_file(): raise ValueError('native training normal missing')
    return manifest

def generate(identities,checkpoints,paths_yaml,protocol,output,*,arm=None,resume=False,device='cuda:0'):
    source=check_identity_manifest(read_json(identities)); validate_protocol(read_json(protocol))
    selected=[arm] if arm else list(ARMS)
    cp={k:Path(v) for k,v in read_json(checkpoints).items()}
    if set(cp)!=set(ARMS): raise ValueError('all three final checkpoints must be declared')
    configs={}; metadata={}
    for a in ARMS: configs[a],metadata[a]=metadata_config(cp[a],a,source['defect_types'])
    if any(asdict(configs[a])!=asdict(configs[ARMS[0]]) for a in ARMS): raise ValueError('R configs differ')
    freeze={'identities_path':str(identities.resolve()),'identities_sha256':digest(identities),'protocol_path':str(protocol.resolve()),'protocol_sha256':digest(protocol),
            'checkpoint_metadata_sha256':{a:digest(cp[a]/'metadata.json') for a in ARMS},'checkpoint_locations_sha256':digest(checkpoints),'ddim_steps':50,'eta':0,'candidate_count':1}
    if not output.exists():
        if resume: raise FileNotFoundError('no pool stage to resume')
        output.mkdir(parents=True,exist_ok=False); new_json(output/'frozen.json',freeze)
    elif read_json(output/'frozen.json')!=freeze: raise ValueError('pool registered identities/config/checkpoints changed')
    paths=load_config(paths_yaml); renderer=None
    for a in selected:
        target=output/a
        if target.exists() and not resume: raise FileExistsError('pool arm exists; explicit resume required')
        if not target.exists(): target.mkdir(exist_ok=False)
        rows=[]; pending=[]
        for row in source['rows']:
            folder=target/row['identity']
            if folder.exists():
                record=read_json(folder/'record.json')
                if record['identity']!=row['identity'] or record['checkpoint_metadata_sha256']!=freeze['checkpoint_metadata_sha256'][a]: raise ValueError('committed row identity changed')
                for key in ('image','mask','raw'):
                    if digest(record[f'{key}_path'])!=record[f'{key}_sha256']: raise ValueError('committed pool file changed')
                rows.append(record)
            else: pending.append(row)
        if pending:
            if renderer is None:
                renderer=load_visa_renderer(checkpoint=cp[a],model_root=paths.model_root,config=configs[a],device=device)
            else:
                _,conditioner,adapter,_=load_can_r_modules(cp[a],config=configs[a],device=torch.device(device),expected_schema=CHECKPOINT_SCHEMA)
                renderer.model=P2InfillNoisePredictor(renderer.components.unet,conditioner,adapter).eval()
                del conditioner,adapter
                gc.collect()
            if renderer.ddim_steps!=50: raise ValueError('DDIM steps differ')
            for row in pending:
                normal,mask=checked_inputs(row)
                raw=renderer(normal.copy(),row['defect_type'],mask.copy(),row['renderer_seed'])
                if raw.shape!=normal.shape or raw.dtype!=np.uint8: raise ValueError('renderer must return RGB uint8 512')
                final=compose_causal(normal,raw,mask)
                if not np.array_equal(final[~mask],normal[~mask]) or not np.array_equal(final[mask],raw[mask]): raise RuntimeError('causal composition violated')
                checked_inputs(row)
                tmp=target/f'.pending-{row["identity"]}-{uuid.uuid4().hex}'
                tmp.mkdir(exist_ok=False)
                Image.fromarray(final).save(tmp/'final.png'); Image.fromarray(raw).save(tmp/'raw.png'); Image.fromarray(mask.astype(np.uint8)*255).save(tmp/'mask.png')
                # Re-read actual files before a directory becomes a completed identity.
                saved=np.array(Image.open(tmp/'final.png').convert('RGB'))
                if not np.array_equal(saved[~mask],normal[~mask]): raise RuntimeError('PNG composition differs')
                folder=target/row['identity']
                record={**row,'role':'generated_train','arm':a,'candidate_count':1,'ddim_steps':50,'eta':0,'outside_rgb_byte_exact':True,
                        'checkpoint_metadata_sha256':freeze['checkpoint_metadata_sha256'][a],
                        'image_path':str(folder/'final.png'),'image_sha256':digest(tmp/'final.png'),
                        'mask_path':str(folder/'mask.png'),'mask_sha256':digest(tmp/'mask.png'),
                        'raw_path':str(folder/'raw.png'),'raw_sha256':digest(tmp/'raw.png')}
                if record['mask_sha256']!=row['mask_sha256']:
                    # The source PNG may have different compression; byte-for-byte frozen labels are copied.
                    import shutil
                    shutil.copyfile(row['mask_path'],tmp/'mask.png'); record['mask_sha256']=digest(tmp/'mask.png')
                new_json(tmp/'record.json',record); tmp.rename(folder); rows.append(record)
                print(json.dumps({'stage':'matched_pool','arm':a,'completed':len(rows),'total':400,'identity':row['identity']}),flush=True)
        by_id={r['identity']:r for r in rows}
        ordered=[by_id[r['identity']] for r in source['rows']]
        payload={'schema_version':'v06_r_generated_pool_v1','arm':a,'identities_sha256':freeze['identities_sha256'],'protocol_sha256':freeze['protocol_sha256'],'count':400,'rows':ordered}
        path=target/'manifest.json'
        if path.exists():
            if read_json(path)!=payload: raise ValueError('completed pool manifest differs')
        else: new_json(path,payload)
    del renderer; gc.collect()
    return {'pool_root':str(output),'completed_arms':selected,'per_arm':400}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('identities','checkpoints','paths-yaml','protocol','output'): p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--arm',choices=ARMS); p.add_argument('--resume',action='store_true'); p.add_argument('--device',default='cuda:0')
    a=p.parse_args()
    print(json.dumps(generate(a.identities,a.checkpoints,a.paths_yaml,a.protocol,a.output,arm=a.arm,resume=a.resume,device=a.device)),flush=True)
if __name__=='__main__': main()
