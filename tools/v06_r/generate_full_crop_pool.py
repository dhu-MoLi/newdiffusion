"""Fixed two-arm pool, original H labels, float Crop32 inverse and physical composition."""
import argparse
from dataclasses import asdict
import gc
import json
from pathlib import Path
import shutil
import uuid
import numpy as np
from PIL import Image
import torch
from cadiff.config import load_config
from cadiff.v06_r.full_crop_training import VIEWS,validate_protocol
from cadiff.v06_r.full_crop_inference import checkpoint_config,load_float_renderer,render_physical
from cadiff.v06_r.downstream import read_json,new_json,digest
from tools.v06_r.render_panel import checked_inputs
from tools.v06_r.generate_pool import check_identity_manifest

def generate(*,identities,checkpoints,paths_yaml,protocol,output,view=None,resume=False,device='cuda:0',smoke_limit=None):
    validate_protocol(protocol); frozen=check_identity_manifest(read_json(identities)); locations={k:Path(v) for k,v in read_json(checkpoints).items()}
    if set(locations)!=set(VIEWS): raise ValueError('new checkpoints must map exactly full/crop32')
    checked={v:checkpoint_config(locations[v],v,allow_smoke=smoke_limit is not None) for v in VIEWS}
    if any(data[1]['identity'].get('protocol_sha256')!=digest(protocol) for data in checked.values()): raise ValueError('checkpoint registered protocol differs from generation')
    configs={v:data[0] for v,data in checked.items()}
    if asdict(configs['full'])!=asdict(configs['crop32']) or list(configs['full'].defect_types)!=frozen['defect_types']: raise ValueError('two-arm config/vocabulary differ')
    if smoke_limit is not None and (type(smoke_limit)!=int or not 1<=smoke_limit<=2): raise ValueError('generation smoke at most2records per arm')
    registration={'identities_sha256':digest(identities),'protocol_sha256':digest(protocol),'checkpoint_locations_sha256':digest(checkpoints),
                  'checkpoint_metadata_sha256':{v:digest(locations[v]/'metadata.json') for v in VIEWS},'smoke_limit':smoke_limit,
                  'vae_load_dtype':'float32_from_original_disk_weights','latent_dtype':'float16','decode_dtype':'float32','physical_H_unchanged':True}
    if not output.exists():
        if resume: raise FileNotFoundError('no pool to resume')
        output.mkdir(parents=True,exist_ok=False); new_json(output/'frozen.json',registration)
    elif read_json(output/'frozen.json')!=registration: raise ValueError('pool registration changed')
    paths=load_config(paths_yaml)
    selected=[view] if view else list(VIEWS)
    source=frozen['rows'][:smoke_limit] if smoke_limit else frozen['rows']
    for arm in selected:
        target=output/arm
        if target.exists() and not resume: raise FileExistsError('arm exists; explicit resume required')
        if not target.exists(): target.mkdir()
        records=[]; pending=[]
        for row in source:
            folder=target/row['identity']
            if folder.exists():
                record=read_json(folder/'record.json')
                if record['checkpoint_metadata_sha256']!=registration['checkpoint_metadata_sha256'][arm]: raise ValueError('row checkpoint changed')
                for kind in ('image','mask','raw'):
                    if digest(record[f'{kind}_path'])!=record[f'{kind}_sha256']: raise ValueError('committed PNG changed')
                records.append(record)
            else: pending.append(row)
        renderer=load_float_renderer(locations[arm],view=arm,model_root=paths.model_root,device=device,allow_smoke=smoke_limit is not None) if pending else None
        if torch.device(device).type=='cuda': torch.cuda.reset_peak_memory_stats(device)
        for row in pending:
            normal,mask=checked_inputs(row)
            raw,final,geometry=render_physical(renderer,normal,mask,row,arm)
            checked_inputs(row)
            folder=target/row['identity']; tmp=target/f'.pending-{uuid.uuid4().hex}'; tmp.mkdir()
            Image.fromarray(raw).save(tmp/'raw.png'); Image.fromarray(final).save(tmp/'final.png'); shutil.copyfile(row['mask_path'],tmp/'mask.png')
            if not np.array_equal(np.array(Image.open(tmp/'final.png'))[~mask],normal[~mask]): raise ValueError('PNG outside H differs')
            record={**row,'arm':arm,'role':'generated_train','candidate_count':1,'ddim_steps':50,'eta':0,
                    'checkpoint_metadata_sha256':registration['checkpoint_metadata_sha256'][arm],'outside_rgb_byte_exact':True,
                    'physical_H_unchanged':True,'crop_geometry':asdict(geometry) if geometry else None,
                    'image_path':str(folder/'final.png'),'image_sha256':digest(tmp/'final.png'),
                    'mask_path':str(folder/'mask.png'),'mask_sha256':digest(tmp/'mask.png'),
                    'raw_path':str(folder/'raw.png'),'raw_sha256':digest(tmp/'raw.png')}
            new_json(tmp/'record.json',record); tmp.rename(folder); records.append(record)
            print(json.dumps({'stage':'full_crop_pool','arm':arm,'completed':len(records),'total':len(source),'identity':row['identity']}),flush=True)
        by_id={r['identity']:r for r in records}; ordered=[by_id[r['identity']] for r in source]
        payload={'schema_version':'v06_r_full_crop_pool_v1','arm':arm,'count':len(ordered),'identities_sha256':registration['identities_sha256'],
                 'protocol_sha256':registration['protocol_sha256'],'engineering_smoke':smoke_limit is not None,'rows':ordered}
        final=target/'manifest.json'
        if final.exists():
            if read_json(final)!=payload: raise ValueError('manifest changed')
        else: new_json(final,payload)
        del renderer; gc.collect()
        if torch.device(device).type=='cuda': torch.cuda.empty_cache()
    return {'pool_root':str(output),'completed_arms':selected,'per_arm':len(source),'engineering_smoke':smoke_limit is not None}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('identities','checkpoints','paths-yaml','protocol','output'): p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--view',choices=VIEWS); p.add_argument('--resume',action='store_true'); p.add_argument('--device',default='cuda:0'); p.add_argument('--smoke-limit',type=int,choices=(1,2))
    a=p.parse_args(); result=generate(identities=a.identities,checkpoints=a.checkpoints,paths_yaml=a.paths_yaml,protocol=a.protocol,output=a.output,view=a.view,resume=a.resume,device=a.device,smoke_limit=a.smoke_limit)
    print(json.dumps(result),flush=True)
if __name__=='__main__': main()
