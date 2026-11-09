"""New Full/Crop32 labels with frozen old detector plans and Real-FS references."""
from __future__ import annotations
from pathlib import Path
import json,os,shutil
import numpy as np
import torch
from torch.utils.data import DataLoader
from cadiff.v06_r.downstream import (
    SEEDS,EPOCHS,LENGTH,BS,DIM,digest,read_json,new_json,atomic_json,
    validate_assets,validate_gen,PlannedDataset,kit_imports,seed_everything,
    save_resume,load_resume,claim_run_directory,evaluate_final,
)
from cadiff.v06_r.full_crop_training import validate_protocol as registered_protocol
ARMS=('full','crop32')
CONDITIONS=tuple(f'{kind}_{arm}' for kind in ('real_gen','gen_only') for arm in ARMS)

def validate_protocol(data):
    from cadiff.v06_r.full_crop_training import PARAMETERS
    if data.get('schema_version')!='v06_r_full_crop32_protocol_v1' or data.get('parameters')!=PARAMETERS:
        raise ValueError('registered Full/Crop32 downstream protocol differs')
    return data

def prepare(*,old_prepared,identities,protocol,pool_root,output):
    registered_protocol(protocol)
    if output.exists() or output.is_symlink(): raise FileExistsError('refuses to overwrite Full/Crop32 evaluation')
    old=read_json(old_prepared/'frozen.json'); validate_assets(old['assets'])
    for path,sha in old['source_sha256'].items():
        if digest(path)!=sha: raise ValueError('historical detector source changed')
    if digest(old['assets_path'])!=old['assets_sha256']: raise ValueError('old role manifest changed')
    if digest(identities)!=old['identities_sha256']: raise ValueError('must reuse exact old400 identities')
    idata=read_json(identities)
    plan_shas=read_json(old_prepared/'plan_sha256.json')
    references={}
    for seed in SEEDS:
        folder=old_prepared/'real_fs'/f'seed-{seed}'; results=read_json(folder/'results.json')
        checkpoint=folder/'checkpoint-final.pt'
        if results['actual_optimizer_steps']!=6400 or results['test_forward_count']!=89 or results['checkpoint_rule']!='last_epoch_only' or digest(checkpoint)!=results['checkpoint_sha256']:
            raise ValueError('old Real-FS is not same fixed-final detector contract')
        if digest(old_prepared/f'plan-{seed}.npz')!=plan_shas[str(seed)]: raise ValueError('paired detector plan changed')
        references[str(seed)]={'results_path':str(folder/'results.json'),'results_sha256':digest(folder/'results.json'),'checkpoint_path':str(checkpoint),'checkpoint_sha256':digest(checkpoint)}
    sources=dict(old['source_sha256']); sources[str(Path(__file__).resolve())]=digest(__file__)
    root=Path(__file__).resolve().parents[3]
    for path in ('src/cadiff/v06_r/full_crop_training.py','tools/v06_r/run_full_crop_downstream.py'):
        sources[str(root/path)]=digest(root/path)
    frozen={**old,'schema_version':'v06_r_full_crop_downstream_frozen_v1','protocol':read_json(protocol),'protocol_path':str(protocol.resolve()),'protocol_sha256':digest(protocol),
            'identities':idata,'identities_path':str(identities.resolve()),'identities_sha256':digest(identities),'source_sha256':sources,
            'pools':{a:{'path':str((pool_root/a/'manifest.json').resolve())} for a in ARMS},'conditions':list(CONDITIONS),
            'condition_aliases':{c:c for c in CONDITIONS},'old_real_fs_references':references,'old_prepared':str(old_prepared.resolve()),'old_frozen_sha256':digest(old_prepared/'frozen.json')}
    output.mkdir(parents=True,exist_ok=False); new_json(output/'frozen.json',frozen)
    for seed in SEEDS: shutil.copyfile(old_prepared/f'plan-{seed}.npz',output/f'plan-{seed}.npz')
    new_json(output/'plan_sha256.json',plan_shas)
    new_json(output/'real_fs_references.json',references)
    return {'prepared':str(output),'conditions':list(CONDITIONS),'seeds':list(SEEDS),'real_fs':'locked historical references only'}

def validate_pool_payload(payload,arm,frozen):
    if payload.get('arm')!=arm or payload.get('identities_sha256')!=frozen['identities_sha256'] or payload.get('protocol_sha256')!=frozen['protocol_sha256'] or payload.get('engineering_smoke') is not False:
        raise ValueError('pool arm/identity/protocol/production contract differs')

def run(output,condition,seed,*,device='cuda:0',resume=False,stop_after_epoch=None):
    if condition not in CONDITIONS or seed not in SEEDS: raise ValueError('unregistered condition/seed')
    frozen=read_json(output/'frozen.json')
    for record in frozen['old_real_fs_references'].values():
        if digest(record['results_path'])!=record['results_sha256'] or digest(record['checkpoint_path'])!=record['checkpoint_sha256']:
            raise ValueError('locked Real-FS reference changed')
    for path,sha in frozen['source_sha256'].items():
        if digest(path)!=sha: raise ValueError('frozen detector/source changed')
    if digest(frozen['assets_path'])!=frozen['assets_sha256']: raise ValueError('frozen assets manifest changed')
    if digest(frozen['protocol_path'])!=frozen['protocol_sha256'] or digest(frozen['identities_path'])!=frozen['identities_sha256']:
        raise ValueError('registered protocol or identities changed')
    validate_protocol(frozen['protocol'])
    assets=validate_assets(frozen['assets'])
    plans=read_json(output/'plan_sha256.json'); pp=output/f'plan-{seed}.npz'
    if digest(pp)!=plans[str(seed)]: raise ValueError('paired sampling plan changed')
    with np.load(pp,allow_pickle=False) as z: plan={k:z[k].copy() for k in z.files}
    generated=[]
    if condition!='__unused_real_fs__':
        arm=next(a for a in ARMS if condition.endswith('_'+a))
        record=frozen['pools'][arm]
        payload=read_json(record['path'])
        validate_pool_payload(payload,arm,frozen)
        if payload.get('arm')!=arm or payload.get('identities_sha256')!=frozen['identities_sha256']:
            raise ValueError('pool arm/identity SHA differs')
        generated=validate_gen(payload['rows'],assets)
        shared=('identity','defect_type','normal_sha256','mask_sha256','renderer_seed','valid_rectangle')
        if [[r[k] for k in shared] for r in generated]!=[[r[k] for k in shared] for r in frozen['identities']['rows']]:
            raise ValueError('pool outputs differ from registered identities')
    target=output/condition/f'seed-{seed}'
    epoch_zero_restart=claim_run_directory(target,resume)
    if (target/'results.json').exists(): return read_json(target/'results.json')
    Model,Loss,weights_init,calculate_pro=kit_imports(frozen['eval_kit'])
    seed_everything(seed)
    model=Model(in_channels=3,out_channels=2).to(device=device,dtype=torch.float32)
    model.apply(weights_init)
    optimizer=torch.optim.Adam(model.parameters(),lr=1e-4)
    scheduler=torch.optim.lr_scheduler.MultiStepLR(optimizer,[160,180],gamma=.2)
    loss_fn=Loss()
    contract={'frozen_sha256':digest(output/'frozen.json'),'plan_sha256':plans[str(seed)],'condition':condition,'seed':seed,
              'pool_manifest_sha256':digest(record['path']) if condition!='__unused_real_fs__' else None}
    start,steps=0,0
    if resume and not epoch_zero_restart:
        start,steps=load_resume(target/'resume.pt',model,optimizer,scheduler,contract,device)
    if epoch_zero_restart:
        atomic_json(target/'epoch_zero_restart.json',{'reason':'explicit resume before first committed epoch','contract':contract,'test_calls':0})
    for epoch in range(start,EPOCHS):
        model.train()
        loader=DataLoader(PlannedDataset(assets,generated,plan,condition,epoch),batch_size=BS,shuffle=False,num_workers=0)
        total=0.
        for batch in loader:
            optimizer.zero_grad()
            loss=loss_fn(torch.softmax(model(batch['image'].to(device)),dim=1),batch['mask'].to(device))
            if not torch.isfinite(loss): raise RuntimeError('nonfinite detector loss')
            loss.backward()
            if not torch.stack([torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None]).all():
                raise RuntimeError('nonfinite detector gradients')
            optimizer.step(); steps+=1; total+=float(loss.detach())
        scheduler.step()
        save_resume(target/'resume.pt',model,optimizer,scheduler,epoch+1,steps,contract)
        progress={'condition':condition,'seed':seed,'completed_epoch':epoch+1,'actual_optimizer_steps':steps,'mean_batch_loss':total/len(loader),'test_calls':0}
        atomic_json(target/'progress.json',progress)
        print(json.dumps(progress),flush=True)
        if stop_after_epoch is not None and epoch+1>=stop_after_epoch and epoch+1<EPOCHS:
            return {'stopped_at_epoch':epoch+1,'actual_optimizer_steps':steps,'test_calls':0}
    if steps!=6400: raise RuntimeError('final actual optimizer budget differs')
    final=target/'checkpoint-final.pt'
    if not final.exists():
        tmp=Path(str(final)+'.tmp'); torch.save({'state_dict':model.state_dict(),'contract':contract,'epoch':EPOCHS,'steps':steps},tmp); os.replace(tmp,final)
    saved=torch.load(final,map_location=device,weights_only=False)
    if saved['contract']!=contract or saved['epoch']!=EPOCHS or saved['steps']!=6400: raise ValueError('final checkpoint identity changed')
    model.load_state_dict(saved['state_dict']); del saved
    # Evaluation is forbidden inside the training loop and has no influence on save.
    results={'condition':condition,'seed':seed,'checkpoint_rule':'last_epoch_only','completed_epochs':EPOCHS,
             'actual_optimizer_steps':steps,'checkpoint_sha256':digest(final),'test_forward_count':len(assets['test']),
             'metric_type_labels':{t.replace(' ','_'):t for t in assets['defect_types']},
             'metrics':evaluate_final(model,assets,calculate_pro,device,target/'predictions-final.npz')}
    new_json(target/'results.json',results)
    return results
