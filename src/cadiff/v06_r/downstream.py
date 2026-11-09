"""Isolated, manifest-driven DRAEM comparison; test is never a selection set."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

ARMS = ('latent_control', 'rgb', 'rgb_edge')
SEEDS = (2026101301, 2026101302, 2026101303)
CONDITIONS = ('real_fs',) + tuple(f'{kind}_{arm}' for kind in ('real_gen', 'gen_only') for arm in ARMS)
EPOCHS, LENGTH, BS, DIM = 200, 500, 16, 256

def validate_protocol(protocol):
    d=protocol['detector']; p=protocol['pool']
    expected={'lr':1e-4,'epochs':200,'epoch_samples':500,'batch_size':16,'grad_accumulation':1,
              'updates_per_epoch':32,'total_optimizer_updates':6400,'scheduler_milestones':[160,180],
              'scheduler_gamma':.2,'precision':'FP32','test_evaluations_per_run':1}
    if protocol.get('schema')!='v06_r_downstream_100_v1' or protocol.get('arms')!=list(ARMS) or protocol.get('detector_seeds')!=list(SEEDS):
        raise ValueError('unregistered downstream protocol')
    if any(d.get(k)!=v for k,v in expected.items()) or any(p.get(k)!=v for k,v in {'per_type':100,'per_arm':400,'total':1200,'candidate_count':1,'ddim_steps':50,'eta':0,'min_shape_pixels':64}.items()):
        raise ValueError('registered numerical parameters differ from implementation')
    if protocol.get('sampling',{}).get('replacement') is not True:
        raise ValueError('replacement sampling must be registered')
    return protocol

def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()

def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def new_json(path, value):
    with Path(path).open('x', encoding='utf-8') as f:
        json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write('\n')

def atomic_json(path, value):
    tmp = Path(str(path)+'.tmp')
    with tmp.open('w', encoding='utf-8') as f:
        json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write('\n')
    os.replace(tmp, path)

def verify_record(row):
    for key in ('image', 'mask'):
        path = row.get(f'{key}_path')
        if path is None:
            if key == 'image' or row.get(f'{key}_sha256') is not None:
                raise ValueError('invalid null asset')
            continue
        if digest(path) != row.get(f'{key}_sha256'):
            raise ValueError(f'{key} SHA changed for {row["identity"]}')

def validate_assets(assets, *, verify_files=True, strict=True):
    if assets.get('schema_version') != 'v06_r_downstream_assets_v1':
        raise ValueError('unknown downstream asset schema')
    types = assets['defect_types']
    if len(types) != 4 or len(set(types)) != 4 or 'combined' in types:
        raise ValueError('four single-label types required')
    required = {'real_fewshot': {'fewshot_train'}, 'train_normals': {'normal_train'},
                'test': {'normal_test', 'heldout_test'}}
    paths = {}
    identities = set()
    for partition, roles in required.items():
        for row in assets[partition]:
            if row['role'] not in roles or row['identity'] in identities:
                raise ValueError('mixed role or duplicate identity')
            identities.add(row['identity'])
            path = str(Path(row['image_path']).resolve())
            if path in paths:
                raise ValueError('train/test image identity overlap')
            paths[path] = partition
            normal = row['role'] in ('normal_train', 'normal_test')
            if normal != (row['defect_type'] == 'good') or (normal and row.get('mask_path') is not None):
                raise ValueError('normal role/type/mask mismatch')
            if not normal and (row['defect_type'] not in types or not row.get('mask_path')):
                raise ValueError('unknown positive type or missing GT')
            if verify_files:
                verify_record(row)
                if not normal:
                    with Image.open(row['mask_path']) as im:
                        values=np.asarray(im.convert('L'))
                        if not np.isin(values,[0,255]).all() or not values.any():
                            raise ValueError('native positive GT must be frozen nonempty binary 0/255, not VisA class codes')
    if strict:
        if tuple(len(assets[k]) for k in required) != (18,453,89):
            raise ValueError('frozen population must be 18/453/89')
        counts = {t:sum(r['defect_type']==t for r in assets['real_fewshot']) for t in types}
        expected = {'chunk of gum missing':5, 'corner missing':4, 'scratches':3, 'similar colour spot':6}
        held = {t:sum(r['defect_type']==t for r in assets['test']) for t in types}
        if counts != expected or held != {'chunk of gum missing':11,'corner missing':9,'scratches':6,'similar colour spot':13}:
            raise ValueError('frozen support or heldout split changed')
        if sum(r['role']=='normal_test' for r in assets['test']) != 50:
            raise ValueError('normal test population changed')
    return assets

def make_plan(seed, real_counts, gen_counts, normal_count, *, epochs=EPOCHS):
    """Independent replacement streams, shared slot geometry, fixed full/tail batches."""
    if len(real_counts) != 4 or len(gen_counts) != 4 or min(*real_counts,*gen_counts,normal_count)<1:
        raise ValueError('every sampling population must be nonempty')
    rng_n,rng_r,rng_g = [np.random.default_rng(np.random.SeedSequence([seed,s])) for s in (11,22,33)]
    shape=(epochs,LENGTH)
    plan={k:np.full(shape,-1,dtype=np.int32) for k in ('normal','type','real','gen','mixed_source')}
    for e in range(epochs):
        for slot in range(LENGTH):
            if slot%2==0:
                plan['normal'][e,slot]=rng_n.integers(normal_count)
                continue
            j=slot//2
            t=j%4 if slot<496 else (e%2)*2+(slot-497)//2
            source=(j//4)%2 if slot<496 else ((slot-497)//2 + e//2)%2
            plan['type'][e,slot]=t
            plan['real'][e,slot]=rng_r.integers(real_counts[t])
            plan['gen'][e,slot]=rng_g.integers(gen_counts[t])
            plan['mixed_source'][e,slot]=source
    return plan

def kit_imports(kit):
    path=str(Path(kit).resolve(strict=True))
    if path not in sys.path:
        sys.path.insert(0,path)
    from evaluation.localization import _weights_init
    from evaluation.unet_utils.model_unet import DiscriminativeSubNetwork
    from evaluation.unet_utils.loss import FocalLoss
    from evaluation.unet_utils.au_pro_util import calculate_au_pro
    return DiscriminativeSubNetwork,FocalLoss,_weights_init,calculate_au_pro

def validate_gen(rows, assets, *, verify_files=True):
    types=assets['defect_types']
    if len(rows)!=400 or len({r['identity'] for r in rows})!=400:
        raise ValueError('400 unique generated identities required')
    if any(sum(r['defect_type']==t for r in rows)!=100 for t in types):
        raise ValueError('generated population must be 100 per type')
    train_paths={str(Path(r['image_path']).resolve()) for r in assets['train_normals']}
    for r in rows:
        if r['role']!='generated_train' or r.get('candidate_count')!=1 or r.get('ddim_steps')!=50 or r.get('eta')!=0:
            raise ValueError('generated role or sampler contract changed')
        if r.get('uses_test_mask') is not False or r.get('normal_role')!='amazon_1cls_train':
            raise ValueError('generation used heldout data')
        if str(Path(r['normal_native_path']).resolve()) not in train_paths:
            raise ValueError('generation normal not in training population')
        rect=r['valid_rectangle']
        if len(rect)!=4 or any(type(v)!=int for v in rect) or not (0<=rect[0]<rect[2]<=512 and 0<=rect[1]<rect[3]<=512):
            raise ValueError('invalid frozen valid rectangle')
        if verify_files:
            verify_record(r)
            with Image.open(r['mask_path']) as im:
                mask=np.asarray(im.convert('L'))
                if mask.shape!=(512,512) or not np.isin(mask,[0,255]).all(): raise ValueError('generated GT must be original binary 512')
                valid=np.zeros((512,512),bool); valid[rect[1]:rect[3],rect[0]:rect[2]]=True
                if np.any((mask>0)&~valid): raise ValueError('generated label leaves frozen valid rectangle')
    return rows

class PlannedDataset(Dataset):
    def __init__(self,assets,generated,plan,condition,epoch):
        self.assets,self.plan,self.condition,self.epoch=assets,plan,condition,epoch
        self.real=[[r for r in assets['real_fewshot'] if r['defect_type']==t] for t in assets['defect_types']]
        self.gen=[[r for r in generated if r['defect_type']==t] for t in assets['defect_types']]
        self.loader=transforms.Compose([transforms.ToTensor(),transforms.Resize([DIM,DIM],antialias=True)])
    def __len__(self): return LENGTH
    def __getitem__(self,slot):
        e=self.epoch
        n=int(self.plan['normal'][e,slot])
        if n>=0:
            r=self.assets['train_normals'][n]
            with Image.open(r['image_path']) as im: image=self.loader(im.convert('RGB'))
            return {'image':image,'mask':torch.zeros((1,DIM,DIM))}
        t=int(self.plan['type'][e,slot])
        use_gen=self.condition.startswith('gen_only') or (self.condition.startswith('real_gen') and self.plan['mixed_source'][e,slot]==1)
        key='gen' if use_gen else 'real'
        group=self.gen if use_gen else self.real
        r=group[t][int(self.plan[key][e,slot])]
        with Image.open(r['image_path']) as im, Image.open(r['mask_path']) as ma:
            image,mask=im.convert('RGB'),ma.convert('L')
            if image.size!=mask.size: raise ValueError('paired geometry differs')
            if use_gen:
                if image.size!=(512,512): raise ValueError('generated input must retain original 512 canvas')
                rectangle=tuple(r['valid_rectangle'])
                image,mask=image.crop(rectangle),mask.crop(rectangle)
            image=self.loader(image)
            mask=(self.loader(mask)>.5).float()
        return {'image':image,'mask':mask}

def seed_everything(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True

def save_resume(path,model,optimizer,scheduler,epoch,steps,contract):
    state={'model':model.state_dict(),'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),
           'epoch':epoch,'steps':steps,'contract':contract,'torch_rng':torch.get_rng_state(),
           'numpy_rng':np.random.get_state(),'python_rng':random.getstate(),
           'cuda_rng':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}
    tmp=Path(str(path)+'.tmp')
    torch.save(state,tmp); os.replace(tmp,path)

def load_resume(path,model,optimizer,scheduler,contract,device):
    state=torch.load(path,map_location=device,weights_only=False)
    if state['contract']!=contract or state['steps']!=state['epoch']*32:
        raise ValueError('resume identity/budget differs')
    model.load_state_dict(state['model']); optimizer.load_state_dict(state['optimizer']); scheduler.load_state_dict(state['scheduler'])
    torch.set_rng_state(state['torch_rng'].cpu()); np.random.set_state(state['numpy_rng']); random.setstate(state['python_rng'])
    if state['cuda_rng'] is not None and torch.cuda.is_available(): torch.cuda.set_rng_state_all([r.cpu() for r in state['cuda_rng']])
    return state['epoch'],state['steps']

def claim_run_directory(target,resume):
    if target.exists() and not resume: raise FileExistsError('run exists; resume must be explicit')
    if not target.exists() and resume: raise FileNotFoundError('no run to resume')
    if not target.exists(): target.mkdir(parents=True,exist_ok=False)
    if resume and not (target/'resume.pt').exists() and not (target/'results.json').exists():
        allowed={'resume.pt.tmp','epoch_zero_restart.json'}
        if any(p.name not in allowed for p in target.iterdir()):
            raise ValueError('committed run artifacts exist but resume state is missing')
        return True
    return False

def metrics_from_predictions(rows,predictions,calculate_pro):
    """Same formulas/thresholds as ROOT eval_kit.evaluate_localization; one cached inference."""
    from sklearn.metrics import roc_auc_score,average_precision_score,precision_recall_curve
    gt=np.array([p['has_anomaly'] for p in predictions])
    scores=np.array([p['image_score'] for p in predictions])
    pg=np.concatenate([p['mask'].ravel() for p in predictions]).astype(np.uint8)
    ps=np.concatenate([p['heat'].ravel() for p in predictions])
    out={'auroc_image':float(roc_auc_score(gt,scores)),'ap_image':float(average_precision_score(gt,scores)),
         'auroc_pixel':float(roc_auc_score(pg,ps)),'ap_pixel':float(average_precision_score(pg,ps))}
    prec,rec,_=precision_recall_curve(gt,scores)
    f1=2*prec*rec/np.maximum(prec+rec,1e-8)
    out['f1_image']=float(np.max(f1[np.isfinite(f1)]))
    best=0.; gb=pg.astype(bool)
    for threshold in np.arange(0.,1.+1e-3,.05):
        pr=ps>threshold; inter=np.logical_and(gb,pr).sum()
        precision=inter/(pr.sum()+1e-8); recall=inter/(gb.sum()+1e-8)
        best=max(best,2*precision*recall/(precision+recall+1e-8))
    out['f1_pixel']=float(best)
    out['pro_pixel']=float(calculate_pro([p['mask'] for p in predictions],[p['heat'] for p in predictions])[0])
    out['image_count']=len(rows)
    return out

@torch.no_grad()
def evaluate_final(model,assets,calculate_pro,device,cache_path=None):
    import cv2
    model.eval(); predictions=[]
    cached=cache_path is not None and cache_path.exists()
    for r in ([] if cached else assets['test']):
        bgr=cv2.imread(r['image_path'],cv2.IMREAD_COLOR)
        if bgr is None: raise FileNotFoundError(r['image_path'])
        rgb=cv2.cvtColor(cv2.resize(bgr,(DIM,DIM)),cv2.COLOR_BGR2RGB).astype(np.float32)/255
        image=torch.from_numpy(rgb.transpose(2,0,1)).unsqueeze(0).to(device)
        probs=torch.softmax(model(image),dim=1)
        score=float(torch.nn.functional.avg_pool2d(probs[:,1:],21,stride=1,padding=10).max())
        if r['mask_path'] is None: mask=np.zeros((DIM,DIM),np.float32)
        else:
            raw=cv2.imread(r['mask_path'],cv2.IMREAD_GRAYSCALE)
            if raw is None: raise FileNotFoundError(r['mask_path'])
            mask=(cv2.resize(raw,(DIM,DIM)).astype(np.float32)/255>.5).astype(np.float32)
        predictions.append({'has_anomaly':float(r['role']=='heldout_test'),'image_score':score,'mask':mask,'heat':probs[0,1].cpu().numpy()})
    if cached:
        with np.load(cache_path,allow_pickle=False) as z:
            if list(z['identities']) != [r['identity'] for r in assets['test']]: raise ValueError('prediction cache identity differs')
            predictions=[{'has_anomaly':float(z['gt'][i]),'image_score':float(z['scores'][i]),'mask':z['masks'][i],'heat':z['heat'][i]} for i in range(len(assets['test']))]
    elif cache_path is not None:
        tmp=Path(str(cache_path)+'.tmp')
        with tmp.open('wb') as f:
            np.savez_compressed(f,identities=np.array([r['identity'] for r in assets['test']]),gt=np.array([p['has_anomaly'] for p in predictions]),
                                scores=np.array([p['image_score'] for p in predictions]),masks=np.stack([p['mask'] for p in predictions]),heat=np.stack([p['heat'] for p in predictions]))
        os.replace(tmp,cache_path)
    groups={'overall':set(assets['defect_types']),'primary':{'scratches','similar colour spot'}}
    groups.update({t.replace(' ','_'):{t} for t in assets['defect_types']})
    results={}
    for name,types in groups.items():
        select=[i for i,r in enumerate(assets['test']) if r['defect_type']=='good' or r['defect_type'] in types]
        results[name]=metrics_from_predictions([assets['test'][i] for i in select],[predictions[i] for i in select],calculate_pro)
    return results

def prepare(assets_path,identities_path,protocol_path,pool_root,eval_kit,output):
    if output.exists(): raise FileExistsError('refuses to overwrite downstream preparation')
    assets=validate_assets(read_json(assets_path))
    protocol=validate_protocol(read_json(protocol_path))
    identities=read_json(identities_path)
    if identities.get('schema_version')!='v06_r_matched_pool_v1' or identities.get('defect_types')!=assets['defect_types'] or len(identities['rows'])!=400:
        raise ValueError('frozen generation identity population differs')
    if len({r['identity'] for r in identities['rows']})!=400 or any(sum(r['defect_type']==t for r in identities['rows'])!=100 for t in assets['defect_types']):
        raise ValueError('generation identities must be unique with 100 per type')
    manifests={arm:{'path':str((pool_root/arm/'manifest.json').resolve())} for arm in ARMS}
    sources={str(Path(eval_kit)/p):digest(Path(eval_kit)/p) for p in ('evaluation/localization.py','evaluation/datasets.py','evaluation/unet_utils/model_unet.py','evaluation/unet_utils/loss.py','evaluation/unet_utils/au_pro_util.py')}
    sources[str(Path(__file__).resolve())]=digest(__file__)
    output.mkdir(parents=True,exist_ok=False)
    contract={'schema_version':'v06_r_downstream_frozen_v1','assets':assets,'assets_path':str(assets_path.resolve()),'assets_sha256':digest(assets_path),
              'pools':manifests,'eval_kit':str(Path(eval_kit).resolve()),'source_sha256':sources,'seeds':list(SEEDS),
              'protocol':protocol,'protocol_path':str(protocol_path.resolve()),'protocol_sha256':digest(protocol_path),
              'identities':identities,'identities_path':str(identities_path.resolve()),'identities_sha256':digest(identities_path),
              'condition_aliases':{c:c.replace('real_gen_','realgen_').replace('gen_only_','genonly_') for c in CONDITIONS},
              'epochs':EPOCHS,'epoch_length':LENGTH,'batch_size':BS,'optimizer_updates':6400,'checkpoint_rule':'last_epoch_only',
              'test_evaluation':'once after final checkpoint; all partitions from same cached inference','generated_preprocessing':'crop_frozen_valid_rectangle_then_original_resize',
              'sampling':'paired replacement; 31 full16 plus tail4 per epoch; real/gen indexes independent'}
    new_json(output/'frozen.json',contract)
    for seed in SEEDS:
        rc=[sum(r['defect_type']==t for r in assets['real_fewshot']) for t in assets['defect_types']]
        plan=make_plan(seed,rc,[100]*4,len(assets['train_normals']))
        np.savez_compressed(output/f'plan-{seed}.npz',**plan)
    new_json(output/'plan_sha256.json',{str(seed):digest(output/f'plan-{seed}.npz') for seed in SEEDS})
    return {'prepared':str(output),'conditions':list(CONDITIONS),'seeds':list(SEEDS)}

def run(output,condition,seed,*,device='cuda:0',resume=False,stop_after_epoch=None):
    if condition not in CONDITIONS or seed not in SEEDS: raise ValueError('unregistered condition/seed')
    frozen=read_json(output/'frozen.json')
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
    if condition!='real_fs':
        arm=next(a for a in ARMS if condition.endswith('_'+a))
        record=frozen['pools'][arm]
        payload=read_json(record['path'])
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
              'pool_manifest_sha256':digest(record['path']) if condition!='real_fs' else None}
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

def smoke(output,smoke_output,*,device='cuda:0',updates=2):
    """Separate engineering output; never runs or queries test."""
    import time
    if updates not in (1,2,32): raise ValueError('smoke allows 1/2 updates or one-epoch32 benchmark')
    if smoke_output.exists() or smoke_output.resolve().is_relative_to(output.resolve()):
        raise FileExistsError('smoke needs a new directory outside formal evaluation')
    frozen=read_json(output/'frozen.json'); assets=validate_assets(frozen['assets'])
    Model,Loss,weights_init,_=kit_imports(frozen['eval_kit']); seed=SEEDS[0]; seed_everything(seed)
    with np.load(output/f'plan-{seed}.npz',allow_pickle=False) as z: plan={k:z[k].copy() for k in z.files}
    model=Model(in_channels=3,out_channels=2).to(device=device,dtype=torch.float32); model.apply(weights_init)
    optimizer=torch.optim.Adam(model.parameters(),lr=1e-4); loss_fn=Loss()
    smoke_output.mkdir(parents=True,exist_ok=False)
    loader=DataLoader(PlannedDataset(assets,[],plan,'real_fs',0),batch_size=BS,shuffle=False,num_workers=0)
    if torch.device(device).type=='cuda': torch.cuda.reset_peak_memory_stats(device); torch.cuda.synchronize(device)
    start=time.monotonic(); losses=[]
    for i,batch in enumerate(loader):
        optimizer.zero_grad()
        loss=loss_fn(torch.softmax(model(batch['image'].to(device)),dim=1),batch['mask'].to(device)); loss.backward()
        if not torch.isfinite(loss) or not torch.stack([torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None]).all(): raise RuntimeError('nonfinite smoke')
        optimizer.step(); losses.append(float(loss.detach()))
        if i+1>=updates: break
    if updates==32:
        scheduler=torch.optim.lr_scheduler.MultiStepLR(optimizer,[160,180],gamma=.2)
        scheduler.step()
        save_resume(smoke_output/'resume-benchmark.pt',model,optimizer,scheduler,1,32,{'engineering_only':True,'seed':seed})
    if torch.device(device).type=='cuda': torch.cuda.synchronize(device)
    seconds=time.monotonic()-start
    result={'engineering_only':True,'actual_optimizer_steps':len(losses),'test_calls':0,'batch_losses':losses,'wall_seconds':seconds,
            'estimated_train_seconds_per_run':seconds*6400/len(losses),'peak_gpu_bytes':torch.cuda.max_memory_allocated(device) if torch.device(device).type=='cuda' else None}
    new_json(smoke_output/'summary.json',result)
    return result
