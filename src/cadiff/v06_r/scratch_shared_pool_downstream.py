"""Accepted-domain single-scratch DRAEM plans; rejected candidates remain in eligibility yield."""
from __future__ import annotations
import hashlib,json,math,os,random,sys
from dataclasses import asdict
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader,Dataset
from torchvision import transforms
from cadiff.v06_r.scratch_shared_pool import WORK,ROOT,RUN,PILOT,CATALOG,read,new,now
from cadiff.v06_r.downstream import kit_imports,seed_everything,save_resume,load_resume,metrics_from_predictions
from cadiff.provenance import sha256_file

SEEDS=(2026101601,2026101602,2026101603)
CONDITIONS=('real_fs','full','crop32')
EPOCHS,LENGTH,BS,DIM=200,500,16,256
KIT=ROOT/'eval_kit'

def make_plan(seed,real_count,normal_count,pool_count,epochs=EPOCHS):
    if min(real_count,normal_count,pool_count)<1 or pool_count>40:raise ValueError('empty/wrong accepted sampling population')
    streams={name:np.random.default_rng(np.random.SeedSequence([seed,tag])) for name,tag in [('normal',11),('base_real',22),('extra_replay',33),('extra_pool',44)]}
    pool_order=streams['extra_pool'].permutation(pool_count)
    shape=(epochs,LENGTH)
    plan={name:np.full(shape,-1,dtype=np.int32) for name in streams}
    for e in range(epochs):
        for slot in range(LENGTH):
            if slot%2==0:plan['normal'][e,slot]=streams['normal'].integers(normal_count)
            elif (slot//2)%2==0:plan['base_real'][e,slot]=streams['base_real'].integers(real_count)
            else:
                plan['extra_replay'][e,slot]=streams['extra_replay'].integers(real_count)
                plan['extra_pool'][e,slot]=pool_order[(e*125+(slot//2)//2)%pool_count]
    return plan

def prepare():
    output=RUN/'downstream'
    if output.exists():raise FileExistsError('downstream preparation already exists')
    output.mkdir()
    catalog=[json.loads(x) for x in CATALOG.read_text().splitlines() if x.strip()]
    candidates=read(RUN/'candidate_manifest.json')['rows'];assets={};files={str(CATALOG):sha256_file(CATALOG)};plans={}
    for cat in ('cashew','pcb1'):
        config=read(PILOT/'protocol.json')['objects'][cat]
        typ=config['positive_type'];manifest=Path(config['manifest'])
        supports=[json.loads(x) for x in manifest.read_text().splitlines() if x.strip()]
        real=[]
        for row in supports:
            maskpath=ROOT/'data/VisA'/row['mask_relpath'];imagepath=ROOT/'data/VisA'/row['image_relpath']
            real.append({'identity':cat+'-real-'+Path(row['image_relpath']).stem,'image_path':str(imagepath),'image_sha256':row['image_sha256'],'mask_path':str(maskpath),'mask_sha256':row['mask_sha256'],'defect_type':typ,'role':'fewshot_train','binary_mask_rule':'native_nonzero_before_resize'})
        normal=[];test=[]
        for row in catalog:
            if row['category']!=cat:continue
            role=row['role'];is_scratch=row.get('original_label')==typ or row['type']==typ or row['type']==typ.replace(' ','_')
            if role not in ('normal_train','normal_test') and not(role=='anomaly_heldout' and is_scratch):continue
            record={'identity':cat+'-'+role+'-'+row['source_id'],'image_path':row['source_path'],'image_sha256':row['image_sha256'],'mask_path':None,'mask_sha256':None,'defect_type':'good' if role.startswith('normal') else typ,'role':role if role.startswith('normal') else 'heldout_test'}
            if role=='anomaly_heldout':
                record['mask_path']=str(ROOT/'data/VisA'/row['mask_relpath']);record['mask_sha256']=row['native_mask']['file_sha256'];record['binary_mask_rule']='native_nonzero_before_resize'
            if role=='normal_train':normal.append(record)
            else:test.append(record)
        normal.sort(key=lambda r:r['image_path']);test.sort(key=lambda r:r['identity']);real.sort(key=lambda r:r['identity'])
        assert len(real)==(5 if cat=='cashew' else 4)
        assert sum(r['role']=='heldout_test' for r in test)==10
        assert sum(r['role']=='normal_test' for r in test)==(50 if cat=='cashew' else 100)
        allrows=real+normal+test
        assert len({r['image_path'] for r in allrows})==len(allrows)
        for row in allrows:
            files[row['image_path']]=row['image_sha256']
            if row['mask_path']:files[row['mask_path']]=row['mask_sha256']
        slots=[]
        for row in candidates:
            if row['category']!=cat:continue
            for seed in (2026101501,2026101502):slots.append({'nominal_index':len(slots),'identity':row['identity'],'review_id':row['review_id'],'inference_seed':seed})
        assert len(slots)==40
        assets[cat]={'category':cat,'defect_types':[typ],'real_fewshot':real,'train_normals':normal,'test':test,'nominal_extra_slots':slots,'test_pixel_reads_during_preparation':0}
        plans[cat]={}
        for seed in SEEDS:
            plan=make_plan(seed,len(real),len(normal),40);path=output/f'plan-{cat}-{seed}.npz';np.savez_compressed(path,**plan);plans[cat][str(seed)]={'path':str(path),'sha256':sha256_file(path)}
    recipe={'schema':'scratch_shared_pool_detector_recipe_v1','registered_utc':now(),'seeds':list(SEEDS),'conditions':list(CONDITIONS),'model':'existing DRAEM DiscriminativeSubNetwork in_channels3 out_channels2, original weights_init, no pretrained model','lr':1e-4,'optimizer':'Adam','epochs':200,'epoch_samples':500,'batch_size':16,'dimension':256,'updates_per_epoch':32,'total_updates':6400,'scheduler_milestones':[160,180],'scheduler_gamma':.2,'precision':'FP32','sampling':{'per_epoch':{'normal':250,'base_real':125,'extra':125},'extra_realfs':'prelocked real replay index','extra_full_crop':'same nominal40slot index, paired available generated sample else same prelocked real replay','replacement':True,'normal_and_base_real_streams_shared_across_conditions':True,'nominal_denominator_per_object':40,'realfs_real_exposures_total':50000,'full_crop_base_real_exposures_total':25000,'full_crop_gen_maximum_exposures_total':25000,'paired_unavailability_adds_common_replay_real_exposures':True,'all_total_presentations':100000,'all_normal_presentations':50000,'no_refill_or_sampling_only_from_successes':True},'mask_preprocessing':'Native VisA real/test nonzero foreground is binarized before resize; gen physical H 0/255. Real+normal native RGB directly resize256; gen512 unpad valid original rectangle then resize256. Torchvision antialias bilinear mask>.5 during training; existing OpenCV metric pipeline for final test.','checkpoint':'fixed final epoch200 only; no test call or best checkpoint selection inside training','evaluation':'same exact scratchheldout10+official normaltest50(cashew)/100(pcb1), one final forward cache, existing AP/AUROC/PRO formulas; no multi-type/whole-object claim','failure_policy':'Nonfinite detector loss/gradient or checkpoint/input integrity failure stops stage; no automatic retry/resume or precision rescue. Explicit complete-epoch optimizer/scheduler/RNG recovery only after review.','test_policy':'CPU preparation locks role/identity/source hashes only, test image/mask pixels first read by fixed-final evaluation after training; no sample selection via test.','old_baseline_reused':False,'eval_kit':str(KIT),'assets_counts':{c:{k:len(assets[c][k]) for k in ('real_fewshot','train_normals','test','nominal_extra_slots')} for c in assets}}
    new(output/'assets.json',assets);new(output/'plan_manifest.json',plans);new(output/'recipe.json',recipe);new(output/'asset_source_sha256.json',files)
    new(output/'cpu_plan_checks.json',check_plans(assets,plans))
    print(json.dumps({'prepared':str(output),'counts':recipe['assets_counts'],'detector_runs_planned':18,'gpu_calls':0,'test_pixel_reads':0}))

def check_plans(assets,plans):
    checks=[]
    for cat in assets:
        for seed,entry in plans[cat].items():
            with np.load(entry['path'],allow_pickle=False) as z:plan={k:z[k] for k in z.files}
            assert all(v.shape==(200,500) for v in plan.values())
            assert np.all((plan['normal']>=0).sum(axis=1)==250)
            assert np.all((plan['base_real']>=0).sum(axis=1)==125)
            assert np.all((plan['extra_replay']>=0).sum(axis=1)==125)
            assert np.array_equal(plan['extra_replay']>=0,plan['extra_pool']>=0)
            for key in ('normal','base_real','extra_replay'):
                bound=len(assets[cat]['train_normals']) if key=='normal' else len(assets[cat]['real_fewshot'])
                assert plan[key].max()<bound
            assert plan['extra_pool'].min()==-1 and plan['extra_pool'].max()<40
            regenerated=make_plan(int(seed),len(assets[cat]['real_fewshot']),len(assets[cat]['train_normals']),40)
            assert all(np.array_equal(plan[k],regenerated[k]) for k in plan)
            checks.append({'category':cat,'seed':int(seed),'normal_slots':50000,'base_real_slots':25000,'extra_slots':25000,'optimizer_updates':6400,'deterministic_exact_regeneration':True,'pool_slots_seen':len(set(plan['extra_pool'][plan['extra_pool']>=0].tolist()))})
    return {'created_utc':now(),'checks':checks,'cuda_initialized':torch.cuda.is_initialized(),'test_pixel_reads':0}

def prepare_accepted():
    from cadiff.v06_r.scratch_shared_pool_execution import eligibility
    decisions=eligibility();output=RUN/'downstream/accepted_sampling'
    if output.exists():raise FileExistsError('accepted sampling already registered')
    output.mkdir();assets=read(RUN/'downstream/assets.json');accepted={};plans={};checks=[]
    for cat,a in assets.items():
        slots=[r for r in a['nominal_extra_slots'] if decisions[r['review_id']]['decision']=='pass']
        accepted[cat]=[{**r,'pool_index':i} for i,r in enumerate(slots)];plans[cat]={}
        if not slots:continue
        for seed in SEEDS:
            plan=make_plan(seed,len(a['real_fewshot']),len(a['train_normals']),len(slots));path=output/f'plan-{cat}-{seed}.npz';np.savez_compressed(path,**plan);plans[cat][str(seed)]={'path':str(path),'sha256':sha256_file(path)}
            counts=np.bincount(plan['extra_pool'][plan['extra_pool']>=0],minlength=len(slots))
            assert counts.max()-counts.min()<=1 and counts.sum()==25000
            assert np.all((plan['normal']>=0).sum(axis=1)==250) and np.all((plan['base_real']>=0).sum(axis=1)==125)
            checks.append({'category':cat,'seed':seed,'accepted_pair_seed_pool_size':len(slots),'extra_exposures_per_pool_sample':counts.tolist(),'maximum_minimum_exposure_difference':int(counts.max()-counts.min()),'normal_exposures':50000,'base_real_exposures':25000,'extra_generated_or_realfs_replay_exposures':25000,'optimizer_updates':6400})
    recipe=read(RUN/'downstream/recipe.json')
    recipe.update(schema='scratch_shared_pool_detector_accepted_domain_v2',registered_utc=now(),supersedes='downstream/recipe.json and plan_manifest.json were unexecuted CPU draft v1; preserve as history, never execute',human_eligibility_sha256=sha256_file(RUN/'human_eligibility.json'),estimand='Within independently human-approved scratch location/type domain, paired frozen Full-versus-Crop R comparison at equal compute and real/gen presentation counts',sampling={'per_epoch':{'normal':250,'base_real':125,'extra':125},'extra_realfs':'prelocked real replay index','extra_full_crop':'balanced seeded cyclic accepted pair-by-both-seeds pool index; every approved slot must succeed in both views, no fallback','normal_and_base_real_streams_shared_across_conditions':True,'accepted_pool_counts':{c:len(v) for c,v in accepted.items()},'realfs_real_exposures_total':50000,'full_crop_real_exposures_total':25000,'full_crop_gen_exposures_total':25000,'all_normal_exposures_total':50000,'all_total_presentations':100000,'extra_pool_balancing':'single seeded permutation then continuous cycles across all25000 extra slots, per-sample exposures differ at most1','rejected_uncertain':'qualification denominator only; never generate/refill or include in detector pool','any_accepted_generation_failure':'pause; no remove/fallback or resample; same-slot same-seed engineering recovery only after explicit registration','zero_pass_object':'NA for R comparison; do not run this object detector group without separately reviewed scope'})
    new(output/'accepted_slots.json',accepted);new(output/'plan_manifest.json',plans);new(output/'recipe.json',recipe);new(output/'cpu_sampling_checks.json',{'created_utc':now(),'checks':checks,'test_pixel_reads':0,'gpu_calls':0})
    print(json.dumps({'accepted_pool_counts':{c:len(v) for c,v in accepted.items()},'active_objects':[c for c,v in accepted.items() if v],'test_pixel_reads':0,'gpu_calls':0}))

class PlannedDataset(Dataset):
    def __init__(self,assets,plan,condition,epoch,generated):
        self.assets,self.plan,self.condition,self.epoch=assets,plan,condition,epoch
        self.generated=generated
        self.loader=transforms.Compose([transforms.ToTensor(),transforms.Resize([DIM,DIM],antialias=True)])
    def __len__(self):return LENGTH
    def resolve(self,slot):
        e=self.epoch;n=int(self.plan['normal'][e,slot]);r=int(self.plan['base_real'][e,slot])
        if n>=0:return self.assets['train_normals'][n],'normal'
        if r>=0:return self.assets['real_fewshot'][r],'base_real'
        pool_index=int(self.plan['extra_pool'][e,slot]);replay=int(self.plan['extra_replay'][e,slot])
        if self.condition!='real_fs':
            return self.generated[self.condition][pool_index],'generated'
        return self.assets['real_fewshot'][replay],'extra_real_replay'
    def __getitem__(self,slot):
        row,kind=self.resolve(slot)
        with Image.open(row['image_path']) as im:image=im.convert('RGB')
        if kind=='normal':return {'image':self.loader(image),'mask':torch.zeros((1,DIM,DIM))}
        with Image.open(row['mask_path']) as im:mask=Image.fromarray((np.asarray(im.convert('L'))>0).astype(np.uint8)*255)
        if image.size!=mask.size:raise ValueError('native image/mask geometry differs')
        if kind=='generated':
            rect=tuple(row['valid_rectangle']);image,mask=image.crop(rect),mask.crop(rect)
        return {'image':self.loader(image),'mask':(self.loader(mask)>.5).float()}

def atomic(path,data):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n');temp.replace(path)

def write_or_reuse_equal(path,data):
    if path.exists():
        if read(path)!=data:raise ValueError('recovered final GT audit differs')
    else:new(path,data)

def reconstruct_exposure(plan,condition,completed_epochs):
    prefix={k:v[:completed_epochs] for k,v in plan.items()}
    extra=int((prefix['extra_pool']>=0).sum())
    counts={'normal':int((prefix['normal']>=0).sum()),'base_real':int((prefix['base_real']>=0).sum()),'generated':extra if condition!='real_fs' else 0,'extra_real_replay':extra if condition=='real_fs' else 0}
    if sum(counts.values())!=completed_epochs*500:raise ValueError('committed epoch exposure reconstruction differs')
    return counts

def audit_training_masks(assets,generated,condition):
    loader=transforms.Compose([transforms.ToTensor(),transforms.Resize([DIM,DIM],antialias=True)])
    rows=[];targets=[(r,'real_support') for r in assets['real_fewshot']]
    if condition!='real_fs':targets.extend((r,'generated') for _,r in sorted(generated[condition].items()))
    for row,role in targets:
        with Image.open(row['mask_path']) as im:mask=(np.asarray(im.convert('L'))>0).astype(np.uint8)*255
        native_area=int((mask>0).sum());image=Image.fromarray(mask)
        if role=='generated':image=image.crop(tuple(row['valid_rectangle']))
        area=int((loader(image)>.5).sum())
        rows.append({'identity':row['identity'],'role':role,'mask_path':row['mask_path'],'native_or_prepared512_positive_pixels':native_area,'after_unpad_size':list(image.size),'training256_positive_pixels':area,'all_zero_after_resize':area==0,'retained_even_if_zero':True})
    return {'schema':'scratch_training_mask_area_audit_v1','rows':rows,'positive_mask_count':len(rows),'all_zero_after_resize_count':sum(r['all_zero_after_resize'] for r in rows),'no_mask_exclusions':True}

def runtime_data(cat):
    assets=read(RUN/'downstream/assets.json')[cat]
    accepted=read(RUN/'downstream/accepted_sampling/accepted_slots.json')[cat]
    mapping={row['nominal_index']:row['pool_index'] for row in accepted}
    records=read(RUN/'generation_records.json');generated={v:{} for v in ('full','crop32')}
    for row in records:
        if row['category']==cat and row['status']=='success':generated[row['view']][mapping[row['nominal_index']]]={**row,'mask_path':row['output_H_path'],'valid_rectangle':row['valid_rectangle']}
    if any(len(generated[v])!=len(accepted) for v in ('full','crop32')):raise ValueError('accepted pool incomplete; no fallback')
    return assets,generated

@torch.no_grad()
def evaluate_scratch_final(model,assets,calculate_pro,device,cache_path):
    import cv2
    model.eval();predictions=[]
    if cache_path.exists():
        with np.load(cache_path,allow_pickle=False) as z:
            if list(z['identities'])!=[r['identity'] for r in assets['test']]:raise ValueError('cached test identity differs')
            predictions=[{'has_anomaly':float(z['gt'][i]),'image_score':float(z['scores'][i]),'mask':z['masks'][i],'heat':z['heat'][i]} for i in range(len(assets['test']))]
    else:
        for row in assets['test']:
            bgr=cv2.imread(row['image_path'],cv2.IMREAD_COLOR)
            if bgr is None:raise FileNotFoundError(row['image_path'])
            rgb=cv2.cvtColor(cv2.resize(bgr,(DIM,DIM)),cv2.COLOR_BGR2RGB).astype(np.float32)/255
            image=torch.from_numpy(rgb.transpose(2,0,1)).unsqueeze(0).to(device)
            probs=torch.softmax(model(image),dim=1)
            score=float(torch.nn.functional.avg_pool2d(probs[:,1:],21,stride=1,padding=10).max())
            if row['mask_path'] is None:mask=np.zeros((DIM,DIM),np.float32)
            else:
                raw=cv2.imread(row['mask_path'],cv2.IMREAD_GRAYSCALE)
                if raw is None:raise FileNotFoundError(row['mask_path'])
                mask=(cv2.resize(raw,(DIM,DIM)).astype(np.float32)/255>.5).astype(np.float32)
            predictions.append({'has_anomaly':float(row['role']=='heldout_test'),'image_score':score,'mask':mask,'heat':probs[0,1].cpu().numpy()})
        temp=Path(str(cache_path)+'.tmp')
        with temp.open('wb') as f:np.savez_compressed(f,identities=np.array([r['identity'] for r in assets['test']]),gt=np.array([p['has_anomaly'] for p in predictions]),scores=np.array([p['image_score'] for p in predictions]),masks=np.stack([p['mask'] for p in predictions]),heat=np.stack([p['heat'] for p in predictions]))
        temp.replace(cache_path)
    gt_areas=[]
    for row,prediction in zip(assets['test'],predictions):
        native_area=0
        if row['mask_path']:
            with Image.open(row['mask_path']) as im:native_area=int((np.asarray(im.convert('L'))>0).sum())
        area=int(prediction['mask'].sum())
        gt_areas.append({'identity':row['identity'],'role':row['role'],'native_positive_pixels':native_area,'evaluation256_positive_pixels':area,'positive_image_with_zero_resized_GT':row['role']=='heldout_test' and area==0,'retained_even_if_zero':True})
    metrics=metrics_from_predictions(assets['test'],predictions,calculate_pro)
    return {'overall':metrics,assets['defect_types'][0].replace(' ','_'):metrics.copy(),'GT_area_audit':{'rows':gt_areas,'positive_images_with_zero_resized_GT':sum(r['positive_image_with_zero_resized_GT'] for r in gt_areas),'no_exclusions':True},'F1_interpretation':'Existing helper scans test thresholds for descriptive oracle F1 only; F1 never selects checkpoint/settings/progression. Primary endpoint is pooled pixel AP.'}

def train(cat,condition,seed,*,smoke=False,resume=False,reviewed_protocol_sha=None):
    from cadiff.v06_r.scratch_shared_pool_execution import verify
    verify(reviewed_protocol_sha,require_generation=True,require_output_review=True)
    if not smoke:
        from cadiff.v06_r.scratch_shared_pool_execution import check
        ready=read(RUN/'detector_smoke_ready.json')
        if ready['protocol_sha256']!=reviewed_protocol_sha or ready['status']!='passed':raise ValueError('detector smoke gate missing')
        check(ready['evidence_sha256'])
    if condition not in CONDITIONS or seed not in SEEDS:raise ValueError('unregistered detector condition/seed')
    if smoke and resume:raise ValueError('smoke does not support resume; explicit isolated rerun only')
    root=RUN/'downstream';assets,generated=runtime_data(cat)
    accepted=read(root/'accepted_sampling/accepted_slots.json')[cat]
    if not accepted:raise ValueError('object has zero accepted pool; detector group NA')
    entry=read(root/'accepted_sampling/plan_manifest.json')[cat][str(seed)]
    if sha256_file(Path(entry['path']))!=entry['sha256']:raise ValueError('plan changed')
    with np.load(entry['path'],allow_pickle=False) as z:plan={k:z[k].copy() for k in z.files}
    target=(RUN/'detector_smoke' if smoke else root/'runs')/cat/condition/f'seed-{seed}'
    if target.exists() and not resume:raise FileExistsError('detector output exists')
    if resume and not target.exists():raise FileNotFoundError('no detector to resume')
    target.mkdir(parents=True,exist_ok=resume)
    if (target/'results.json').exists():return read(target/'results.json')
    mask_audit=audit_training_masks(assets,generated,condition)
    if not resume:new(target/'training_mask_area_audit.json',mask_audit)
    Model,Loss,weights_init,calculate_pro=kit_imports(KIT);seed_everything(seed)
    device='cuda:0';model=Model(in_channels=3,out_channels=2).to(device=device,dtype=torch.float32);model.apply(weights_init)
    from cadiff.v06_r.full_crop_training import module_sha256
    if not resume:new(target/'initial_state_sha256.json',{'model':module_sha256(model),'seed':seed})
    optimizer=torch.optim.Adam(model.parameters(),lr=1e-4);scheduler=torch.optim.lr_scheduler.MultiStepLR(optimizer,[160,180],gamma=.2);loss_fn=Loss()
    contract={'protocol_sha256':reviewed_protocol_sha,'plan_sha256':entry['sha256'],'availability_sha256':sha256_file(RUN/'paired_availability.json'),'category':cat,'condition':condition,'seed':seed,'smoke':smoke}
    start,steps=load_resume(target/'resume.pt',model,optimizer,scheduler,contract,device) if resume else (0,0)
    exposure=reconstruct_exposure(plan,condition,start)
    for epoch in range(start,1 if smoke else EPOCHS):
        model.train();dataset=PlannedDataset(assets,plan,condition,epoch,generated);loader=DataLoader(dataset,batch_size=BS,shuffle=False,num_workers=0);total=0.;seen=0
        for batch in loader:
            optimizer.zero_grad();loss=loss_fn(torch.softmax(model(batch['image'].to(device)),dim=1),batch['mask'].to(device))
            if not torch.isfinite(loss):raise RuntimeError('nonfinite detector loss')
            loss.backward()
            if not all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):raise RuntimeError('nonfinite detector gradients')
            optimizer.step();steps+=1;total+=float(loss.detach())
            for slot in range(seen,seen+len(batch['image'])):exposure[dataset.resolve(slot)[1]]+=1
            seen+=len(batch['image'])
            if smoke:break
        scheduler.step()
        save_resume(target/'resume.pt',model,optimizer,scheduler,epoch+1,steps,contract)
        progress={'category':cat,'condition':condition,'seed':seed,'completed_epoch':epoch+1,'actual_optimizer_steps':steps,'mean_batch_loss':total/(1 if smoke else len(loader)),'source_exposures':exposure,'test_calls':0,'smoke':smoke};atomic(target/'progress.json',progress);print(json.dumps(progress),flush=True)
    if smoke:
        new(target/'smoke_summary.json',{**progress,'complete':True,'test_forward_count':0});return
    if steps!=6400:raise ValueError('detector final successful-update budget differs')
    if sum(exposure.values())!=100000 or exposure['normal']!=50000 or exposure['base_real']!=25000:raise ValueError('detector source exposure budget differs')
    if condition=='real_fs' and (exposure['extra_real_replay']!=25000 or exposure['generated']):raise ValueError('RealFS replay exposure differs')
    if condition!='real_fs' and (exposure['generated']!=25000 or exposure['extra_real_replay']):raise ValueError('accepted RG exposure differs; no fallback')
    if not all(torch.isfinite(p).all() for p in model.parameters()):raise RuntimeError('nonfinite final detector parameters')
    final=target/'checkpoint-final.pt'
    if not final.exists():torch.save({'state_dict':model.state_dict(),'contract':contract,'epoch':200,'steps':6400},final)
    saved=torch.load(final,map_location=device,weights_only=False)
    if saved['contract']!=contract or saved['steps']!=6400:raise ValueError('final checkpoint contract differs')
    model.load_state_dict(saved['state_dict']);del saved
    # First test pixel reads occur here, after the sole final checkpoint is committed.
    test_assets={**assets,'test':[]};maskcache=target/'final_eval_binary_masks';maskcache.mkdir(exist_ok=True)
    for row in assets['test']:
        if sha256_file(Path(row['image_path']))!=row['image_sha256']:raise ValueError('test image changed')
        copy=row.copy()
        if row['mask_path']:
            if sha256_file(Path(row['mask_path']))!=row['mask_sha256']:raise ValueError('test mask changed')
            with Image.open(row['mask_path']) as im:binary=(np.asarray(im.convert('L'))>0).astype(np.uint8)*255
            path=maskcache/(row['identity']+'.png');Image.fromarray(binary).save(path);copy['mask_path']=str(path);copy['mask_sha256']=sha256_file(path)
        test_assets['test'].append(copy)
    metrics=evaluate_scratch_final(model,test_assets,calculate_pro,device,target/'predictions-final.npz')
    write_or_reuse_equal(target/'test_mask_area_audit.json',metrics['GT_area_audit'])
    new(target/'results.json',{'category':cat,'condition':condition,'seed':seed,'checkpoint_rule':'fixed_final_epoch200','actual_optimizer_steps':6400,'checkpoint_sha256':sha256_file(final),'source_exposures':exposure,'test_forward_count':len(assets['test']),'metrics':metrics,'task_scope':'scratch-only plus official normal_test','protocol_sha256':reviewed_protocol_sha,'training_mask_area_audit_sha256':sha256_file(target/'training_mask_area_audit.json'),'test_mask_area_audit_sha256':sha256_file(target/'test_mask_area_audit.json'),'primary_endpoint':'pooled pixel AP','F1_is_descriptive_test_oracle_only':True})

def main():
    import argparse
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['prepare','prepare-accepted','train']);p.add_argument('--category',choices=['cashew','pcb1']);p.add_argument('--condition',choices=CONDITIONS);p.add_argument('--seed',type=int,choices=SEEDS);p.add_argument('--smoke',action='store_true');p.add_argument('--resume',action='store_true');p.add_argument('--reviewed-protocol-sha');a=p.parse_args()
    if a.mode=='prepare':prepare()
    elif a.mode=='prepare-accepted':prepare_accepted()
    else:train(a.category,a.condition,a.seed,smoke=a.smoke,resume=a.resume,reviewed_protocol_sha=a.reviewed_protocol_sha)
if __name__=='__main__':main()
