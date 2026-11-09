"""Narrow two-object Full/Crop32 pilot. CPU preparation is separate from reviewed GPU execution."""
from __future__ import annotations
import argparse
from dataclasses import asdict,replace
from datetime import datetime,timezone
import fcntl,gc,hashlib,json,os,shutil,subprocess,sys,time,traceback
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from cadiff.provenance import sha256_file
from cadiff.v06_r.runner import write_new_json,load_train_manifest,load_config
from cadiff.v06_r.scratch_pilot_training import load_pilot_config,transform_samples,training_sources

WORK=Path(__file__).resolve().parents[3]
ROOT=WORK.parents[1]
OLD=WORK/'runs/v06_r/visa/20261015-fixed-h-existing-r-oracle'
RUN=WORK/'runs/v06_r/visa/20261015-scratch-full-vs-crop32'
PY=ROOT/'.conda/envs/cad-v0/bin/python'
VIEWS=('full','crop32')
SEEDS=(2026101501,2026101502)

def now(): return datetime.now(timezone.utc).isoformat()
def read(path): return json.loads(Path(path).read_text())
def fresh(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    write_new_json(path,data)
def state(path,data):
    temp=path.with_suffix('.tmp'); temp.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n'); temp.replace(path)
def check_hashes(files):
    for name,digest in files.items():
        if sha256_file(Path(name))!=digest: raise ValueError('frozen file changed: '+name)

def prepare():
    if RUN.exists(): raise FileExistsError('pilot run already exists')
    contracts=read(OLD/'checkpoint_contracts.json')
    oracle=read(OLD/'protocol.json'); inputs=read(OLD/'inputs_manifest.json')
    objects={}; files={}
    RUN.mkdir(parents=True,exist_ok=False)
    science=Path('/tmp/scratch_pilot_scientific_contract_20261015.json')
    with (RUN/'scientific_contract.json').open('xb') as f: f.write(science.read_bytes())
    files[str(RUN/'scientific_contract.json')]=sha256_file(RUN/'scientific_contract.json')
    for contract in contracts:
        cat=contract['category']; typ=contract['requested_type']
        ids={'cashew':['078','079','080','081','082'],'pcb1':['085','086','087','088']}[cat]
        raw=read(Path(contract['metadata_path']))['config'].copy()
        raw.update(defect_types=[typ],capacity_defect_type=typ,seed=20261015 if cat=='cashew' else 20261016,context_fill_seed=202610150 if cat=='cashew' else 202620150,noise_seed=202610151 if cat=='cashew' else 202620151,steps=1200)
        rows=[json.loads(x) for x in Path(contract['training_manifest']).read_text().splitlines() if x.strip()]
        rows=[r for r in rows if r.get('defect_type')==typ and r.get('roles')==['fewshot_train']]
        assert [Path(r['image_relpath']).stem for r in rows]==ids
        manifest=RUN/'inputs'/cat/'train_manifest.jsonl'; manifest.parent.mkdir(parents=True,exist_ok=True)
        with manifest.open('x') as f:
            for row in rows: f.write(json.dumps(row,sort_keys=True)+'\n')
        files[str(manifest)]=sha256_file(manifest)
        for row in rows:
            for key in ('image','mask'):
                path=ROOT/'data/VisA'/row[key+'_relpath']
                assert sha256_file(path)==row[key+'_sha256']; files[str(path)]=row[key+'_sha256']
        for key in ('metadata_path','training_manifest'):
            path=Path(contract[key]); files[str(path)]=sha256_file(path)
        objects[cat]={'config':raw,'positive_type':typ,'negative_labels':['normal'],'training_support_count':len(rows),'source_ids':ids,'manifest':str(manifest),'old_metadata_config_source':contract['metadata_path'],'old_weights_loaded':False,'generation_pairs':[r['row_id'] for r in inputs if r['category']==cat]}
    for row in inputs:
        folder=RUN/'inputs'/row['row_id']; folder.mkdir()
        for source_key,name,sha_key in [('normal_path','normal.png','normal_file_sha256'),('mask_path','H.png','mask_file_sha256')]:
            source=Path(row[source_key]); assert sha256_file(source)==row[sha_key]
            target=folder/name
            with target.open('xb') as f: f.write(source.read_bytes())
            files[str(target)]=sha256_file(target)
            row[source_key]=str(target)
    fresh(RUN/'inputs_manifest.json',inputs); files[str(RUN/'inputs_manifest.json')]=sha256_file(RUN/'inputs_manifest.json')
    for path in [OLD/'protocol.json',OLD/'inputs_manifest.json',OLD/'inspection/support_structure_readiness/manifest.json',OLD/'inspection/support_structure_readiness/astra_review.json',WORK/'configs/paths/omen-home.yaml']:
        files[str(path)]=sha256_file(path)
    files.update({p:h for p,h in oracle['frozen_files'].items() if p.startswith(oracle['model_root']+'/')})
    protocol={'schema_version':'v06_r_scratch_full_crop32_protocol_v1','registered_utc':now(),'purpose':'Exploratory scratch-only prepared512 scale/context Full-versus-Crop32 comparison; no claim of restored native detail.','parameters':{'steps':1200,'crop_margin':32,'learning_rate':1e-4,'type_rank':8,'save_every':100,'ddim_steps':50,'eta':0,'candidate_count':1},'objects':objects,'data_root':str(ROOT/'data/VisA'),'paths_yaml':str(WORK/'configs/paths/omen-home.yaml'),'model_root':oracle['model_root'],'inference_seeds':list(SEEDS),'planned_training_arms':4,'planned_generation_attempts':20,'detector_runs':0,'execution_order':['cashew/full','cashew/crop32','pcb1/full','pcb1/crop32'],'initialization':'fresh SD1.5 frozen backbone, fresh128 conditioner and Rank8 adapter; old metadata configuration only, no old adapter weights','training_loss':'original latent_control base plus unchanged hardest ranking formula, single scratch vocabulary makes negative candidate set normal-only; differs from old multiclass task','training_schedule':'deterministic image-relpath round robin5/4, equal1200 successful updates per arm; no randomized order claim','pairing':'Within object identical initial parameter RNG, sample, timestep, noise and context seed sequence; view-transformed context tensors may differ. Object seed blocks differ.','precision':{'train_prepare':'original FP16 VAE encoding before adapter initialization','after_init':'frozen VAE promoted FP32 as original Full/Crop implementation','inference':'fresh original FP32 VAE weights encode/decode, FP16 latent DDIM/unet, FP32 adapter masters'},'failure_policy':'existing bounded same-batch AMP scaler backoff is retained and logged; each outer step must advance AppliedAdamW exactly once; exception stops controller, no automatic external retry, no precision/loss rescue or replacement generation; exact checkpoint resume requires separate reviewed authorization','resume_policy':'all attempts retained in separate segment directories, full identity and optimizer/scaler/RNG restored; no automatic resume; rerun lost steps after last checkpoint identified and excluded from unique successful-step budget','fixed_final':'checkpoint-step-001200 only; no test-selected/best checkpoints','smoke':'isolated smoke paths, <=4 updates/arm and first preregistered pair/seed per object/view, formal starts fresh with no smoke state loaded; smoke artifacts excluded from20','generation':'same original5 accepted H/normal pairs and2 seeds per arm, complete20 attempts, no reroll or choice by appearance; preserve H and exact H-exterior RGB','prepared512_geometry':'full prepared512 or bbox(H)+32 clipped to valid rectangle then original square letterbox512; inverse float LANCZOS then exact physical H composition','min_shape_pixels':64,'cashew078_exception':'prepared real training GT61px retained as original training fact; never generation H, no lowering64px formal threshold','scope_limits':['Scratch-only trained-domain mechanism pilot on5/4 real supports;5 accepted pairs are reused training-support oracle combinations, not heldout confirmation.','No claim of whole-object/type coverage, new autonomous mask module, restored lost native detail, or original R repair causal estimate.','One generator-training seed per object/arm;2 inference seeds are not training repeats.','Old ten FP16-VAE oracle outputs are retained historical context and not matched causal comparator.','Direct visible structure/type/realism and visible G versus supplied H remain separate evidence; engineering integrity does not establish them.'],'direct_quality_gate':'All20 retained, review both seeds without requesting preferred-seed selection. Assess scratch identity, visible extent versus H, texture/material realism and PCB line-neighborhood relationship; retain weak/uncertain/regressed cases; no contrast-only proxy.','downstream_gate':'No detector or expanded H/normal generation authorized by this controller. Future narrow downstream contract requires direct-quality review and separate registration; no object-level4R/18D.','frozen_files':files}
    fresh(RUN/'protocol.json',protocol)
    checks={'created_utc':now(),'no_gpu_called':True,'cuda_initialized':torch.cuda.is_initialized(),'training':{},'generation_pair_count':len(inputs),'planned_generation_attempts':20}
    for cat in objects:
        config=load_pilot_config(RUN/'protocol.json',cat)
        samples,valid,records=load_train_manifest(Path(objects[cat]['manifest']),ROOT/'data/VisA',config)
        checks['training'][cat]={'count':len(samples),'source_ids':[Path(s.image_relpath).stem for s in samples],'prepared_gt_pixels':[int(s.mask.sum()) for s in samples],'prompt':config.normal_prompt,'negative_labels':['normal'],'views':{}}
        for view in VIEWS:
            transformed,legal,transforms=transform_samples(samples,valid,view)
            checks['training'][cat]['views'][view]={'count':len(transformed),'transforms':transforms}
        assert len(samples)==objects[cat]['training_support_count']
    for row in inputs:
        h=np.asarray(Image.open(row['mask_path']))>0
        assert h.sum()>=64
    fresh(RUN/'cpu_checks.json',checks)
    fresh(RUN/'group_state.json',{'status':'cpu_prepared_waiting_review','updated_utc':now(),'gpu_started':False,'protocol_sha256':sha256_file(RUN/'protocol.json')})
    print(json.dumps({'run':str(RUN),'cpu_prepared':True,'training_counts':{c:objects[c]['training_support_count'] for c in objects},'protocol_sha256':sha256_file(RUN/'protocol.json')}))

def freeze():
    protocol=read(RUN/'protocol.json'); check_hashes(protocol['frozen_files'])
    plan=[]
    for cat in ('cashew','pcb1'):
        for view in VIEWS:
            for row in read(RUN/'inputs_manifest.json'):
                if row['category']!=cat: continue
                for seed in SEEDS:
                    plan.append({'category':cat,'view':view,'row_id':row['row_id'],'inference_seed':seed,'status':'planned','record_path':str(RUN/'generation'/cat/view/(row['row_id']+'_seed-'+str(seed))/'record.json')})
    fresh(RUN/'generation_plan.json',plan)
    paths=list((WORK/'src/cadiff').rglob('*.py'))+[WORK/'tools/v06_r/train_scratch_pilot.py',WORK/'tools/v06_r/run_scratch_pilot.py']
    paths.extend([RUN/'generation_plan.json',RUN/'cpu_engineering_tests.json',RUN/'inspection/audit_completed_pilot.py',RUN/'inspection/cpu_engineering_tests.py',RUN/'inspection/science_implementation_review.json'])
    fresh(RUN/'execution_manifest.json',{'schema_version':'scratch_pilot_execution_manifest_v1','created_utc':now(),'protocol_sha256':sha256_file(RUN/'protocol.json'),'cpu_checks_sha256':sha256_file(RUN/'cpu_checks.json'),'files':{str(p):sha256_file(p) for p in sorted(set(paths))}})
    print(json.dumps({'frozen_sources':len(paths),'execution_manifest_sha256':sha256_file(RUN/'execution_manifest.json')}))

def verify(approved):
    p=RUN/'protocol.json'
    if approved!=sha256_file(p): raise ValueError('explicit reviewed protocol hash required')
    frozen=read(RUN/'execution_manifest.json')
    if frozen['protocol_sha256']!=approved: raise ValueError('execution manifest protocol differs')
    if frozen['cpu_checks_sha256']!=sha256_file(RUN/'cpu_checks.json'): raise ValueError('CPU checks changed')
    check_hashes(frozen['files']); protocol=read(p); check_hashes(protocol['frozen_files']); return protocol

def load_renderer(checkpoint,cat,view,smoke):
    from diffusers import AutoencoderKL
    from cadiff.v06_r.scratch_pilot_inference import FloatRenderer
    from cadiff.v06_r.runner import load_frozen_components,encode_prompt,P2InfillNoisePredictor
    from cadiff.v06_c.visa_exact_mask_r import CHECKPOINT_SCHEMA
    from cadiff.v06_c.can_exact_mask_r import load_can_r_modules
    data=read(checkpoint/'metadata.json'); config=load_pilot_config(RUN/'protocol.json',cat,smoke=smoke,steps=4 if smoke else 1200)
    if data['config']!=json.loads(json.dumps(asdict(config))) or data['completed_steps']!=config.steps or data['identity']['view']!=view or data['identity']['category']!=cat or data['identity']['smoke']!=smoke or data['identity']['protocol_sha256']!=sha256_file(RUN/'protocol.json'): raise ValueError('final checkpoint identity differs')
    for name,entry in data['files'].items():
        if sha256_file(checkpoint/name)!=entry['sha256']: raise ValueError('checkpoint changed')
    device=torch.device('cuda:0'); protocol=read(RUN/'protocol.json')
    components=load_frozen_components(Path(protocol['model_root']),device,torch.float16)
    vae=AutoencoderKL.from_pretrained(protocol['model_root'],subfolder='vae',local_files_only=True,torch_dtype=torch.float32).to(device=device,dtype=torch.float32).eval().requires_grad_(False)
    components=replace(components,vae=vae)
    _,conditioner,adapter,_=load_can_r_modules(checkpoint,config=config,device=device,expected_schema=CHECKPOINT_SCHEMA)
    model=P2InfillNoisePredictor(components.unet,conditioner,adapter).eval()
    prompt=encode_prompt(components,config.normal_prompt,device).detach().clone()
    return FloatRenderer(components=components,model=model,prompt=prompt,device=device,allowed_types=config.defect_types)

def generate(cat,view,smoke,approved):
    from cadiff.v06_r.full_crop_inference import render_physical
    verify(approved)
    base=RUN/'smoke' if smoke else RUN
    checkpoint=base/'training'/cat/view/('checkpoint-step-000004' if smoke else 'checkpoint-step-001200')
    output=base/'generation'/cat/view
    if output.exists(): raise FileExistsError('generation stage already exists')
    output.mkdir(parents=True)
    renderer=load_renderer(checkpoint,cat,view,smoke)
    inputs=[r for r in read(RUN/'inputs_manifest.json') if r['category']==cat]
    if smoke: inputs=inputs[:1]
    seeds=SEEDS[:1] if smoke else SEEDS
    records=[]
    for row in inputs:
        for seed in seeds:
            folder=output/(row['row_id']+'_seed-'+str(seed)); folder.mkdir()
            identity={'category':cat,'view':view,'row_id':row['row_id'],'inference_seed':seed,'renderer_seed':seed,'smoke':smoke,'checkpoint':str(checkpoint),'checkpoint_metadata_sha256':sha256_file(checkpoint/'metadata.json'),'normal_path':row['normal_path'],'mask_path':row['mask_path'],'requested_type':row['requested_type'],'started_utc':now()}
            fresh(folder/'attempt.json',identity)
            try:
                normal=np.asarray(Image.open(row['normal_path']).convert('RGB')).copy(); mask=np.asarray(Image.open(row['mask_path']).convert('L'))>0
                args={'valid_rectangle':row['target_valid_rectangle512'],'defect_type':row['requested_type'],'renderer_seed':seed}
                raw,final,geometry=render_physical(renderer,normal,mask,args,view)
                if not np.array_equal(final[~mask],normal[~mask]): raise ValueError('H exterior changed')
                Image.fromarray(raw).save(folder/'raw.png'); Image.fromarray(final).save(folder/'image.png')
                with (folder/'H.png').open('xb') as f: f.write(Path(row['mask_path']).read_bytes())
                record={**identity,'completed_utc':now(),'status':'success','image_path':str(folder/'image.png'),'raw_path':str(folder/'raw.png'),'output_H_path':str(folder/'H.png'),'image_sha256':sha256_file(folder/'image.png'),'raw_sha256':sha256_file(folder/'raw.png'),'output_H_sha256':sha256_file(folder/'H.png'),'outside_H_changed_pixels':0,'inside_H_changed_pixels':int(np.any(final!=normal,axis=2)[mask].sum()),'H_area':int(mask.sum()),'geometry':asdict(geometry) if geometry else None}
                fresh(folder/'record.json',record); records.append(record)
            except BaseException as error:
                fresh(folder/'failure.json',{**identity,'failed_utc':now(),'error_type':type(error).__name__,'error':str(error)}); raise
    fresh(output/'records.json',records)
    fresh(output/'summary.json',{'status':'complete','count':len(records),'failed':0,'records_sha256':sha256_file(output/'records.json'),'smoke':smoke})
    print(json.dumps({'generation_stage':cat+'/'+view,'complete':len(records),'smoke':smoke}))

def audit_pairs(base,steps):
    result={}
    for cat in ('cashew','pcb1'):
        roots=[base/'training'/cat/v for v in VIEWS]
        initial=[read(p/'initial_state_sha256.json') for p in roots]
        if initial[0]!=initial[1]: raise ValueError('Full/Crop initialization differs')
        rows=[[json.loads(x) for x in (p/'training.jsonl').read_text().splitlines()] for p in roots]
        keys=('step','sample_id','defect_type','timestep','context_seed','noise_sha256')
        if any(len(r)!=steps for r in rows): raise ValueError('successful-step count differs')
        if any(tuple(a[k] for k in keys)!=tuple(b[k] for k in keys) for a,b in zip(*rows)): raise ValueError('paired schedule/noise differs')
        result[cat]={'initial_state_equal':True,'paired_successful_updates':steps,'paired_fields':list(keys),'negative_labels':['normal'],'internal_amp_scale_backoffs':{v:sum(r['amp_internal_scale_backoffs'] for r in rs) for v,rs in zip(VIEWS,rows)}}
    fresh(base/'training_audit.json',result)
    return result

def controller(smoke,approved):
    protocol=verify(approved); base=RUN/'smoke' if smoke else RUN
    if not smoke:
        ready=read(RUN/'smoke_ready.json')
        if ready['protocol_sha256']!=approved or ready['status']!='passed': raise ValueError('smoke gate not passed')
        for name,digest in ready['evidence_sha256'].items():
            if sha256_file(Path(name))!=digest: raise ValueError('smoke evidence changed')
    if smoke: base.mkdir(exist_ok=False)
    lock=(RUN/'controller.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    statusfile=base/'group_state.json'; stage='startup'
    def update(status,**extra): state(statusfile,{'status':status,'stage':stage,'updated_utc':now(),'pid':os.getpid(),'smoke':smoke,'protocol_sha256':approved,**extra})
    update('running')
    env=os.environ.copy(); env['PYTHONPATH']=str(WORK/'src'); env['CUBLAS_WORKSPACE_CONFIG']=':4096:8'; env['TOKENIZERS_PARALLELISM']='false'
    try:
        for cat in ('cashew','pcb1'):
            for view in VIEWS:
                verify(approved); stage='train/'+cat+'/'+view; update('running')
                cmd=[str(PY),str(WORK/'tools/v06_r/train_scratch_pilot.py'),'--category',cat,'--view',view,'--protocol',str(RUN/'protocol.json'),'--paths-yaml',protocol['paths_yaml'],'--manifest',protocol['objects'][cat]['manifest'],'--dataset-root',protocol['data_root'],'--output',str(base/'training'/cat/view)]
                if smoke: cmd+=['--smoke','--steps','4']
                subprocess.run(cmd,cwd=WORK,env=env,check=True)
        audit_pairs(base,4 if smoke else 1200)
        for cat in ('cashew','pcb1'):
            for view in VIEWS:
                verify(approved); stage='generate/'+cat+'/'+view; update('running')
                cmd=[str(PY),str(WORK/'tools/v06_r/run_scratch_pilot.py'),'generate','--category',cat,'--view',view,'--reviewed-protocol-sha',approved]
                if smoke: cmd+=['--smoke']
                subprocess.run(cmd,cwd=WORK,env=env,check=True)
        records=[]
        for cat in ('cashew','pcb1'):
            for view in VIEWS: records+=read(base/'generation'/cat/view/'records.json')
        expected=4 if smoke else 20
        if len(records)!=expected: raise ValueError('generation budget incomplete')
        fresh(base/'records.json',records)
        if not smoke:
            stage='independent_cpu_completion_audit'; update('running')
            subprocess.run([str(PY),str(RUN/'inspection/audit_completed_pilot.py'),'--run',str(RUN),'--work',str(WORK)],cwd=WORK,env=env,check=True)
        fresh(base/'summary.json',{'status':'complete','completed_utc':now(),'training_arms':4,'successful_updates_each':4 if smoke else 1200,'generation_attempts':expected,'generation_successes':expected,'generation_failures':0,'detector_runs':0,'records_sha256':sha256_file(base/'records.json'),'smoke':smoke,'human_semantics_verified':False})
        if smoke:
            evidence=[base/'summary.json',base/'training_audit.json',base/'records.json']
            evidence.extend((base/'training').rglob('summary.json'))
            evidence.extend((base/'training').rglob('initial_state_sha256.json'))
            evidence.extend((base/'training').rglob('training.jsonl'))
            evidence.extend((base/'generation').rglob('record.json'))
            evidence.extend((base/'generation').rglob('*.png'))
            fresh(RUN/'smoke_ready.json',{'status':'passed','completed_utc':now(),'protocol_sha256':approved,'evidence_sha256':{str(p):sha256_file(p) for p in evidence},'formal_will_reinitialize':True})
        stage='complete'; update('complete')
    except BaseException as error:
        if not smoke:
            attempts=[]
            for slot in read(RUN/'generation_plan.json'):
                record=Path(slot['record_path']); failure=record.parent/'failure.json'
                status='success' if record.exists() else 'failed' if failure.exists() else 'interrupted_or_unknown' if (record.parent/'attempt.json').exists() else 'unattempted'
                attempts.append({**slot,'status':status})
            fresh(RUN/'all_attempts_status.json',attempts)
        update('failed',error_type=type(error).__name__,error=str(error)); raise

def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('mode',choices=('prepare','freeze','generate','controller'))
    p.add_argument('--category',choices=('cashew','pcb1')); p.add_argument('--view',choices=VIEWS); p.add_argument('--smoke',action='store_true'); p.add_argument('--reviewed-protocol-sha')
    a=p.parse_args()
    if a.mode=='prepare': prepare()
    elif a.mode=='freeze': freeze()
    elif a.mode=='generate': generate(a.category,a.view,a.smoke,a.reviewed_protocol_sha)
    else: controller(a.smoke,a.reviewed_protocol_sha)
if __name__=='__main__': main()
