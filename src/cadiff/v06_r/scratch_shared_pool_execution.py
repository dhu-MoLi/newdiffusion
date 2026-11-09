"""Reviewed shared-pool generation and fixed-budget scratch detectors, one GPU at a time."""
from __future__ import annotations
from dataclasses import asdict
import fcntl,json,os,subprocess,sys
from pathlib import Path
import statistics
import numpy as np
from PIL import Image
import torch
from cadiff.v06_r.scratch_shared_pool import WORK,ROOT,RUN,PILOT,read,new,now
from cadiff.provenance import sha256_file

PY=ROOT/'.conda/envs/cad-v0/bin/python'
VIEWS=('full','crop32')

def check(files):
    for path,digest in files.items():
        if sha256_file(Path(path))!=digest:raise ValueError('frozen file changed: '+path)

def eligibility():
    data=read(RUN/'human_eligibility.json');candidates=read(RUN/'candidate_manifest.json')['rows']
    if data.get('candidate_manifest_sha256')!=sha256_file(RUN/'candidate_manifest.json') or data.get('reviewer_kind')!='human':raise ValueError('binding independent human eligibility is required')
    rows=data['rows'];ids=[r['review_id'] for r in rows]
    if len(rows)!=40 or len(set(ids))!=40 or set(ids)!={r['review_id'] for r in candidates}:raise ValueError('human review must retain exact40 candidates')
    decisions={r['review_id']:r for r in rows}
    if any(r.get('decision') not in ('pass','reject','uncertain') for r in rows):raise ValueError('human review requires three-value decisions')
    for row in candidates:
        if decisions[row['review_id']]['decision']=='pass' and not row['numeric_canvas_valid']:raise ValueError('human-pass H is outside valid canvas')
    return decisions

def freeze():
    if (RUN/'protocol.json').exists():raise FileExistsError('protocol already frozen')
    decisions=eligibility();pre=read(RUN/'pre_inspection_freeze.json');check(pre['files'])
    recipe=read(RUN/'downstream/accepted_sampling/recipe.json')
    if recipe['total_updates']!=6400 or recipe['seeds']!=[2026101601,2026101602,2026101603]:raise ValueError('detector recipe differs')
    accepted=read(RUN/'downstream/accepted_sampling/accepted_slots.json')
    active=[cat for cat in ('cashew','pcb1') if accepted[cat]]
    protocol={'schema':'scratch_shared_pool_execution_accepted_domain_v2','registered_utc':now(),'active_objects':active,'NA_objects':[c for c in ('cashew','pcb1') if c not in active],'candidate_manifest_sha256':sha256_file(RUN/'candidate_manifest.json'),'pre_inspection_freeze_sha256':sha256_file(RUN/'pre_inspection_freeze.json'),'human_eligibility_sha256':sha256_file(RUN/'human_eligibility.json'),'recipe_sha256':sha256_file(RUN/'downstream/accepted_sampling/recipe.json'),'renderer_contract_sha256':sha256_file(RUN/'frozen_renderers.json'),'selection':'original40 candidate pairs retained for eligibility yield; all human-pass pairs times both seeds form accepted domain; reject/uncertain not generated, no refill','inference_seeds':[2026101501,2026101502],'maximum_generation_attempts':160,'actual_eligible_generation_attempts':4*sum(r['decision']=='pass' for r in decisions.values()),'generator_training_runs':0,'detector_runs':9*len(active),'generator_checkpoints':'only four frozen pilot final1200; no update or loading other R','precision':'fresh original FP32 VAE encode/decode, latent/UNet FP16, original conditioner/adapter FP32; unchanged scratch-pilot renderer','generation_failure':'Any accepted slot operational/nonfinite/H-exterior or H/checkpoint identity violation records attempt and stops controller; no shrinking pool/fallback/refill/precision change; same-slot same-seed engineering recovery only after explicit registration. Nonzero other object may continue under Root registered control.','availability':'Every accepted pair-seed must succeed in both views before object D. All accepted slots retained in balanced seeded cyclic extra stream. Original nominal40/object retained only for qualification and engineering yield. Zero-pass object three detector conditions NA, no detector execution.','gross_semantics':'Engineering integrity does not establish visible G/H, type or realism. Gross wrong-part/type/artifact patterns pause before detectors for direct review; no appearance-based posthoc sample rejection to improve AP.','detector_gate':'After paired generation Root/Astra direct review binding generation_records+availability SHA. Detector smoke and subsequent production are separately launched by Root; no additional human G labeling prerequisite.','compute_effect':'All3 conditions identical6400 updates, normal50000 and common base-real25000 slots. RealFS extra25000 real replay; Full/Crop extra25000 accepted-domain generated presentations, no fallback. This is a controlled real-to-gen mixture intervention, not unchanged real exposure.','evaluation':'scratch-only test identities frozen before generation; single fixed-final checkpoint cache after epoch200; all resized GT areas and zero-positive masks retained/audited, no tiny-mask exclusion or test-based selection; no old multi-type RealFS reuse','smoke':'Each active object3conditions, first fixed seed, one successful16-example batch each; no test, separate outputs, formal fresh initialization; bind resume.pt and validate actual optimizer state before production','no_automatic_report':True}
    protocol['qualification_authority']='Human alone assigns pass/reject/uncertain. Root/Astra only precheck and faithfully record user decisions; this supersedes ambiguous Root/Astra wording in the preserved selection v1 snapshot.'
    protocol['F1_reporting']='Existing helper test-threshold scans are descriptive oracle F1, never model/setting/progression selection. Pooled pixel AP is primary.'
    new(RUN/'protocol.json',protocol)
    files=dict(pre['files'])
    pilot_protocol=read(PILOT/'protocol.json')
    base_files={p:h for p,h in pilot_protocol['frozen_files'].items() if p.startswith(pilot_protocol['model_root']+'/')}
    if not base_files:raise ValueError('frozen SD1.5 component hashes missing')
    check(base_files);files.update(base_files)
    for p in [RUN/'human_eligibility.json',RUN/'pre_inspection_freeze.json',RUN/'downstream/accepted_sampling/recipe.json',RUN/'downstream/accepted_sampling/accepted_slots.json',RUN/'downstream/assets.json',RUN/'downstream/accepted_sampling/plan_manifest.json',RUN/'downstream/accepted_sampling/cpu_sampling_checks.json',RUN/'downstream/asset_source_sha256.json',RUN/'protocol.json']:
        files[str(p)]=sha256_file(p)
    for entry in read(RUN/'downstream/accepted_sampling/plan_manifest.json').values():
        for record in entry.values():files[record['path']]=record['sha256']
    for p in (WORK/'src/cadiff').rglob('*.py'):files[str(p)]=sha256_file(p)
    for p in (ROOT/'eval_kit').rglob('*.py'):files[str(p)]=sha256_file(p)
    for p in (WORK/'tools/v06_r').glob('*scratch_shared_pool*.py'):files[str(p)]=sha256_file(p)
    for p in (RUN/'inspection').glob('*scientific*contract*.json'):files[str(p)]=sha256_file(p)
    for p in (RUN/'inspection').glob('*cpu*'):files[str(p)]=sha256_file(p)
    new(RUN/'execution_manifest.json',{'schema':'scratch_shared_pool_execution_manifest_v1','created_utc':now(),'protocol_sha256':sha256_file(RUN/'protocol.json'),'files':files})
    print(json.dumps({'protocol_sha256':sha256_file(RUN/'protocol.json'),'execution_manifest_sha256':sha256_file(RUN/'execution_manifest.json'),'actual_generation_attempts':protocol['actual_eligible_generation_attempts']}))

def verify(approved,require_generation=False,require_output_review=False):
    if approved!=sha256_file(RUN/'protocol.json'):raise ValueError('explicit reviewed protocol hash required')
    manifest=read(RUN/'execution_manifest.json')
    if manifest['protocol_sha256']!=approved:raise ValueError('execution manifest protocol differs')
    check(manifest['files']);check(read(RUN/'downstream/asset_source_sha256.json'));eligibility()
    if require_generation:
        summary=read(RUN/'paired_availability.json')
        if summary['generation_records_sha256']!=sha256_file(RUN/'generation_records.json'):raise ValueError('paired-generation records changed')
        check(summary['artifact_sha256'])
        accepted=read(RUN/'downstream/accepted_sampling/accepted_slots.json')
        for cat in ('cashew','pcb1'):
            count=len(accepted[cat]);counts=summary['counts'][cat]
            if counts['paired_available']!=count or counts['full_success']!=count or counts['crop32_success']!=count:raise ValueError('accepted object pool incomplete; pause, no fallback')
    if require_output_review:
        review=read(RUN/'direct_output_review.json')
        if review.get('go_downstream') is not True or review['generation_records_sha256']!=sha256_file(RUN/'generation_records.json') or review['paired_availability_sha256']!=sha256_file(RUN/'paired_availability.json'):raise ValueError('direct output review required before detector')
    return read(RUN/'protocol.json')

def generate(cat,view,approved):
    from cadiff.v06_r.scratch_pilot import load_renderer
    from cadiff.v06_r.full_crop_inference import render_physical
    verify(approved)
    if cat not in ('cashew','pcb1') or view not in VIEWS:raise ValueError('unknown generation object/view')
    output=RUN/'generation'/cat/view
    if output.exists():raise FileExistsError('generation stage exists')
    output.mkdir(parents=True)
    contract=read(RUN/'frozen_renderers.json')[cat][view];checkpoint=Path(contract['path'])
    decisions=eligibility();candidates=[r for r in read(RUN/'candidate_manifest.json')['rows'] if r['category']==cat]
    renderer=load_renderer(checkpoint,cat,view,False) if any(decisions[r['review_id']]['decision']=='pass' for r in candidates) else None
    slots=read(RUN/'downstream/assets.json')[cat]['nominal_extra_slots'];records=[]
    for slot in slots:
        row=next(r for r in candidates if r['identity']==slot['identity']);decision=decisions[row['review_id']]['decision'];seed=slot['inference_seed']
        identity={**slot,'category':cat,'view':view,'human_decision':decision,'normal_path':row['normal_path'],'normal_sha256':row['normal_sha256'],'input_H_path':row['H_path'],'input_H_sha256':row['H_sha256'],'normal_native_path':row['native_path'],'source_image_path':row['source_image_path'],'source_image_sha256':row['source_image_sha256'],'source_mask_path':row['source_mask_path'],'source_mask_sha256':row['source_mask_sha256'],'requested_type':row['requested_type'],'checkpoint':str(checkpoint),'checkpoint_metadata_sha256':contract['metadata_sha256'],'protocol_sha256':approved}
        folder=output/(row['review_id']+'_seed-'+str(seed));folder.mkdir()
        if decision!='pass':
            record={**identity,'status':'ineligible_'+decision,'attempted':False};new(folder/'record.json',record);records.append(record);continue
        new(folder/'attempt.json',{**identity,'started_utc':now()})
        try:
            with Image.open(row['normal_path']) as im:normal=np.asarray(im.convert('RGB')).copy()
            with Image.open(row['H_path']) as im:mask=np.asarray(im.convert('L'))>0
            g=row['geometry'];rect=[g['pad_left'],g['pad_top'],g['pad_left']+g['resized_width'],g['pad_top']+g['resized_height']]
            raw,final,geometry=render_physical(renderer,normal,mask,{'valid_rectangle':rect,'defect_type':row['requested_type'],'renderer_seed':seed},view)
            if not np.array_equal(final[~mask],normal[~mask]) or not np.any(final):raise ValueError('invalid physical output/exterior')
            Image.fromarray(raw).save(folder/'raw.png');Image.fromarray(final).save(folder/'image.png')
            with (folder/'H.png').open('xb') as f:f.write(Path(row['H_path']).read_bytes())
            record={**identity,'status':'success','attempted':True,'completed_utc':now(),'image_path':str(folder/'image.png'),'image_sha256':sha256_file(folder/'image.png'),'raw_path':str(folder/'raw.png'),'raw_sha256':sha256_file(folder/'raw.png'),'output_H_path':str(folder/'H.png'),'output_H_sha256':sha256_file(folder/'H.png'),'valid_rectangle':rect,'geometry':asdict(geometry) if geometry else None,'H_area':int(mask.sum()),'outside_H_changed_pixels':0,'inside_H_changed_pixels':int(np.any(final!=normal,axis=2)[mask].sum()),'visible_G_and_realism':'unjudged'}
            new(folder/'record.json',record);records.append(record)
        except BaseException as error:
            new(folder/'failure.json',{**identity,'status':'failed','attempted':True,'failed_utc':now(),'error_type':type(error).__name__,'error':str(error)});raise
    new(output/'records.json',records);new(output/'summary.json',{'status':'complete','nominal_slots':40,'success':sum(r['status']=='success' for r in records),'ineligible':sum(not r['attempted'] for r in records),'failed':0,'records_sha256':sha256_file(output/'records.json')})
    print(json.dumps({'generation_stage':cat+'/'+view,'success':sum(r['status']=='success' for r in records),'nominal_slots':40}))

def collect():
    records=[]
    for cat in ('cashew','pcb1'):
        for view in VIEWS:records+=read(RUN/'generation'/cat/view/'records.json')
    if len(records)!=160:raise ValueError('missing nominal generation rows')
    artifacts={};objects={};counts={}
    for cat in ('cashew','pcb1'):
        rows=[]
        for nominal in range(40):
            pair=[next(r for r in records if r['category']==cat and r['view']==view and r['nominal_index']==nominal) for view in VIEWS]
            available=all(r['status']=='success' for r in pair)
            for row in pair:
                if row['status']!='success':continue
                for kind in ('image','raw','output_H'):
                    path=row[kind+'_path'];digest=row[kind+'_sha256']
                    if sha256_file(Path(path))!=digest:raise ValueError('output artifact changed')
                    artifacts[path]=digest
                if row['output_H_sha256']!=row['input_H_sha256']:raise ValueError('output H differs')
                with Image.open(row['image_path']) as im:final=np.asarray(im.convert('RGB'))
                with Image.open(row['normal_path']) as im:normal=np.asarray(im.convert('RGB'))
                with Image.open(row['input_H_path']) as im:mask=np.asarray(im.convert('L'))>0
                if not np.array_equal(final[~mask],normal[~mask]):raise ValueError('H-exterior composition violated')
            if pair[0]['human_decision']=='pass' and not available:raise ValueError('accepted slot incomplete; pause, no fallback')
            rows.append({'nominal_index':nominal,'identity':pair[0]['identity'],'review_id':pair[0]['review_id'],'inference_seed':pair[0]['inference_seed'],'full_status':pair[0]['status'],'crop32_status':pair[1]['status'],'paired_available':available,'accepted_domain_status':'included' if available else 'human_reject_or_uncertain_excluded_before_generation'})
        objects[cat]=rows;counts[cat]={'nominal_pair_seed_slots':40,'paired_available':sum(r['paired_available'] for r in rows),'full_success':sum(r['status']=='success' for r in records if r['category']==cat and r['view']=='full'),'crop32_success':sum(r['status']=='success' for r in records if r['category']==cat and r['view']=='crop32')}
    new(RUN/'generation_records.json',records)
    new(RUN/'paired_availability.json',{'schema':'scratch_shared_pool_paired_availability_v1','created_utc':now(),'generation_records_sha256':sha256_file(RUN/'generation_records.json'),'objects':objects,'counts':counts,'artifact_sha256':artifacts,'semantics':'engineering eligibility only; direct gross-failure review required before detectors'})
    print(json.dumps({'paired_availability':counts,'detectors_started':0}))

def atomic(path,data):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n');temp.replace(path)

def aggregate():
    from cadiff.v06_r.scratch_shared_pool_downstream import SEEDS,CONDITIONS
    results={};summary={}
    active=read(RUN/'protocol.json')['active_objects']
    for cat in ('cashew','pcb1'):
        if cat not in active:
            summary[cat]={'status':'NA_zero_human_pass','conditions':{c:'NA_not_run' for c in CONDITIONS}};continue
        results[cat]={cond:[read(RUN/'downstream/runs'/cat/cond/f'seed-{seed}'/'results.json') for seed in SEEDS] for cond in CONDITIONS}
        summary[cat]={'conditions':{},'paired_differences_percentage_points':{}}
        for cond,rows in results[cat].items():
            for row in rows:
                if row['actual_optimizer_steps']!=6400 or row['test_forward_count']!=(60 if cat=='cashew' else 110):raise ValueError('detector completion contract differs')
            metrics={key:{'per_seed':[r['metrics']['overall'][key] for r in rows],'mean':statistics.mean(r['metrics']['overall'][key] for r in rows),'sample_std':statistics.stdev(r['metrics']['overall'][key] for r in rows)} for key in rows[0]['metrics']['overall'] if key!='image_count'}
            summary[cat]['conditions'][cond]={'metrics':metrics,'source_exposures':[r['source_exposures'] for r in rows],'test_images':rows[0]['test_forward_count']}
        for a,b in [('full','real_fs'),('crop32','real_fs'),('crop32','full')]:
            delta=[100*(ra['metrics']['overall']['ap_pixel']-rb['metrics']['overall']['ap_pixel']) for ra,rb in zip(results[cat][a],results[cat][b])]
            summary[cat]['paired_differences_percentage_points'][a+'-minus-'+b]={'per_seed':delta,'mean':statistics.mean(delta),'sample_std':statistics.stdev(delta),'regressed_seed_count':sum(d<0 for d in delta)}
    new(RUN/'downstream/aggregate.json',{'schema':'scratch_shared_pool_detector_aggregate_v1','created_utc':now(),'seeds':list(SEEDS),'summary':summary,'all_results':results,'interpretation':'Paired detector initialization repeats nested under one fixed generator-training seed per view/object; exploratory restricted scratch task, no full-object/generalization or formal semantic-rate claim.'})

def controller(mode,approved):
    if mode not in ('generation','detector_smoke','detectors'):raise ValueError('invalid controller mode')
    protocol=verify(approved,require_generation=mode!='generation',require_output_review=mode!='generation')
    if mode=='detectors':
        ready=read(RUN/'detector_smoke_ready.json')
        if ready['protocol_sha256']!=approved or ready['status']!='passed':raise ValueError('detector smoke gate missing')
        check(ready['evidence_sha256'])
    lock=(RUN/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    env=os.environ.copy();env['PYTHONPATH']=str(WORK/'src');env['CUBLAS_WORKSPACE_CONFIG']=':4096:8';env['TOKENIZERS_PARALLELISM']='false'
    statusfile=RUN/(mode+'_state.json');stage='startup'
    def update(status,**extra):atomic(statusfile,{'mode':mode,'stage':stage,'status':status,'updated_utc':now(),'pid':os.getpid(),'protocol_sha256':approved,**extra})
    update('running')
    try:
        if mode=='generation':
            for cat in ('cashew','pcb1'):
                for view in VIEWS:
                    stage=cat+'/'+view;update('running')
                    subprocess.run([str(PY),str(WORK/'tools/v06_r/run_scratch_shared_pool_execution.py'),'generate','--category',cat,'--view',view,'--reviewed-protocol-sha',approved],cwd=WORK,env=env,check=True)
            stage='paired_integrity_audit';update('running');collect()
        else:
            from cadiff.v06_r.scratch_shared_pool_downstream import SEEDS,CONDITIONS
            seeds=SEEDS[:1] if mode=='detector_smoke' else SEEDS
            for cat in protocol['active_objects']:
                for seed in seeds:
                    for cond in CONDITIONS:
                        stage=cat+'/'+str(seed)+'/'+cond;update('running')
                        cmd=[str(PY),str(WORK/'tools/v06_r/run_scratch_shared_pool_downstream.py'),'train','--category',cat,'--condition',cond,'--seed',str(seed),'--reviewed-protocol-sha',approved]
                        if mode=='detector_smoke':cmd+=['--smoke']
                        subprocess.run(cmd,cwd=WORK,env=env,check=True)
            for cat in protocol['active_objects']:
                for seed in seeds:
                    base=RUN/'detector_smoke' if mode=='detector_smoke' else RUN/'downstream/runs'
                    initial=[read(base/cat/cond/f'seed-{seed}'/'initial_state_sha256.json') for cond in CONDITIONS]
                    if initial[0]!=initial[1] or initial[0]!=initial[2]:raise ValueError('paired detector initialization differs')
            if mode=='detector_smoke':
                audit=[]
                for cat in protocol['active_objects']:
                    for cond in CONDITIONS:
                        path=RUN/'detector_smoke'/cat/cond/f'seed-{seeds[0]}'/'resume.pt'
                        checkpoint=torch.load(path,map_location='cpu',weights_only=False)
                        if checkpoint['steps']!=1 or checkpoint['epoch']!=1 or checkpoint['contract']['protocol_sha256']!=approved or checkpoint['contract']['smoke'] is not True:raise ValueError('actual smoke checkpoint budget differs')
                        values=checkpoint['model'].values()
                        if not all(torch.isfinite(v).all() for v in values):raise ValueError('actual smoke model has nonfinite state')
                        optimizer_states=list(checkpoint['optimizer']['state'].values())
                        if not optimizer_states or any(int(s['step'])!=1 for s in optimizer_states):raise ValueError('actual smoke Adam step differs')
                        audit.append({'category':cat,'condition':cond,'seed':seeds[0],'actual_optimizer_steps':1,'optimizer_state_parameter_count':len(optimizer_states),'all_model_state_finite':True,'resume_sha256':sha256_file(path),'test_forward_count':0})
                        del checkpoint
                new(RUN/'detector_smoke_audit.json',{'schema':'scratch_shared_actual_smoke_audit_v1','rows':audit,'formal_reinitializes':True})
                paths=list((RUN/'detector_smoke').rglob('*.json'))+list((RUN/'detector_smoke').rglob('resume.pt'))
                paths.append(RUN/'detector_smoke_audit.json')
                new(RUN/'detector_smoke_ready.json',{'status':'passed','completed_utc':now(),'protocol_sha256':approved,'evidence_sha256':{str(p):sha256_file(p) for p in paths},'formal_reinitializes':True,'test_calls':0})
            else:aggregate()
        stage='complete';update('complete')
    except BaseException as error:
        if mode=='generation':
            snapshot=[];decisions=eligibility()
            for cat in ('cashew','pcb1'):
                slots=read(RUN/'downstream/assets.json')[cat]['nominal_extra_slots']
                for view in VIEWS:
                    for slot in slots:
                        folder=RUN/'generation'/cat/view/(slot['review_id']+'_seed-'+str(slot['inference_seed']))
                        record=folder/'record.json';failure=folder/'failure.json';attempt=folder/'attempt.json'
                        decision=decisions[slot['review_id']]['decision']
                        status=read(record)['status'] if record.exists() else 'failed' if failure.exists() else 'interrupted_or_unknown' if attempt.exists() else 'ineligible_'+decision if decision!='pass' else 'unattempted'
                        snapshot.append({**slot,'category':cat,'view':view,'status':status})
            new(RUN/'generation_attempts_failure_snapshot.json',snapshot)
        update('failed',error_type=type(error).__name__,error=str(error));raise

def main():
    import argparse
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['freeze','generate','collect','controller']);p.add_argument('--stage',choices=['generation','detector_smoke','detectors']);p.add_argument('--category',choices=['cashew','pcb1']);p.add_argument('--view',choices=VIEWS);p.add_argument('--reviewed-protocol-sha');a=p.parse_args()
    if a.mode=='freeze':freeze()
    elif a.mode=='generate':generate(a.category,a.view,a.reviewed_protocol_sha)
    elif a.mode=='collect':verify(a.reviewed_protocol_sha);collect()
    else:controller(a.stage,a.reviewed_protocol_sha)
if __name__=='__main__':main()
