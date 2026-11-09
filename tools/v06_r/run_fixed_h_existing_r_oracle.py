"""Small fixed-H conditional-R probe. Preparation is CPU-only; execution requires review SHA."""
from __future__ import annotations
import argparse, gc, hashlib, json, os, sys, time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from PIL import Image

SELECTED=('C2','C3','C4','P3','P4')
SEEDS=(2026101501,2026101502)
def now():return datetime.now(timezone.utc).isoformat()
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def write(p,x):
 p=Path(p);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(x,ensure_ascii=False,indent=2,sort_keys=True)+'\n');tmp.replace(p)
def load(p):return json.loads(Path(p).read_text())
def binary_sha(b):return hashlib.sha256(np.packbits(b).tobytes()).hexdigest()
def rgb_sha(a):return hashlib.sha256(a.tobytes()).hexdigest()
def prepare_pixels(identity,root):
 from cadiff.data.geometry import resize_pad_image,resize_pad_mask
 assert sha(identity['source_path'])==identity['image_sha256']
 with Image.open(identity['source_path']) as im:rgb,valid,meta=resize_pad_image(im,512)
 result=np.asarray(rgb).copy();mask=None
 if identity.get('mask_relpath'):
  mp=root/'data/VisA'/identity['mask_relpath'];assert sha(mp)==identity['native_mask']['file_sha256']
  with Image.open(mp) as im:prepared=Image.fromarray((np.asarray(im.convert('L'))>0).astype(np.uint8)*255)
  if prepared.size!=(meta.source_width,meta.source_height):prepared=prepared.resize((meta.source_width,meta.source_height),Image.Resampling.NEAREST)
  mask=np.asarray(resize_pad_mask(prepared,meta))>0
 return result,np.asarray(valid)>0,mask
def cpu_contract_checks(checkpoints,pair_inputs):
 import torch
 from cadiff.v06_c.visa_exact_mask_r import visa_p2_a1_config,CHECKPOINT_SCHEMA
 from cadiff.v06_c.can_exact_mask_r import load_can_r_modules
 from cadiff.diffusion.vae_roundtrip import encode_mode,decode_latent
 from cadiff.p2_a0.masking import build_latent_masks
 details=[]
 for item in checkpoints:
  meta=load(item['metadata_path']);config=visa_p2_a1_config(item['category'],tuple(meta['config']['defect_types']))
  assert json.loads(json.dumps(asdict(config)))==meta['config']
  vocabulary,conditioner,adapter,_=load_can_r_modules(Path(item['checkpoint']),config=config,device=torch.device('cpu'),expected_schema=CHECKPOINT_SCHEMA)
  route=vocabulary.index(item['requested_type']);assert route==item['route_index']
  assert all(p.dtype==torch.float32 for p in (*conditioner.parameters(),*adapter.parameters()))
  details.append({'category':item['category'],'checkpoint_config_exact':True,'vocabulary_sha_matches':vocabulary.sha256==meta['vocabulary_sha256'],'route_index':route,'adapter_conditioner_state_dict_loaded_cpu':True,'master_dtype':'float32'})
  del conditioner,adapter;gc.collect()
 for inp in pair_inputs:
  h=np.asarray(Image.open(inp['mask_path']))>0;normal=np.asarray(Image.open(inp['normal_path']).convert('RGB'))
  assert h.shape==(512,512) and int(h.sum())==inp['physical512_area'] and binary_sha(h)==inp['prepared512_binary_sha256']
  assert normal.shape==(512,512,3) and rgb_sha(normal)==inp['normal_pixels_sha256']
  lm=build_latent_masks(torch.from_numpy(h.copy())[None,None].float(),(64,64)).binary
  assert lm.shape==(1,1,64,64) and lm.any() and not lm.all()
  inp['latent64_binary_pixels']=int(lm.sum().item())
 class FakeVAE:
  config=type('Config',(),{'scaling_factor':.18215})()
  def __init__(self,bad_encode=False,bad_decode=False):self.bad_encode,self.bad_decode=bad_encode,bad_decode
  def encode(self,pixels):
   v=torch.zeros((1,4,64,64));v[0,0,0,0]=float('nan') if self.bad_encode else 0
   return type('Encoded',(),{'latent_dist':type('Posterior',(),{'mode':lambda s:v})()})()
  def decode(self,latent):
   v=torch.zeros((1,3,512,512));v[0,0,0,0]=float('nan') if self.bad_decode else 0
   return type('Decoded',(),{'sample':v})()
 rejected=[]
 for name,fn in [('encode_mode_nan',lambda:encode_mode(FakeVAE(bad_encode=True),torch.zeros((1,3,512,512)))),('decode_latent_prequant_nan',lambda:decode_latent(FakeVAE(bad_decode=True),torch.zeros((1,4,64,64))))]:
  try:fn()
  except ValueError:rejected.append(name)
  else:raise AssertionError('nonfinite CPU helper contract did not reject '+name)
 assert not torch.cuda.is_initialized(),'CPU preparation initialized CUDA'
 return {'state':'passed','checkpoint_checks':details,'fixed_pair_geometry_mask_normal_checks':len(pair_inputs),'nonfinite_original_helper_checks':rejected,'cuda_initialized':False,'base_model_loaded':False,'heldout_image_or_mask_read':False}
def prepare(args):
 out=args.output;out.mkdir(parents=True,exist_ok=False);started=now();write(out/'status.json',{'state':'preparing_cpu','started_utc':started,'gpu_inference_executed':0})
 review=load(args.human_review);card_path=args.audit/'oracle_readiness/human_compatibility_review/manifest.json';card=load(card_path)
 ids=[json.loads(s) for s in (args.audit/'identity_contract.jsonl').read_text().splitlines()]
 rows=card['rows'];assert len(rows)==8
 reasons={'C1':'conservative exclusion for strict isolation; user described mild body-exterior concern, not a human strong failure','P1':'human unsuitable position','P2':'human explicitly clarified unsuitable position'}
 retained=[];inputs=[];jobs=[]
 for row in rows:
  chosen=row['row_id'] in SELECTED
  retained.append({**row,'oracle_selection':'selected' if chosen else 'excluded','exclusion_reason':None if chosen else reasons[row['row_id']],'human_location_evidence':'reasonable position' if chosen else ('mild concern' if row['row_id']=='C1' else 'unreasonable position'),'human_semantic_type_verified':False,'human_evidence_path':str(args.human_review),'human_evidence_sha256':sha(args.human_review)})
  if not chosen:continue
  source=next(i for i in ids if i['category']==row['category'] and i['source_id']==row['source_shape_id'] and i['role']=='anomaly_support')
  normal=next(i for i in ids if i['category']==row['category'] and i['source_id']==row['normal_id'] and i['role']=='normal_train')
  nrgb,nvalid,_=prepare_pixels(normal,args.root);_,svalid,h=prepare_pixels(source,args.root)
  assert int(h.sum())==row['physical512_area']>=64 and binary_sha(h)==row['prepared512_binary_sha256']
  assert not (h&~svalid).any() and not (h&~nvalid).any()
  ip=out/'inputs'/row['row_id'];ip.mkdir(parents=True);Image.fromarray(nrgb).save(ip/'normal.png');Image.fromarray(h.astype(np.uint8)*255).save(ip/'H.png')
  inp={'row_id':row['row_id'],'category':row['category'],'requested_type':row['requested_type_name'],'source_shape_id':row['source_shape_id'],'target_normal_id':row['normal_id'],'physical512_area':int(h.sum()),'normal_path':str(ip/'normal.png'),'mask_path':str(ip/'H.png'),'normal_file_sha256':sha(ip/'normal.png'),'mask_file_sha256':sha(ip/'H.png'),'normal_pixels_sha256':rgb_sha(nrgb),'prepared512_binary_sha256':binary_sha(h),'target_valid_rectangle512':row['target_valid_rectangle512'],'source_image_path':source['source_path'],'source_image_sha256':source['image_sha256'],'source_mask_path':str(args.root/'data/VisA'/source['mask_relpath']),'source_mask_file_sha256':source['native_mask']['file_sha256'],'target_image_path':normal['source_path'],'target_image_sha256':normal['image_sha256']}
  inputs.append(inp)
  for seed in SEEDS:jobs.append({'job_id':f"{row['row_id']}-seed{seed}",'row_id':row['row_id'],'category':row['category'],'requested_type':row['requested_type_name'],'seed':seed,'context_fill_seed':seed+10000000})
 assert tuple(x['row_id'] for x in inputs)==SELECTED and len(jobs)==10 and len({j['job_id'] for j in jobs})==10
 checkpoints=[];training=[];files={str(card_path):sha(card_path),str(args.human_review):sha(args.human_review),str(args.audit/'identity_contract.jsonl'):sha(args.audit/'identity_contract.jsonl')}
 for c,tag,typ in [('cashew','20260922-seas','small scratches'),('pcb1','20260927-magic','scratch')]:
  cp=args.root/'runs/v06_b/visa'/c/'exact-mask-r'/('seed-'+tag)/'checkpoint';mp=cp/'metadata.json';meta=load(mp)
  assert meta['config']['category']==c and meta['config']['image_size']==512 and meta['config']['dtype']=='float16' and meta['config']['steps']==1200
  weights=[]
  for name in ['adapter.safetensors','conditioner.safetensors']:
   p=cp/name;actual=sha(p);assert actual==meta['files'][name]['sha256'] and p.stat().st_size==meta['files'][name]['size'];files[str(p)]=actual;weights.append({'path':str(p),'sha256':actual,'bytes':p.stat().st_size})
  files[str(mp)]=sha(mp);mf=args.root/'runs/v06_b/visa'/c/'splits'/tag/'manifest.jsonl';files[str(mf)]=sha(mf)
  tr=[json.loads(s) for s in mf.read_text().splitlines() if s.strip()];tr=sorted([x for x in tr if 'fewshot_train' in x.get('roles',[])],key=lambda x:x['image_relpath'])
  for index,x in enumerate(tr):
   for key in ['image','mask']:
    p=args.root/'data/VisA'/x[key+'_relpath'];actual=sha(p);assert actual==x[key+'_sha256'];files[str(p)]=actual
   training.append({'category':c,'prepared_train_index':index,'image_relpath':x['image_relpath'],'mask_relpath':x['mask_relpath'],'image_sha256':x['image_sha256'],'mask_sha256':x['mask_sha256'],'defect_type':x['defect_type'],'training_context_fill_seed':meta['config']['context_fill_seed']+index,'training_context_source':'FP16 posterior-mode latent of same real anomaly; hole=maxpooled same GT, hole context replaced with Gaussian noise'})
  checkpoints.append({'category':c,'checkpoint':str(cp),'metadata_path':str(mp),'metadata_sha256':sha(mp),'config':meta['config'],'vocabulary_sha256':meta['vocabulary_sha256'],'requested_type':typ,'route_index':meta['config']['defect_types'].index(typ),'weights':weights,'training_manifest':str(mf),'training_manifest_sha256':sha(mf),'training_support_count':len(tr),'extra_combined_training_count':sum(x['defect_type']=='combined' for x in tr)})
 # Lock the original model layout and every required file; no component is loaded here.
 from cadiff.diffusion.components import REQUIRED,validate_model_layout
 validate_model_layout(args.model_root)
 base=[]
 for rel in REQUIRED:
  p=args.model_root/rel;files[str(p)]=sha(p);base.append({'path':str(p),'sha256':files[str(p)],'bytes':p.stat().st_size})
 source_files=[Path(__file__)]+sorted((args.worktree/'src/cadiff').rglob('*.py'))
 for p in source_files:files[str(p)]=sha(p)
 for inp in inputs:
  for pk,sk in [('normal_path','normal_file_sha256'),('mask_path','mask_file_sha256'),('source_image_path','source_image_sha256'),('source_mask_path','source_mask_file_sha256'),('target_image_path','target_image_sha256')]:files[inp[pk]]=inp[sk]
 cpu=cpu_contract_checks(checkpoints,inputs)
 write(out/'cpu_contract_checks.json',cpu);write(out/'all8_human_slots.json',retained);write(out/'inputs_manifest.json',inputs);write(out/'jobs.json',jobs);write(out/'checkpoint_contracts.json',checkpoints);write(out/'training_support_contexts.json',training);write(out/'base_model_files.json',base)
 protocol={'schema':'visa_fixed_h_existing_r_oracle_v1','registered_utc':now(),'run':str(out),'purpose':'Exploratory conditional R capability on five human-location-accepted fixed real-GT/normal pairs; supplied H, not autonomous H evaluation','data_scope':'Training-support source anomalies and official train normals only; heldout/test image+mask files never read by this runner','all_human_pair_slots':8,'selected_pair_slots':5,'excluded_slots':['C1','P1','P2'],'retained_all8_slots':str(out/'all8_human_slots.json'),'selected_row_ids':list(SELECTED),'inference_seeds':list(SEEDS),'planned_single_candidate_inferences':10,'generator_training_runs':0,'detector_runs':0,'no_candidate_replacement':True,'no_mask_translation_rotation_scale_or_growth':True,'precision':{'vae':'float16 original weights','unet':'float16','text_encoder':'float16','adapter_conditioner_master':'float32'},'model_root':str(args.model_root),'ddim_steps':50,'eta':0.0,'guidance':'original one-conditional predictor; no CFG branch','prompt':'a photo of a normal <category>; requested type selects original128 embedding + Rank8 route','inference_context':'Posterior-mode encoded target NORMAL512; H adaptive-max-pool512->64; hole latent Gaussian seed+10000000; initial noise uses inference seed. Source anomaly RGB is not an inference conditioning input.','training_context':'Same real anomaly posterior-mode latent, own GT hole replaced by Gaussian(config.context_fill_seed+ordered train index); all registered training support hashes and combined scope retained.','exact_renderer_api':'cadiff.v06_c.visa_exact_mask_r.load_visa_renderer -> original CanExactMaskRenderer.__call__','original_finite_guards':['encode_mode latent finite','denoise_masked_infill per-step model prediction finite and final latent finite','decode_latent floating output finite before clamp/quantization and postclamp finite'],'composite':'compose_causal(normal,raw,fixedH), exact H-exterior RGB preserved; output H equals supplied H, not generated annotation','budget':'Ten50DDIM single candidates; no retries, alternative precision, posthoc selection or refill; no new training','failure_policy':'Record the failed job and all already completed jobs, status=failed, stop immediately; no auto retry or silent resume','assessment':'All ten actual outputs retained; human visible/type quality remains unreviewed. Outside-H integrity is engineering only. No object-detection AP selection or detector evaluation.','claim_limits':['Five human-location-accepted training-support oracle pairs, not fresh heldout confirmation.','No formal H success rate or reachability claim; H is manually supplied from real training GT.','No causal repair comparison; source/target orientation and R capability are not independently intervened factors.','Human location acceptance does not prove image/label semantic correctness, visibility, realism, or downstream utility.','Two inference seeds are not generator-training repetitions.'],'frozen_files':files,'cpu_checks_sha256':sha(out/'cpu_contract_checks.json'),'jobs_sha256':sha(out/'jobs.json'),'inputs_manifest_sha256':sha(out/'inputs_manifest.json'),'checkpoint_contracts_sha256':sha(out/'checkpoint_contracts.json'),'human_review_path':str(args.human_review),'human_review_sha256':sha(args.human_review),'startup':'Prepared only. Root/Astra startup review required before explicit --execute --reviewed-protocol-sha.'}
 write(out/'protocol.json',protocol);write(out/'status.json',{'state':'prepared_cpu_review_pending','prepared_utc':now(),'protocol_sha256':sha(out/'protocol.json'),'cpu_checks':'passed','gpu_inference_executed':0,'base_model_loaded':False})
 print(json.dumps({'state':'prepared_cpu_review_pending','run':str(out),'protocol_sha256':sha(out/'protocol.json'),'runner_sha256':sha(Path(__file__)),'jobs':len(jobs),'cpu_checks':cpu},ensure_ascii=False),flush=True)
def execute(args):
 out=args.output;protocol=load(out/'protocol.json');ph=sha(out/'protocol.json')
 if args.reviewed_protocol_sha!=ph:raise ValueError('explicit reviewed protocol SHA required')
 if load(out/'status.json')['state']!='prepared_cpu_review_pending':raise FileExistsError('run is not untouched prepared state; no automatic resume')
 for p,expected in protocol['frozen_files'].items():
  if sha(p)!=expected:raise ValueError('frozen source/input/model drift: '+p)
 for filename,key in [('jobs.json','jobs_sha256'),('inputs_manifest.json','inputs_manifest_sha256'),('checkpoint_contracts.json','checkpoint_contracts_sha256'),('cpu_contract_checks.json','cpu_checks_sha256')]:
  if sha(out/filename)!=protocol[key]:raise ValueError('execution contract changed: '+filename)
 jobs=load(out/'jobs.json');inputs={x['row_id']:x for x in load(out/'inputs_manifest.json')};cp={x['category']:x for x in load(out/'checkpoint_contracts.json')}
 import torch
 from cadiff.v06_c.visa_exact_mask_r import load_visa_renderer,visa_p2_a1_config
 from cadiff.v06_b.renderer import compose_causal
 records=[];renderer=None;active=None;current=None;started=now()
 write(out/'status.json',{'state':'running','started_utc':started,'planned':10,'completed':0,'protocol_sha256':ph})
 try:
  for job in jobs:
   current=job
   if job['category']!=active:
    del renderer;gc.collect();torch.cuda.empty_cache();item=cp[job['category']];config=visa_p2_a1_config(item['category'],tuple(item['config']['defect_types']))
    renderer=load_visa_renderer(checkpoint=Path(item['checkpoint']),model_root=Path(protocol['model_root']),config=config,device=args.device)
    assert renderer.ddim_steps==50;active=job['category']
   inp=inputs[job['row_id']];normal=np.asarray(Image.open(inp['normal_path']).convert('RGB')).copy();h=np.asarray(Image.open(inp['mask_path']))>0
   if binary_sha(h)!=inp['prepared512_binary_sha256'] or rgb_sha(normal)!=inp['normal_pixels_sha256']:raise ValueError('cached physical input changed')
   raw=renderer(normal,job['requested_type'],h.astype(np.uint8),job['seed']);final=compose_causal(normal,raw,h)
   if not np.array_equal(final[~h],normal[~h]):raise ValueError('exterior RGB integrity failed')
   dest=out/'samples'/job['job_id'];dest.mkdir(parents=True,exist_ok=False);Image.fromarray(raw).save(dest/'raw.png');Image.fromarray(final).save(dest/'final_y.png')
   record={**job,'failed':False,'physical512_area':int(h.sum()),'input_mask_binary_sha256':binary_sha(h),'normal_pixels_sha256':rgb_sha(normal),'exterior_rgb_exact':True,'raw_path':str(dest/'raw.png'),'raw_sha256':sha(dest/'raw.png'),'final_path':str(dest/'final_y.png'),'final_sha256':sha(dest/'final_y.png'),'checkpoint_metadata_sha256':cp[job['category']]['metadata_sha256'],'supplied_mask_path':inp['mask_path'],'visible_type_and_realism_review':'unreviewed','not_formal_H_output':True}
   write(dest/'record.json',record);records.append(record);write(out/'records.json',records);write(out/'status.json',{'state':'running','started_utc':started,'planned':10,'completed':len(records),'last_job':job['job_id'],'protocol_sha256':ph})
  write(out/'summary.json',{'state':'completed','completed_utc':now(),'planned':10,'successful':len(records),'failed':0,'selected_pair_slots':5,'all_human_slots':8,'protocol_sha256':ph,'exterior_exact_all':all(x['exterior_rgb_exact'] for x in records),'visible_type_quality':'unreviewed','scope':protocol['purpose']})
  write(out/'status.json',{'state':'completed','started_utc':started,'completed_utc':now(),'completed':len(records),'protocol_sha256':ph,'summary_sha256':sha(out/'summary.json')})
 except BaseException as error:
  failure={'state':'failed','failed_utc':now(),'job':current,'error_type':type(error).__name__,'error':str(error),'completed_jobs':[x['job_id'] for x in records],'completed_count':len(records),'automatic_retry':False,'protocol_sha256':ph};write(out/'failure.json',failure);write(out/'status.json',failure);raise
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--worktree',type=Path,required=True);ap.add_argument('--audit',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--model-root',type=Path,required=True);ap.add_argument('--human-review',type=Path,required=True);ap.add_argument('--execute',action='store_true');ap.add_argument('--reviewed-protocol-sha');ap.add_argument('--device',default='cuda:0');args=ap.parse_args()
 sys.path.insert(0,str(args.worktree/'src'));os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
 if args.execute:execute(args)
 else:prepare(args)
if __name__=='__main__':main()
