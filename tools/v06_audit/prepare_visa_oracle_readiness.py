"""CPU-only identity/weight readiness; no model imports, image generation or selection by AP."""
import csv, hashlib, json, sys, time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from PIL import Image
def now():return datetime.now(timezone.utc).isoformat()
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def write(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,sort_keys=True)+'\n')
def csvwrite(p,rows):
 with Path(p).open('w',newline='') as f:
  writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader()
  for x in rows:writer.writerow({k:json.dumps(v) if isinstance(v,(list,dict)) else v for k,v in x.items()})
def prepared_mask(identity,source_root):
 mp=source_root/identity['mask_relpath']
 assert sha(mp)==identity['mask_sha256'] if 'mask_sha256' in identity else sha(mp)==identity['native_mask']['file_sha256']
 with Image.open(mp) as im:binary=Image.fromarray((np.asarray(im.convert('L'))>0).astype(np.uint8)*255)
 # Exactly the frozen load_extra_identity adapter; no additional crop, scaling or placement.
 binary=binary.resize((identity['native_width'],identity['native_height']),Image.Resampling.NEAREST)
 l,t,r,b=identity['valid_rectangle512'];binary=binary.resize((r-l,b-t),Image.Resampling.NEAREST)
 canvas=np.zeros((512,512),dtype=bool);canvas[t:b,l:r]=np.asarray(binary)>0
 return canvas
def main():
 root=Path(sys.argv[1]);w=Path(sys.argv[2]);audit=Path(sys.argv[3]);out=audit/'oracle_readiness'
 out.mkdir(exist_ok=False);start=time.monotonic()
 source_root=root/'data/VisA';identity_path=audit/'identity_contract.jsonl';scope_path=audit/'supplemental_identity_review/historical_renderer_scope.json'
 specs=list(csv.DictReader((w/'configs/checkpoints/visa_split_catalog_20260922/visa_magic_single_anomaly_amazon1cls_v1.csv').open()))
 specs=[x for x in specs if int(x['fewshot_n'])>0];assert len(specs)==58
 ids=[json.loads(s) for s in identity_path.read_text().splitlines()];scope=json.loads(scope_path.read_text())
 metadata_paths=[root/'runs/v06_b/visa'/c/'exact-mask-r'/('seed-'+('20260922-seas' if c in ['candle','capsules','cashew','chewinggum'] else '20260927-magic'))/'checkpoint/metadata.json' for c in sorted({s['category'] for s in specs})]
 locks={str(p):sha(p) for p in [identity_path,scope_path,Path(__file__)]+metadata_paths}
 protocol={'schema':'visa_existing_r_oracle_readiness_v1','registered_utc':now(),'scope':'CPU candidate identity and weight readiness only; not a registered inference experiment','gpu_budget':0,'inference_executed':0,'all_supported_types':58,'nominal_full_budget_if_every_type_has_two_shapes':464,'candidate_rule':'Within each single supported type, source training IDs sorted lexicographically; first2 real GT masks with frozen prepared512 area>=64 and no pixels outside source valid rectangle. Retain all rejection and shortfall records.','target_rule':'For each of all12 categories, first2 official normal_train source IDs lexicographic; never swap by mask overlap, appearance or AP.','spatial_transfer':'Keep original prepared512 coordinates; no new translation, scale, crop or target fitting. Record outside target normal valid rectangle.','seed_rule':'Two symbolic slots per mask/normal pair, actual numeric seeds unregistered/null; no inference authorized by readiness.','semantic_rule':'Requested-label agreement from source metadata is separate from visual semantics; source/target component and location/type compatibility remain unknown.','weights_rule':'Read metadata then SHA256/size of existing adapter and conditioner only; never load model; first4 SeaS and next8 MAGIC training scopes retained.','locked_inputs':locks,'limitations':['Real-GT oracle candidates cannot establish formal H reachability.','Matching coordinates and valid rectangles cannot establish object/component/type compatibility.','No generated image, returned label, visibility or quality evidence is produced.','Specific limited inference budget, seeds and subset remain to be reviewed and separately registered.']}
 write(out/'protocol.json',protocol);write(out/'status.json',{'state':'running','started_utc':now()})
 shape_rows=[];type_rows=[];pairs=[];normal_rows=[];mask_arrays={};selected_by_type={}
 for c in sorted({s['category'] for s in specs}):
  normal=sorted([i for i in ids if i['category']==c and i['role']=='normal_train'],key=lambda i:i['source_id'])[:2]
  assert len(normal)==2
  for i in normal:
   assert sha(i['source_path'])==i['image_sha256']
   normal_rows.append({'category':c,'normal_id':i['source_id'],'image_path':i['source_path'],'image_sha256':i['image_sha256'],'valid_rectangle512':i['valid_rectangle512'],'native_size':[i['native_width'],i['native_height']],'selection_rule':'first2 official train normal IDs without replacement','component_compatibility':'unknown'})
 for spec in specs:
  c=spec['category'];slug=spec['slug'];support=sorted([i for i in ids if i['category']==c and i['type']==slug and i['role']=='anomaly_support'],key=lambda i:i['source_id'])
  assert len(support)==int(spec['fewshot_n']);chosen=[];qualified=0
  for identity in support:
   assert sha(identity['source_path'])==identity['image_sha256']
   b=prepared_mask(identity,source_root);area=int(b.sum());l,t,r,bt=identity['valid_rectangle512'];outside=area-int(b[t:bt,l:r].sum());eligible=area>=64 and outside==0
   if eligible:qualified+=1
   selected=eligible and len(chosen)<2
   if selected:chosen.append(identity['source_id']);mask_arrays[(c,slug,identity['source_id'])]=b
   shape_rows.append({'category':c,'type_slug':slug,'type_name':spec['type'],'source_id':identity['source_id'],'image_path':identity['source_path'],'image_sha256':identity['image_sha256'],'mask_path':str(source_root/identity['mask_relpath']),'native_mask_file_sha256':identity['native_mask']['file_sha256'],'prepared512_area':area,'source_valid_rectangle512':identity['valid_rectangle512'],'source_outside_valid_rectangle_pixels':outside,'source_numeric_eligible':eligible,'selected_first2':selected,'selection_reason':'selected_first2_numerically_eligible' if selected else ('eligible_after_first2_not_selected' if eligible else 'below64' if area<64 else 'outside_source_valid_rectangle'),'prepared_binary_sha256':hashlib.sha256(np.packbits(b).tobytes()).hexdigest(),'visual_source_type_compatibility':'unknown','source_target_component_compatibility':'unknown'})
  selected_by_type[(c,slug)]=chosen
  type_rows.append({'category':c,'type_slug':slug,'support_count':len(support),'numerically_eligible_support_count':qualified,'selected_shape_ids':chosen,'selected_shape_count':len(chosen),'missing_shapes_to_two':2-len(chosen),'normal_ids':[n['normal_id'] for n in normal_rows if n['category']==c],'symbolic_seed_slots':2,'nominal_budget':8,'candidate_slot_budget_without_replacement':len(chosen)*2*2,'inference_executed':0,'semantic_readiness':'unknown'})
  for sid in chosen:
   b=mask_arrays[(c,slug,sid)];area=int(b.sum())
   for n in [n for n in normal_rows if n['category']==c]:
    l,t,r,bt=n['valid_rectangle512'];outside=area-int(b[t:bt,l:r].sum())
    pairs.append({'category':c,'type_slug':slug,'source_mask_id':sid,'target_normal_id':n['normal_id'],'transfer':'unchanged_prepared512_coordinates','target_valid_rectangle512':n['valid_rectangle512'],'mask_area_physical512':area,'outside_target_valid_rectangle_pixels':outside,'inside_target_valid_rectangle':outside==0,'seed_slots':['seed_slot0','seed_slot1'],'actual_inference_seeds':None,'potential_inference_count':2,'semantic_type_compatibility':'unknown','source_target_component_compatibility':'unknown','formal_H_reachability':'unknown','oracle_legality':'not_established','inference_executed':0})
 weight_rows=[]
 for mp in metadata_paths:
  metadata=json.loads(mp.read_text());c=metadata['config']['category'];scope_item=next(x for x in scope['categories'] if x['category']==c);checks=[]
  for name in ['adapter.safetensors','conditioner.safetensors']:
   p=mp.parent/name;registered=metadata['files'][name];actual=sha(p) if p.is_file() else None;size=p.stat().st_size if p.is_file() else None
   checks.append({'name':name,'path':str(p),'exists':p.is_file(),'actual_sha256':actual,'registered_sha256':registered['sha256'],'sha256_matches':actual==registered['sha256'],'actual_bytes':size,'registered_bytes':registered['size'],'size_matches':size==registered['size']})
  requested=[s['type'] for s in specs if s['category']==c];missing=sorted(set(requested)-set(metadata['config']['defect_types']))
  weight_rows.append({'category':c,'checkpoint_path':str(mp.parent),'metadata_path':str(mp),'metadata_sha256':sha(mp),'split_id':metadata['split_id'],'steps':metadata['config']['steps'],'type_rank':metadata['config']['type_rank'],'image_size':metadata['config']['image_size'],'dtype':metadata['config']['dtype'],'registered_defect_types':metadata['config']['defect_types'],'supported_single_requested_types':requested,'missing_requested_types':missing,'weights':checks,'weights_match_metadata':all(x['sha256_matches'] and x['size_matches'] for x in checks),'training_manifest_path':next(x['path'] for x in scope_item['evidence'] if x['path'].endswith('manifest.jsonl')),'extra_combined_training_count':scope_item['combined_training_count'],'extra_combined_training_ids':scope_item['combined_training_ids'],'source_scope':'historical registered R; exact prior generation checkpoint lineage unverified; candidates reference this exact checkpoint for future protocol','model_loaded':False})
 for name,rows in [('support_shape_inventory',shape_rows),('type_readiness',type_rows),('fixed_normal_identities',normal_rows),('same_coordinate_candidate_pairs',pairs),('checkpoint_readiness',weight_rows)]:
  write(out/(name+'.json'),rows)
  if rows:csvwrite(out/(name+'.csv'),rows)
 summary={'schema':protocol['schema'],'completed_utc':now(),'elapsed_seconds':time.monotonic()-start,'protocol_sha256':sha(out/'protocol.json'),'categories':12,'supported_types':58,'all_training_single_support_shapes':len(shape_rows),'eligible_support_shapes':sum(x['source_numeric_eligible'] for x in shape_rows),'selected_shapes':sum(x['selected_first2'] for x in shape_rows),'types_with_two_selected_shapes':sum(x['selected_shape_count']==2 for x in type_rows),'shortfall_types':[x for x in type_rows if x['missing_shapes_to_two']],'fixed_normal_images':len(normal_rows),'mask_normal_pairs':len(pairs),'pairs_inside_target_valid_rectangle':sum(x['inside_target_valid_rectangle'] for x in pairs),'pairs_outside_target_valid_rectangle':sum(not x['inside_target_valid_rectangle'] for x in pairs),'nominal_all58_full_budget':464,'candidate_slot_budget_without_replacement':sum(x['potential_inference_count'] for x in pairs),'target_rectangle_inside_slot_budget':sum(x['potential_inference_count'] for x in pairs if x['inside_target_valid_rectangle']),'checkpoints':len(weight_rows),'weight_files':sum(len(x['weights']) for x in weight_rows),'all_weight_hash_size_matches':all(x['weights_match_metadata'] for x in weight_rows),'all_supported_types_have_registered_route':all(not x['missing_requested_types'] for x in weight_rows),'inference_executed':0,'gpu_budget':0,'semantic_type_component_compatibility':'unknown for every candidate','oracle_legality':'not_established','conclusion':'CPU identity and weight preparation completed; numerical candidate checks do not authorize or establish a legal oracle inference experiment.'}
 write(out/'summary.json',summary)
 postlocks={p:sha(p) for p in locks};assert postlocks==locks,'input drift'
 write(out/'input_integrity_after.json',{'all_locked_inputs_unchanged':True,'locked_inputs':postlocks})
 write(out/'status.json',{'state':'completed','completed_utc':summary['completed_utc'],'summary_sha256':sha(out/'summary.json'),'input_unchanged':True,'inference_executed':0})
 print(json.dumps(summary,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
