"""CPU-first fixed shared scratch pool; target compatibility requires independent review."""
from __future__ import annotations
from dataclasses import asdict
from datetime import datetime,timezone
import hashlib,json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont,ImageFilter,ImageOps
import torch
from cadiff.data.geometry import resize_pad_image
from cadiff.provenance import sha256_file
from cadiff.v06_r.runner import load_train_manifest,write_new_json
from cadiff.v06_r.scratch_pilot_training import load_pilot_config

WORK=Path(__file__).resolve().parents[3]
ROOT=WORK.parents[1]
PILOT=WORK/'runs/v06_r/visa/20261015-scratch-full-vs-crop32'
RUN=WORK/'runs/v06_r/visa/20261015-scratch-shared-pool'
CATALOG=WORK/'runs/v06_audit/visa/20261015-split-mask-coverage/identity_contract.jsonl'
PRIOR_REVIEW=WORK/'runs/v06_r/visa/20261015-fixed-h-existing-r-oracle/inspection/support_structure_readiness/astra_review.json'
SEEDS=(2026101501,2026101502)

def now(): return datetime.now(timezone.utc).isoformat()
def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))
def new(path,data):
    path.parent.mkdir(parents=True,exist_ok=True);write_new_json(path,data)
def pixels_sha(a): return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()

def prepare():
    if RUN.exists(): raise FileExistsError('shared pool run already exists')
    RUN.mkdir(parents=True,exist_ok=False)
    selection={'schema':'scratch_shared_pool_selection_v1','registered_utc':now(),'scope':'candidate identities only, no target compatibility approval and no GPU','normal_rule':'Per object official normal_train, exclude every pilot normal native-file SHA, then sort(image_sha256,image_relpath) and take first10; no output or visual selection','cashew_H_rule':'fixed079080 real scratch trainingGT, already source-reviewed; original prepared512 coordinates, no affine/rotation/translation/growth','pcb_H_rule':'Check all source085086087088 from original scratch training roles, source native GT and preparedGT numeric validity; reuse prior all9 source-only scientific review, choose first2 numeric-qualified by source_id, never pick by output/target appearance. This is not target-H compatibility.','source_numeric_rule':'original registered single scratch training support, exact source image/maskSHA, nonempty originalGT inside source valid canvas, prepared512GT>=64','target_qualification_rule':'All40 fixed target pairs remain unknown until Root/Astra source/normal/H review. Numeric canvas-validity alone never accepts; retain reject/unknown, no substitute source/normal or refills.','planned_normals_per_object':10,'planned_H_per_object':2,'planned_candidate_pairs':40,'inference_seeds':list(SEEDS),'maximum_single_candidate_generations_two_views':160,'models':'Four frozen pilot final1200 models, fresh original FP32 VAE inference; no R training','downstream':'separately registered18 fixed-final detectors only after eligibility, paired generation and direct engineering/gross-failure review; old gum/multitype baseline not reused'}
    new(RUN/'selection_protocol.json',selection)
    pilot_inputs=read(PILOT/'inputs_manifest.json')
    exclude={r['target_image_sha256'] for r in pilot_inputs}
    catalog=[json.loads(line) for line in CATALOG.read_text().splitlines() if line.strip()]
    selected={}; sources={}; source_checks={}; selected_H={}; files={str(CATALOG):sha256_file(CATALOG),str(PRIOR_REVIEW):sha256_file(PRIOR_REVIEW),str(PILOT/'inputs_manifest.json'):sha256_file(PILOT/'inputs_manifest.json'),str(PILOT/'protocol.json'):sha256_file(PILOT/'protocol.json')}
    for cat in ('cashew','pcb1'):
        normals=[r for r in catalog if r['category']==cat and r['role']=='normal_train' and r['image_sha256'] not in exclude]
        assert len({r['image_sha256'] for r in normals})==len(normals)
        normals=sorted(normals,key=lambda r:(r['image_sha256'],r['image_relpath']))[:10]
        assert len(normals)==10; selected[cat]=normals
        config=load_pilot_config(PILOT/'protocol.json',cat)
        manifest=Path(read(PILOT/'protocol.json')['objects'][cat]['manifest'])
        samples,valid,records=load_train_manifest(manifest,ROOT/'data/VisA',config)
        choices={'cashew':['078','079','080','081','082'],'pcb1':['085','086','087','088']}[cat]
        checks=[]; source_map={}
        for sample in samples:
            sid=Path(sample.image_relpath).stem
            if sid not in choices: continue
            legal=valid[sample.sample_id].bool(); gt=sample.mask.bool();area=int(gt.sum())
            qualified=area>=64 and bool(gt.any()) and not bool(torch.any(gt&~legal))
            checks.append({'source_id':sid,'type':sample.defect_type,'role':sample.role,'image_path':str(sample.image_path),'mask_path':str(sample.mask_path),'image_sha256':sample.image_sha256,'mask_sha256':sample.mask_sha256,'preparedGT_area':area,'numeric_source_qualified':qualified,'source_visual_evidence':str(PRIOR_REVIEW),'source_visual_evidence_sha256':sha256_file(PRIOR_REVIEW),'target_compatibility':'not_assessed'})
            source_map[sid]=(sample,valid[sample.sample_id])
        checks.sort(key=lambda r:r['source_id']);source_checks[cat]=checks;sources[cat]=source_map
        eligible=[r['source_id'] for r in checks if r['numeric_source_qualified']]
        selected_H[cat]=eligible[:2]
        assert len(selected_H[cat])==2
    identity={'schema':'scratch_shared_pool_locked_selection_v1','selection_protocol_sha256':sha256_file(RUN/'selection_protocol.json'),'catalog_sha256':sha256_file(CATALOG),'pilot_excluded_normal_sha256':sorted(exclude),'selected_normals':selected,'all_source_checks':source_checks,'selected_H_ids':selected_H,'generation_outputs_inspected_for_selection':0}
    new(RUN/'selected_identity_contract.json',identity)
    # All identities and the deterministic selection rules are fixed before reading target pixels.
    normal_assets={};mask_assets={}
    for cat,normal_rows in selected.items():
        for row in normal_rows:
            path=Path(row['source_path']);assert sha256_file(path)==row['image_sha256'];files[str(path)]=row['image_sha256']
            with Image.open(path) as image: prepared,valid,geometry=resize_pad_image(image,512)
            folder=RUN/'inputs'/cat/'normals'/row['source_id'];folder.mkdir(parents=True)
            prepared.save(folder/'normal.png');valid.save(folder/'valid.png')
            normal_assets[cat,row['source_id']]={'normal_path':str(folder/'normal.png'),'normal_sha256':sha256_file(folder/'normal.png'),'prepared_pixels_sha256':pixels_sha(np.asarray(prepared)),'valid_path':str(folder/'valid.png'),'valid_sha256':sha256_file(folder/'valid.png'),'geometry':asdict(geometry),'native_path':str(path),'native_sha256':row['image_sha256'],'normal_role':'normal_train'}
            files[str(folder/'normal.png')]=sha256_file(folder/'normal.png');files[str(folder/'valid.png')]=sha256_file(folder/'valid.png')
        for sid,(sample,legal) in sources[cat].items():
            folder=RUN/'inputs'/cat/'sources'/sid;folder.mkdir(parents=True)
            image=((sample.pixels+1)/2).mul(255).round().byte().permute(1,2,0).numpy()
            gt=sample.mask[0].byte().mul(255).numpy()
            Image.fromarray(image).save(folder/'source.png');Image.fromarray(gt).save(folder/'H.png')
            mask_assets[cat,sid]={'source_id':sid,'source_image_path':str(sample.image_path),'source_image_sha256':sample.image_sha256,'source_mask_path':str(sample.mask_path),'source_mask_sha256':sample.mask_sha256,'source_prepared_path':str(folder/'source.png'),'source_prepared_sha256':sha256_file(folder/'source.png'),'H_path':str(folder/'H.png'),'H_sha256':sha256_file(folder/'H.png'),'H_area':int((gt>0).sum()),'H_bbox':bbox(gt>0),'requested_type':sample.defect_type}
            for p in (sample.image_path,sample.mask_path,folder/'source.png',folder/'H.png'): files[str(p)]=sha256_file(p)
    rows=[]
    for cat in ('cashew','pcb1'):
        for sid in selected_H[cat]:
            for normal in selected[cat]:
                nid=normal['source_id'];na=normal_assets[cat,nid]
                sa=mask_assets[cat,sid]
                with Image.open(sa['H_path']) as im: h=np.asarray(im)>0
                with Image.open(na['valid_path']) as im: v=np.asarray(im)>0
                review_id=('C' if cat=='cashew' else 'P')+str(sum(r['category']==cat for r in rows)+1).zfill(2)
                rows.append({'identity':cat+'-N'+nid+'-H'+sid,'review_id':review_id,'category':cat,'normal_id':nid,**na,**sa,'numeric_canvas_valid':bool(not np.any(h&~v)),'qualification_status':'unknown','eligibility_reason':'pending independent HUMAN normal/H/type/location review'})
    new(RUN/'candidate_manifest.json',{'schema':'scratch_shared_pool_candidates_v1','rows':rows,'candidate_pair_count':len(rows),'selected_identity_contract_sha256':sha256_file(RUN/'selected_identity_contract.json')})
    for path in (RUN/'selection_protocol.json',RUN/'selected_identity_contract.json',RUN/'candidate_manifest.json'):files[str(path)]=sha256_file(path)
    checkpoints={}
    for cat in ('cashew','pcb1'):
        checkpoints[cat]={}
        for view in ('full','crop32'):
            ck=PILOT/'training'/cat/view/'checkpoint-step-001200';metadata=read(ck/'metadata.json')
            assert metadata['completed_steps']==1200 and len(metadata['config']['defect_types'])==1
            checkpoints[cat][view]={'path':str(ck),'metadata_sha256':sha256_file(ck/'metadata.json'),'config':metadata['config'],'weights':{name:data for name,data in metadata['files'].items() if name.endswith('.safetensors')}}
            files[str(ck/'metadata.json')]=sha256_file(ck/'metadata.json')
            for name,data in metadata['files'].items():
                if name.endswith('.safetensors'):
                    assert sha256_file(ck/name)==data['sha256'];files[str(ck/name)]=data['sha256']
    new(RUN/'frozen_renderers.json',checkpoints);files[str(RUN/'frozen_renderers.json')]=sha256_file(RUN/'frozen_renderers.json')
    new(RUN/'pre_inspection_freeze.json',{'schema':'scratch_shared_pool_pre_inspection_freeze_v1','created_utc':now(),'files':files,'pair_count':40,'heldout_images_read':0,'outputs_inspected':0,'gpu_calls':0,'cuda_initialized':torch.cuda.is_initialized()})
    panels(rows)
    new(RUN/'status.json',{'status':'candidates_locked_waiting_eligibility','created_utc':now(),'candidate_pairs':40,'numeric_canvas_valid_pairs':sum(r['numeric_canvas_valid'] for r in rows),'semantically_approved_pairs':0,'actual_generation_attempts':0,'gpu_started':False,'no_candidate_refill':True})
    print(json.dumps({'run':str(RUN),'candidate_pairs':40,'selected_H':selected_H,'candidate_manifest_sha256':sha256_file(RUN/'candidate_manifest.json'),'pre_inspection_freeze_sha256':sha256_file(RUN/'pre_inspection_freeze.json'),'cuda_initialized':torch.cuda.is_initialized()}))

def bbox(mask):
    yy,xx=np.where(mask)
    return [int(xx.min()),int(yy.min()),int(xx.max())+1,int(yy.max())+1]

def panels(rows):
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',18)
    size,gap=360,12; width=5*(size+gap)+gap; pages={}
    for cat in ('cashew','pcb1'):
        sources=sorted({r['source_id'] for r in rows if r['category']==cat})
        for sid in sources:
            selected=[r for r in rows if r['category']==cat and r['source_id']==sid]
            canvas=Image.new('RGB',(width,130+len(selected)*(size+58)),(244,246,249));draw=ImageDraw.Draw(canvas)
            draw.text((gap,12),cat+' | source H'+sid+' | NO GENERATION | human review: PASS / REJECT / UNCERTAIN',font=font,fill='black')
            draw.text((gap,40),'Real training GT transfer at original prepared512 coordinates. No output, no alignment transform, no refill.',font=font,fill='black')
            titles=['Training source + GT','Target normal','Target normal + H','Fixed ROI target normal','Fixed ROI target + H']
            for j,title in enumerate(titles):draw.text((gap+j*(size+gap),78),title,font=font,fill='black')
            for i,row in enumerate(selected):
                with Image.open(row['normal_path']) as im:normal=im.convert('RGB')
                with Image.open(row['source_prepared_path']) as im:source=im.convert('RGB')
                with Image.open(row['H_path']) as im:mask=im.convert('L')
                boundary=np.asarray(mask.filter(ImageFilter.FIND_EDGES))>0
                annotated=np.asarray(normal).copy();annotated[boundary]=[255,30,30]
                src=np.asarray(source).copy();src[boundary]=[255,30,30]
                x0,y0,x1,y1=row['H_bbox'];box=(max(0,x0-32),max(0,y0-32),min(512,x1+32),min(512,y1+32))
                images=[Image.fromarray(src),normal,Image.fromarray(annotated),normal.crop(box),Image.fromarray(annotated).crop(box)]
                y=112+i*(size+58)
                draw.text((gap,y),row['review_id']+' '+row['identity']+' | H '+str(row['H_area'])+'px | canvas '+str(row['numeric_canvas_valid'])+' | HUMAN STATUS UNKNOWN',font=font,fill='black')
                for j,image in enumerate(images):
                    canvas.paste(ImageOps.pad(image,(size,size),method=Image.Resampling.NEAREST,color=(230,232,235)),(gap+j*(size+gap),y+30))
            path=RUN/'inspection'/(cat+'_H'+sid+'_all10_candidates.png');path.parent.mkdir(exist_ok=True)
            if path.exists():raise FileExistsError(path)
            canvas.save(path);pages[str(path)]={'sha256':sha256_file(path),'rows':[r['identity'] for r in selected],'display':'nearest display zoom only; red outline is supplied H, not G'}
    new(RUN/'inspection/panels_manifest.json',{'schema':'scratch_shared_pool_panels_v1','candidate_manifest_sha256':sha256_file(RUN/'candidate_manifest.json'),'all40_candidates_displayed':True,'pages':pages})

def main():
    import argparse
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['prepare']);a=p.parse_args();prepare()
if __name__=='__main__':main()
