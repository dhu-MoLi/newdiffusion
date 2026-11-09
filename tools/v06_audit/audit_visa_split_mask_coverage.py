"""Isolated CPU audit. Reads legacy sources; only writes a fresh audit run."""
from __future__ import annotations
import argparse, csv, hashlib, io, json, os, sys, time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from PIL import Image

SCHEMA = 'visa_split_mask_coverage_audit_v1'
CATS = ('candle','capsules','cashew','chewinggum','fryum','macaroni1','macaroni2','pcb1','pcb2','pcb3','pcb4','pipe_fryum')
def now(): return datetime.now(timezone.utc).isoformat()
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()
def jread(p): return json.loads(Path(p).read_text())
def lines(p): return [json.loads(s) for s in Path(p).read_text().splitlines() if s.strip()]
def csvread(p):
    with Path(p).open(newline='') as f: return list(csv.DictReader(f))
def writej(p,v): Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,sort_keys=True)+'\n')
def writecsv(p,rows):
    keys=list(dict.fromkeys(k for row in rows for k in row))
    with Path(p).open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=keys); writer.writeheader()
        for row in rows:
            writer.writerow({k:json.dumps(v,ensure_ascii=False,sort_keys=True) if isinstance(v,(dict,list,tuple)) else v for k,v in row.items()})
def geom(size):
    w,h=size; scale=min(512/w,512/h); rw=max(1,round(w*scale)); rh=max(1,round(h*scale))
    l=(512-rw)//2; t=(512-rh)//2
    return [l,t,l+rw,t+rh]
def binary_info(path, rectangle=None):
    if not path.is_file(): return {'state':'missing'},None
    try:
        data=path.read_bytes(); im=Image.open(io.BytesIO(data)); a=np.asarray(im)
        if a.ndim != 2: return {'state':'invalid_channels','shape':list(a.shape)},None
        unique=np.unique(a).tolist(); b=a>0; area=int(b.sum())
        d={'state':'empty' if area==0 else 'nonempty','shape':list(a.shape),'mode':im.mode,'values':unique,'binary_encoding':set(unique).issubset({0,1,255}), 'area_pixels':area,'file_sha256':hashlib.sha256(data).hexdigest(),'decoded_binary_sha256':hashlib.sha256(str(b.shape).encode()+np.packbits(b).tobytes()).hexdigest()}
        if b.shape==(512,512):
            d['area_physical512']=area; d['below64']=area<64
            if rectangle:
                l,t,r,bt=rectangle; inside=int(b[t:bt,l:r].sum()); d['outside_valid_rectangle_pixels']=area-inside
        else: d['area_physical512']=None
        return d,b
    except Exception as e: return {'state':'read_error','error':str(e)},None

def selfcheck():
    for vmax in (1,255):
        a=np.zeros((512,512),np.uint8); a[10:18,20:28]=vmax
        assert int((a>0).sum())==64
        buf=io.BytesIO(); Image.fromarray(a).save(buf,format='PNG'); buf.seek(0)
        assert int((np.asarray(Image.open(buf))>0).sum())==64
    assert geom((1600,900))==[0,112,512,400]
    assert geom((900,1600))==[112,0,400,512]
    assert int((np.zeros((512,512),np.uint8)>0).sum())==0
    return {'status':'passed','checks':['PNG 0/1 retained','PNG 0/255 retained','physical 64 pixels','landscape rectangle','portrait rectangle','empty distinct from missing']}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--root',type=Path,required=True); ap.add_argument('--worktree',type=Path,required=True); ap.add_argument('--output',type=Path,required=True); ap.add_argument('--prepare',action='store_true'); ap.add_argument('--execute',action='store_true')
    args=ap.parse_args(); r=args.root.resolve(); w=args.worktree.resolve(); out=args.output
    catrel=Path('configs/checkpoints/visa_split_catalog_20260922'); catalog=w/catrel
    magic=csvread(catalog/'visa_magic_single_anomaly_amazon1cls_v1.csv'); seas=csvread(catalog/'visa_seas_type_amazon1cls_v1.csv')
    supported=[x for x in magic if int(x['fewshot_n'])>0]
    selected={c:r/'runs/v06_c/visa'/c/'generate'/('20260922-seas-500' if c in CATS[:4] else '20260927-magic-500') for c in CATS}
    locked=[]
    for p in sorted(catalog.iterdir()):
        if p.is_file(): locked.append({'path':str(p),'sha256':sha(p),'main_sha256':sha(r/catrel/p.name) if (r/catrel/p.name).is_file() else None})
    for c,p in selected.items():
        for name in ('records.jsonl','synthetic_manifest.jsonl','summary.json'):
            locked.append({'path':str(p/name),'sha256':sha(p/name),'category':c})
        locked.append({'path':str(r/'data/VisA'/c/'image_anno.csv'),'sha256':sha(r/'data/VisA'/c/'image_anno.csv')})
    for name in ('src/cadiff/data/geometry.py','src/cadiff/v06_c/visa_pilot.py','src/cadiff/v06_c/visa_place_bulk.py','src/cadiff/v06_atlas_diag/extra_a.py','configs/checkpoints/v06_a_idea_a2_tokencut.yaml','configs/checkpoints/v06_b_tokencut_site_scale_h.yaml'):
        for base in (r,w):
            p=base/name
            if p.is_file(): locked.append({'path':str(p),'sha256':sha(p)})
    if args.prepare:
        out.mkdir(parents=True,exist_ok=False)
        protocol={'schema':SCHEMA,'registered_utc':now(),'cpu_only':True,'root':str(r),'worktree':str(w),'output':str(out),'locked_inputs':locked,'selected_history_runs':{k:str(v) for k,v in selected.items()},'catalog_types':len(magic),'supported_types':len(supported),'test_only_types':len(magic)-len(supported),'expected_attempts':500*len(supported),'attempt_key':['category','run_path','job_id'],'record_selection':'single-label catalog slugs only; one locked history run per category; retain failed records','mask_reader':'nonzero bool, preserve 0/1 PNG; never count missing as empty','geometry':'native source dimensions -> current frozen resize_pad 512 round/center; generated masks must already be512; no resizing to manufacture >=64','numeric_success':'record.failed false AND manifest match AND image/mask/sample files readable AND published mask binary, nonempty, physical512 area>=64, inside valid rectangle, subset stored proposal, area agrees record','semantic_status':'unreviewed; numerical validity cannot establish location/type/visible-defect correctness','budget':{'gpu':0,'attempts':500*len(supported),'source_identity_hashes':'all official normal plus1200 anomalies; SHA256; decoded hash on anomalies; native masks nonzero and prepared512','workers':4,'representatives':'first five attempt indices for every supported type, failed included; no images rendered'},'selfcheck':selfcheck()}
        writej(out/'protocol.json',protocol); writej(out/'status.json',{'state':'prepared','utc':now()}); print(json.dumps({'state':'prepared','output':str(out),'protocol_sha256':sha(out/'protocol.json'),'expected_attempts':protocol['expected_attempts']}),flush=True)
        return
    if not args.execute: raise ValueError('choose prepare or execute')
    protocol=jread(out/'protocol.json')
    if (out/'summary.json').exists() or jread(out/'status.json')['state']!='prepared': raise FileExistsError('execute only a prepared fresh run')
    assert protocol['locked_inputs']==locked, 'locked input drift'
    os.environ['CUDA_VISIBLE_DEVICES']=''; t0=time.monotonic(); writej(out/'status.json',{'state':'running','started_utc':now(),'phase':'identities'})
    identity_rows=[]; anomalies={}; normal_by_id={}; type_rows={}; source_root=r/'data/VisA'
    for c in CATS:
        ann=csvread(source_root/c/'image_anno.csv'); anomalies[c]={Path(x['image']).stem:x for x in ann if x['label']!='normal'}
    official=csvread(catalog/'amazon_1cls.csv')
    for x in official:
        if x['label']=='normal':
            c=x['object']; role='normal_train' if x['split']=='train' else 'normal_test'
            identity_rows.append({'category':c,'source_id':Path(x['image']).stem,'image_relpath':x['image'],'role':role,'type':'good','mask_relpath':None})
    for c in CATS:
        byslug={x['slug']:x for x in magic if x['category']==c}; sea={x['slug']:x for x in seas if x['category']==c}
        for sid,x in sorted(anomalies[c].items()):
            combined=',' in x['label']; slug='combined' if combined else x['label'].replace(' ','_')
            m=byslug.get(slug); s=sea[slug]
            role=('excluded_combined' if combined else ('anomaly_support' if sid in m['fewshot_ids'].split() else 'anomaly_heldout'))
            identity_rows.append({'category':c,'source_id':sid,'image_relpath':x['image'],'role':role,'type':slug,'mask_relpath':x['mask'],'seas_role':'anomaly_support' if sid in s['fewshot_ids'].split() else 'anomaly_heldout','original_label':x['label']})
    def inspect_identity(row):
        p=source_root/row['image_relpath']; d=dict(row); d['source_path']=str(p); d['resolved_path']=str(p.resolve()); d['image_sha256']=sha(p)
        with Image.open(p) as im:
            d['native_width'],d['native_height']=im.size; d['valid_rectangle512']=geom(im.size)
            if d['role'].startswith('anomaly') or d['role']=='excluded_combined':
                a=np.asarray(im.convert('RGB')); d['decoded_rgb_sha256']=hashlib.sha256(str(a.shape).encode()+a.tobytes()).hexdigest()
        if d.get('mask_relpath'):
            mp=source_root/d['mask_relpath']; mi,mb=binary_info(mp); d['native_mask']=mi
            with Image.open(mp) as im:
                # Match load_extra_identity: nonzero ->255 before nearest resizes.
                b=Image.fromarray((np.asarray(im.convert('L'))>0).astype(np.uint8)*255)
            b=b.resize((d['native_width'],d['native_height']),Image.Resampling.NEAREST)
            l,t,rr,bb=d['valid_rectangle512']; b=b.resize((rr-l,bb-t),Image.Resampling.NEAREST)
            d['prepared512_mask_area']=int((np.asarray(b)>0).sum()); d['prepared512_below64']=d['prepared512_mask_area']<64
        return d
    with ThreadPoolExecutor(max_workers=4) as ex: identity_rows=list(ex.map(inspect_identity,identity_rows))
    with (out/'identity_contract.jsonl').open('w') as f:
        for d in identity_rows:
            f.write(json.dumps(d,ensure_ascii=False,sort_keys=True)+'\n')
            if d['role'].startswith('normal'): normal_by_id[(d['category'],d['source_id'])]=d
    # Role crossing checks are scoped within each distinct protocol; combined support is legitimate in SeaS.
    leakage={};
    for mode in ('magic','seas'):
        conflicts=[]
        for field in ('image_relpath','resolved_path','image_sha256','decoded_rgb_sha256'):
            groups=defaultdict(list)
            for d in identity_rows:
                role=d['role'] if mode=='magic' or d['role'].startswith('normal') else d['seas_role']
                if role=='excluded_combined' or not d.get(field): continue
                groups[d[field]].append((role,d['category'],d['source_id']))
            for key,items in groups.items():
                roles={x[0] for x in items}
                if ('normal_train' in roles and 'normal_test' in roles) or ('anomaly_support' in roles and 'anomaly_heldout' in roles) or (any(x.startswith('normal') for x in roles) and any(x.startswith('anomaly') for x in roles)):
                    conflicts.append({'identity_field':field,'key':key,'items':items})
        leakage[mode]={'conflicts':conflicts,'count':len(conflicts),'decoded_rgb_scope':'all anomaly images; normal source identity/file SHA checked; re-encoded normal duplicate detection unverified'}
    writej(out/'identity_leakage.json',leakage)
    print(json.dumps({'phase':'identities_complete','count':len(identity_rows),'leakage_conflicts':{k:v['count'] for k,v in leakage.items()},'seconds':time.monotonic()-t0}),flush=True)
    writej(out/'status.json',{'state':'running','utc':now(),'phase':'attempts','source_identities':len(identity_rows)})
    attempt_count=0; all_counters=Counter(); representatives=[]; run_rows=[]; attempt_digest=hashlib.sha256()
    with (out/'attempt_index.jsonl').open('w') as af:
        for c in CATS:
            run=selected[c]; records=lines(run/'records.jsonl'); manifest=lines(run/'synthetic_manifest.jsonl'); mid={x['job_id']:x for x in manifest}; assert len(mid)==len(manifest)
            catspec=[x for x in magic if x['category']==c]; allowed={x['slug'] for x in catspec}; selected_records=[x for x in records if x['type_slug'] in allowed]
            assert len({x['job_id'] for x in selected_records})==len(selected_records)
            group=defaultdict(list)
            for x in selected_records: group[x['type_slug']].append(x)
            rc=Counter()
            for spec in catspec:
                slug=spec['slug']; rr=sorted(group[slug],key=lambda x:(x['index'],x['job_id'])); supported_here=int(spec['fewshot_n'])>0
                if supported_here: assert len(rr)==500 and {x['index'] for x in rr}==set(range(500)), (c,slug,len(rr))
                else: assert len(rr)==0
                tc=Counter(); reasons=Counter(); areas=[]; source_ids=set()
                for x in rr:
                    job=x['job_id']; sample=run/'samples'/job; target=normal_by_id.get((c,x['target_id'])); rect=target['valid_rectangle512'] if target else None
                    d={'category':c,'type_slug':slug,'run_path':str(run),'job_id':job,'attempt_key':f'{c}|{run.name}|{job}','index':x['index'],'seed':x['seed'],'target_id':x['target_id'],'source_id':x.get('source_id'),'record_failed':x['failed'],'failure_reason':x.get('failure_reason'),'record_placed_pixels':x.get('placed_pixels'),'target_is_official_train':bool(target and target['role']=='normal_train'),'target_valid_rectangle512':rect,'in_success_manifest':job in mid,'manifest_record_exact_match':mid.get(job)==x if job in mid else None,'human_semantic_status':'unreviewed'}
                    sid=x.get('source_id'); source_ids.add(sid)
                    source_match=anomalies[c].get(sid)
                    d['source_type_matches_request']=source_match['label']==spec['type'] if source_match else None
                    d['source_in_magic_support']=sid in spec['fewshot_ids'].split() if sid else None
                    d['published_image_exists']=(run/x['image_relpath']).is_file()
                    d['sample_image_exists']=(sample/'final_y.png').is_file()
                    d['sample_record_exists']=(sample/'record.json').is_file()
                    d['sample_record_exact_match']=jread(sample/'record.json')==x if d['sample_record_exists'] else None
                    d['published_mask'],pub=binary_info(run/x['mask_relpath'],rect)
                    d['sample_mask'],sm=binary_info(sample/'m_final.png',rect)
                    d['proposal'],proposal=binary_info(sample/'proposal.png',rect)
                    d['mask_subset_proposal']=bool(not (pub & ~proposal).any()) if pub is not None and proposal is not None and pub.shape==proposal.shape else None
                    d['published_equals_sample_mask']=bool(np.array_equal(pub,sm)) if pub is not None and sm is not None else None
                    d['placeholder_empty_on_failure']=bool(x['failed'] and d['sample_mask']['state']=='empty')
                    violations=[]
                    if not d['target_is_official_train']: violations.append('target_not_official_train')
                    if sid and not d['source_in_magic_support']: violations.append('source_not_request_type_support')
                    if sid and d['source_type_matches_request'] is False: violations.append('source_type_mismatch')
                    if not x['failed']:
                        if job not in mid or mid.get(job)!=x: violations.append('success_manifest_mismatch')
                        if not d['published_image_exists']: violations.append('success_image_missing')
                        else:
                            try:
                                with Image.open(run/x['image_relpath']) as im:
                                    d['published_image_size']=list(im.size); im.verify()
                                if d['published_image_size']!=[512,512]: violations.append('published_image_geometry')
                            except Exception as error: violations.append('success_image_unreadable'); d['image_error']=str(error)
                        if not d['sample_image_exists'] or not d['sample_record_exact_match']: violations.append('sample_incomplete_or_record_mismatch')
                        mi=d['published_mask']
                        if mi['state']!='nonempty': violations.append('success_mask_'+mi['state'])
                        if mi.get('shape')!=[512,512]: violations.append('mask_not_physical512')
                        if not mi.get('binary_encoding'): violations.append('mask_not_binary')
                        if mi.get('area_physical512',0) is not None and mi.get('area_physical512',0)<64: violations.append('mask_below64')
                        if mi.get('outside_valid_rectangle_pixels',0): violations.append('mask_outside_valid_rectangle')
                        if mi.get('area_physical512')!=x.get('placed_pixels'): violations.append('mask_area_record_mismatch')
                        if d['mask_subset_proposal'] is not True: violations.append('mask_not_subset_stored_proposal')
                        if d['published_equals_sample_mask'] is not True: violations.append('published_sample_mask_mismatch')
                    elif job in mid: violations.append('failed_in_success_manifest')
                    d['violations']=violations; d['numeric_success']=bool(not x['failed'] and not violations)
                    state='failed_record' if x['failed'] else 'success_record'
                    tc[state]+=1; tc['numeric_success']+=d['numeric_success']; tc['violating_attempts']+=bool(violations); tc['failure_placeholder_empty']+=d['placeholder_empty_on_failure']
                    tc['failed_published_mask_missing']+=bool(x['failed'] and d['published_mask']['state']=='missing')
                    tc['success_published_empty']+=bool(not x['failed'] and d['published_mask']['state']=='empty')
                    tc['published_png_01_nonempty']+=bool(d['published_mask'].get('values')==[0,1] and d['published_mask']['state']=='nonempty')
                    if x['failed']: reasons[str(x.get('failure_reason'))]+=1
                    if not x['failed'] and d['published_mask'].get('area_physical512') is not None: areas.append(d['published_mask']['area_physical512'])
                    if x['index']<5: representatives.append({'category':c,'type_slug':slug,'job_id':job,'index':x['index'],'record_failed':x['failed'],'failure_reason':x.get('failure_reason'),'sample_root':str(sample),'human_semantic_status':'unreviewed','selection_rule':'first5 indices without replacement; includes failure'})
                    s=json.dumps(d,ensure_ascii=False,sort_keys=True)+'\n'; af.write(s); attempt_digest.update(s.encode()); attempt_count+=1
                tr={'category':c,'type_slug':slug,'type_name':spec['type'],'support_n':int(spec['fewshot_n']),'heldout_n':int(spec['heldout_n']),'capability':'supported' if supported_here else 'test_only_no_support','attempts':len(rr),'success_records':tc['success_record'],'failed_records':tc['failed_record'],'numeric_success':tc['numeric_success'],'all_attempt_numeric_yield':tc['numeric_success']/len(rr) if rr else None,'numeric_given_record_success':tc['numeric_success']/tc['success_record'] if tc['success_record'] else None,'failure_reasons':dict(reasons),'counters':dict(tc),'physical512_success_area_min':min(areas) if areas else None,'physical512_success_area_median':float(np.median(areas)) if areas else None,'physical512_success_area_max':max(areas) if areas else None,'human_semantic_status':'unreviewed'}
                type_rows[(c,slug)]=tr; rc.update(tc)
            run_rows.append({'category':c,'run_path':str(run),'historical_split_id':jread(run/'summary.json').get('split_id'),'total_history_records':len(records),'selected_single_attempts':len(selected_records),'excluded_history_records':len(records)-len(selected_records),'counters':dict(rc)})
            all_counters.update(rc); print(json.dumps({'category_complete':c,'attempts':len(selected_records),'numeric_success':rc['numeric_success'],'violating_attempts':rc['violating_attempts'],'elapsed_seconds':time.monotonic()-t0}),flush=True)
    assert attempt_count==protocol['expected_attempts']==29000
    writecsv(out/'type_capability.csv',list(type_rows.values())); writej(out/'type_capability.json',list(type_rows.values())); writej(out/'history_run_contracts.json',run_rows); writej(out/'representatives_prelocked.json',representatives)
    all_role_counts=Counter(x['role'] for x in identity_rows)
    summary={'schema':SCHEMA,'completed_utc':now(),'elapsed_seconds':time.monotonic()-t0,'protocol_sha256':sha(out/'protocol.json'),'attempt_index_sha256':attempt_digest.hexdigest(),'attempts':attempt_count,'counters':dict(all_counters),'types_total':len(magic),'supported_types':len(supported),'test_only_types':len(magic)-len(supported),'types_with_numeric_success':sum(t['numeric_success']>0 for t in type_rows.values()),'all_attempt_numeric_yield':all_counters['numeric_success']/attempt_count,'identity_role_counts':dict(all_role_counts),'catalog_main_worktree_equal':all(x['sha256']==x.get('main_sha256') for x in locked if 'main_sha256' in x),'identity_conflicts':{k:v['count'] for k,v in leakage.items()},'limitations':['History combines four SeaS source runs filtered to singles and eight MAGIC source runs; not a unified new experiment.','Native GT masks below64 remain legitimate real annotations; 64 is current generated-H gate at prepared512.','Numeric success does not assess location/type semantic compatibility or visible G/H correspondence; human unreviewed.','Normal byte identities and file SHA checked; distinct-encoding identical normal pixels remain unverified.','No GPU, rerender, threshold change, training, detector evaluation or AP selection.'],'conclusion':'Historical numeric support/coverage audit only; preserve all failures and zero-success types.'}
    writej(out/'summary.json',summary)
    post=[{'path':x['path'],'before':x['sha256'],'after':sha(x['path'])} for x in locked]; writej(out/'input_integrity_after.json',{'all_unchanged':all(x['before']==x['after'] for x in post),'files':post})
    writej(out/'status.json',{'state':'completed','completed_utc':now(),'summary_sha256':sha(out/'summary.json'),'input_unchanged':all(x['before']==x['after'] for x in post)})
    print(json.dumps(summary,ensure_ascii=False),flush=True)
if __name__=='__main__': main()
