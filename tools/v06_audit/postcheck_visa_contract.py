"""Independent read-only addendum to completed stage1 audit; no GPU imports."""
import csv, hashlib, json, sys, time
from collections import Counter
from pathlib import Path
from PIL import Image
import numpy as np

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def loadcsv(p):
    with Path(p).open(newline='') as f:return list(csv.DictReader(f))
def write(p,x): Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,sort_keys=True)+'\n')
def main():
    root=Path(sys.argv[1]); w=Path(sys.argv[2]); out=Path(sys.argv[3]); add=out/'postcheck_implementation'; add.mkdir(exist_ok=False); dest=add/'contract_postcheck.json'
    if dest.exists():raise FileExistsError(dest)
    start=time.monotonic(); catalog=w/'configs/checkpoints/visa_split_catalog_20260922'
    specs=loadcsv(catalog/'visa_magic_single_anomaly_amazon1cls_v1.csv')
    ids=[json.loads(s) for s in (out/'identity_contract.jsonl').read_text().splitlines()]
    seas=loadcsv(catalog/'visa_seas_type_amazon1cls_v1.csv')
    capabilities=json.loads((out/'type_capability.json').read_text())
    capability_lookup={(x['category'],x['type_slug']):x for x in capabilities}
    contracts=[]
    for spec in seas:
        key=(spec['category'],spec['slug']); single=capability_lookup.get(key)
        contracts.append({'category':key[0],'type_slug':key[1],'seas_n':int(spec['n']),'seas_support_n':int(spec['fewshot_n']),'seas_heldout_n':int(spec['heldout_n']),'seas_support_ids':spec['fewshot_ids'].split(),'seas_heldout_ids':spec['heldout_ids'].split(),'single_audit_status':single['capability'] if single else 'combined_excluded_from_locked_single_attempt_scope','single_attempts':single['attempts'] if single else None,'single_current_gate_eligible':single['numeric_success'] if single else None,'single_yield':single['all_attempt_numeric_yield'] if single else None,'not_attempted_is_not_failure':True})
    write(add/'all74_type_protocol_contracts.json',contracts)
    fields=list(contracts[0])
    with (add/'all74_type_protocol_contracts.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for x in contracts:writer.writerow({k:json.dumps(v) if isinstance(v,list) else v for k,v in x.items()})
    lookup={(x['category'],x['source_id'],x['role'].startswith('normal')):x for x in ids}
    code=root/'eval_kit/evaluation/datasets.py'; evalcode=root/'eval_kit/evaluate.py'
    registration={'registered_utc':__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat(),'purpose':'Independent native-layout identity reconstruction plus sample image readability; no rendering/reselection','source_files':{str(p):sha(p) for p in [Path(__file__),code,evalcode,out/'protocol.json']},'actual_historical_forward_ids':'unverified; result metadata does not persist exact eval_anomaly_names; reconstruct all catalog types using current loader skip floor(n/3), not proof of runtime forward list'}
    write(add/'contract_postcheck_registration.json',registration)
    layout_rows=[]; problems=[]
    for c in sorted({x['category'] for x in specs}):
        croot=root/'runs/v06_c/visa'/c
        er=croot/'eval'/('20260926-magic-single' if c in ('candle','capsules','cashew','chewinggum') else '20260927-magic-500')/c
        results=sorted(er.glob('results*.json')); assert len(results)==1,(c,results)
        meta=json.loads(results[0].read_text()); layout=Path(meta['mvtec_path'])/c
        d={'category':c,'result_metadata':str(results[0]),'result_sha256':sha(results[0]),'mvtec_layout':str(layout),'reconstruction_scope':'catalog single slugs plus good; floor(n/3) skip; actual historical args unverified','types':[],'issues':[]}
        cs=[x for x in specs if x['category']==c]
        for spec in cs:
            slug=spec['slug']; files=sorted((layout/'test'/slug).glob('*'),key=lambda p:int(p.stem) if p.stem.isdigit() else p.stem)
            expected=spec['fewshot_ids'].split()+spec['heldout_ids'].split()
            if [p.stem for p in files]!=sorted(expected): d['issues'].append({'type':slug,'issue':'native_layout_identity_mismatch'})
            held=[p.stem for p in files[len(files)//3:]]
            if held!=spec['heldout_ids'].split():d['issues'].append({'type':slug,'issue':'reconstructed_heldout_mismatch'})
            source_path_errors=[]; masks=[]
            for p in files:
                identity=lookup[(c,p.stem,False)]; native=Path(identity['source_path'])
                if p.resolve()!=native.resolve() and sha(p)!=identity['image_sha256']:source_path_errors.append(p.stem)
                mask=layout/'ground_truth'/slug/(p.stem+'_mask.png')
                if not mask.exists():mask=layout/'ground_truth'/slug/(p.stem+'.png')
                try:
                    a=np.asarray(Image.open(mask).convert('L')); b=np.asarray(Image.open(root/'data/VisA'/identity['mask_relpath']).convert('L'))
                    eq=a.shape==b.shape and bool(np.array_equal(a>0,b>0))
                    if not eq:masks.append({'source_id':p.stem,'issue':'layout_GT_nonzero_mismatch','layout_shape':list(a.shape),'native_shape':list(b.shape)})
                    if set(np.unique(a).tolist())-set([0,255]):masks.append({'source_id':p.stem,'issue':'layout_GT_not_0255'})
                except Exception as e:masks.append({'source_id':p.stem,'issue':'layout_GT_missing_or_unreadable','error':str(e)})
            d['types'].append({'slug':slug,'full_native_ids':[p.stem for p in files],'reconstructed_support_ids':[p.stem for p in files[:len(files)//3]],'reconstructed_heldout_ids':held,'image_identity_errors':source_path_errors,'GT_issues':masks})
        for split,role in [('train','normal_train'),('test','normal_test')]:
            files=list((layout/split/'good').glob('*')); wanted={x['source_id'] for x in ids if x['category']==c and x['role']==role}; got={p.stem for p in files}
            d[role+'_count']=len(files)
            if got!=wanted:d['issues'].append({'role':role,'issue':'normal_layout_identity_mismatch','missing':sorted(wanted-got),'unexpected':sorted(got-wanted)})
            bad=[p.stem for p in files if p.resolve()!=Path(lookup[(c,p.stem,True)]['source_path']).resolve() and sha(p)!=lookup[(c,p.stem,True)]['image_sha256']]
            if bad:d['issues'].append({'role':role,'issue':'normal_layout_source_bytes_mismatch','ids':bad})
        layout_rows.append(d)
    counters=Counter(); read_errors=[]; violation_counts=Counter(); failure_reasons=Counter(); below64=[]
    for s in (out/'attempt_index.jsonl').open():
        x=json.loads(s); counters['attempts']+=1
        violation_counts.update(x['violations'])
        if x['record_failed']:
            failure_reasons[str(x['failure_reason'])]+=1
            counters['failed_missing_source']+=x['source_id'] is None
        else:
            counters['success_records']+=1; counters['success_missing_source']+=x['source_id'] is None
            p=Path(x['run_path'])/'samples'/x['job_id']/'final_y.png'
            try:
                with Image.open(p) as im:
                    size=im.size; im.verify()
                if size!=(512,512):raise ValueError('not512')
                counters['success_sample_images_readable512']+=1
            except Exception as e:read_errors.append({'job_id':x['job_id'],'error':str(e)})
            if x['published_mask'].get('area_physical512',999999)<64:below64.append({'category':x['category'],'type':x['type_slug'],'job_id':x['job_id'],'area':x['published_mask']['area_physical512']})
    result={'completed_utc':__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat(),'elapsed_seconds':time.monotonic()-start,'registration_sha256':sha(add/'contract_postcheck_registration.json'),'layout_contracts':layout_rows,'layout_top_level_issues':sum(len(x['issues']) for x in layout_rows),'layout_type_image_identity_errors':sum(len(t['image_identity_errors']) for d in layout_rows for t in d['types']),'layout_GT_issues':sum(len(t['GT_issues']) for d in layout_rows for t in d['types']),'counters':dict(counters),'success_sample_read_errors':read_errors,'all_violation_counts':dict(violation_counts),'record_failure_reasons':dict(failure_reasons),'historical_published_below_current64_count':len(below64),'historical_published_below_current64_by_type':dict(Counter(x['category']+'/'+x['type'] for x in below64)),'scientific_scope':'The main numeric_success field is historical pool eligibility when screened by current64; not a fresh execution, not a measured success rate of current A/H. Historical record success and current-gate eligibility must be reported separately.','root_supplement':'Independent root normal decoded RGB and historical R training support/combined checks required; see root supplement when available.','limits':['Current stored source/layout contracts reconstructed; historical exact test forward ID list remains unverified absent persisted eval arguments.','Stored proposal inclusion is a necessary domain check; site geometry not persisted and not independently recomputed.','Sample final_y PNG integrity verified, not decoded semantics; scientific G/H correspondence remains unreviewed.']}
    write(dest,result); print(json.dumps({k:v for k,v in result.items() if k!='layout_contracts'},ensure_ascii=False),flush=True)
    supplemental=out/'supplemental_identity_review'
    status=json.loads((supplemental/'status.json').read_text()) if (supplemental/'status.json').exists() else None
    caveats={'utc':__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat(),'main_summary':str(out/'summary.json'),'main_summary_sha256':sha(out/'summary.json'),'implementation_postcheck':str(dest),'implementation_postcheck_sha256':sha(dest),'root_supplement':str(supplemental),'root_supplement_status':status,'root_supplement_status_sha256':sha(supplemental/'status.json') if status else None,'supersedes_main_summary_normal_decoded_unverified_limitation':bool(status and status.get('state')=='complete' and status.get('decoded_normal_count')==9621),'interpretation':['Old-pool record success and current64 eligibility are distinct; neither is a fresh measured execution of current H.','The main summary normal decoded duplicate limitation is resolved by root supplemental review when marked complete; retain original run history.','Four pilot renderer training scopes legitimately include SeaS combined support; this is a protocol distinction, not leakage.','Records do not persist checkpoint hash: renderer registration scope is checked, exact per-attempt renderer lineage remains unverified.','Mask numerical/proposal inclusion checks do not prove site legality, semantic type/location, or visible G/H correspondence.','Official full source roles and all74 type contracts are present;29000attempt yield only covers the locked62-single protocol with58attempted types.','No exact historical detector forward identity list is established solely from current layout reconstruction.']}
    write(add/'final_caveats.json',caveats)
if __name__=='__main__':main()
