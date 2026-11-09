"""Read-only 6-run aggregate; immutable output and registered exploratory screen."""
import argparse,csv,json
from pathlib import Path
from statistics import mean,stdev,median
from cadiff.v06_r.downstream import read_json,digest,new_json,SEEDS
from cadiff.v06_r.full_crop_rep2_downstream import CONDITIONS
from cadiff.v06_r.full_crop_rep2_training import validate_protocol

def stats(values):
    return {'per_seed':{str(s):float(v) for s,v in zip(SEEDS,values)},'mean':mean(values),'sample_std':stdev(values)}

def build_payload(evaluation,protocol):
    registered=validate_protocol(protocol); frozen=read_json(evaluation/'frozen.json')
    if digest(protocol)!=frozen['protocol_sha256']: raise ValueError('aggregate protocol differs from prepared freeze')
    references=read_json(evaluation/'real_fs_references.json')
    if references!=frozen['old_real_fs_references']: raise ValueError('Real-FS reference manifest differs')
    sources={str(protocol.resolve()):digest(protocol),str((evaluation/'frozen.json').resolve()):digest(evaluation/'frozen.json'),
             str((evaluation/'real_fs_references.json').resolve()):digest(evaluation/'real_fs_references.json')}
    all_results={}
    for condition in (*CONDITIONS,'real_fs_reference'):
        rows={}
        for seed in SEEDS:
            if condition=='real_fs_reference':
                reference=references[str(seed)]; path=Path(reference['results_path']); checkpoint=Path(reference['checkpoint_path'])
                if digest(path)!=reference['results_sha256'] or digest(checkpoint)!=reference['checkpoint_sha256']: raise ValueError('locked Real-FS changed')
            else:
                folder=evaluation/condition/f'seed-{seed}'; path=folder/'results.json'; checkpoint=folder/'checkpoint-final.pt'
            payload=read_json(path)
            expected='real_fs' if condition=='real_fs_reference' else condition
            if payload.get('condition')!=expected or payload.get('seed')!=seed or payload.get('actual_optimizer_steps')!=6400 or payload.get('test_forward_count')!=89 or payload.get('checkpoint_rule')!='last_epoch_only':
                raise ValueError('aggregate requires all9 fixed final result identities')
            if digest(checkpoint)!=payload['checkpoint_sha256']: raise ValueError('result checkpoint changed')
            sources[str(path.resolve())]=digest(path); sources[str(checkpoint.resolve())]=digest(checkpoint)
            rows[str(seed)]=payload
        all_results[condition]=rows
    groups=all_results['real_fs_reference'][str(SEEDS[0])]['metrics'].keys()
    summaries={}
    for condition,rows in all_results.items():
        summaries[condition]={}
        for group in groups:
            names=[k for k,v in rows[str(SEEDS[0])]['metrics'][group].items() if isinstance(v,(int,float)) and k!='image_count']
            summaries[condition][group]={name:stats([rows[str(seed)]['metrics'][group][name] for seed in SEEDS]) for name in names}
    deltas={}
    for condition in CONDITIONS:
        comparators=['real_fs_reference']
        if condition.endswith('_crop32'): comparators.append(condition.replace('_crop32','_full'))
        deltas[condition]={}
        for comparator in comparators:
            deltas[condition][comparator]={}
            for group in groups:
                values=[all_results[condition][str(seed)]['metrics'][group]['ap_pixel']-all_results[comparator][str(seed)]['metrics'][group]['ap_pixel'] for seed in SEEDS]
                deltas[condition][comparator][group]=stats(values)
    pair=deltas['real_gen_crop32']['real_gen_full']; primary=[pair['primary']['per_seed'][str(seed)] for seed in SEEDS]
    gate=registered['advance_screen']; required=(gate['primary_paired_delta_median_min'],gate['positive_seeds_min'])
    if required!=(.005,2): raise ValueError('advance screen parameters differ')
    primary_types=[t.replace(' ','_') for t in registered['primary_types']]
    checks={'primary_delta_median_min':median(primary)>=required[0],'positive_seed_count_min':sum(v>0 for v in primary)>=required[1],
            'both_primary_mean_delta_positive':all(pair[t]['mean']>0 for t in primary_types),
            'overall_mean_delta_nonnegative':pair['overall']['mean']>=0}
    type_changes={t.replace(' ','_'):{**pair[t.replace(' ','_')],'negative_mean_flag':pair[t.replace(' ','_')]['mean']<0,'negative_seed_flags':{str(seed):pair[t.replace(' ','_')]['per_seed'][str(seed)]<0 for seed in SEEDS}} for t in registered['defect_types']}
    screen={'comparison':'RealGen Crop32 versus matched fresh Full','checks':checks,'primary_delta_median':median(primary),'positive_seed_count':sum(v>0 for v in primary),
            'exploratory_numeric_screen_passed':all(checks.values()),'all_type_deltas_and_regression_flags':type_changes,
            'compatibility_regression_review_needed':any(v['negative_mean_flag'] for v in type_changes.values()),
            'serious_compatibility_regression_not_defined_by_an_unregistered_cutoff':True,
            'not_statistical_significance':True,'not_automatic_promotion_or_training_authorization':True}
    return {'schema_version':'v06_r_full_crop_aggregate_v1','new_detector_run_count':6,'locked_real_fs_reference_count':3,
            'protocol_sha256':digest(protocol),'source_sha256':sources,'complete':True,'all_results':all_results,'summaries':summaries,
            'paired_pixel_ap_deltas':deltas,'advance_screen':screen,'detector_seeds':list(SEEDS),
            'generation_repeat_interpretation':'Exploratory independent R-training replication2 on the same previously observed89-image split; independent R n=2 across studies, detector3 per pool nested, not6 independent R replicates. Full/Crop delta per generator repetition is the replication unit; no automatic rep3.'}

def aggregate(evaluation,protocol,output):
    payload=build_payload(evaluation,protocol)
    json_path=output/'summary.json'; csv_path=output/'summary.csv'
    if output.exists():
        if not json_path.is_file() or read_json(json_path)!=payload or not csv_path.is_file(): raise FileExistsError('existing aggregate differs; refuses overwrite')
        # Verify CSV exact content as well; never silently reuse a partial or edited table.
        if csv_path.read_text()!=csv_text(payload): raise ValueError('existing aggregate CSV differs')
        return {'output':str(output),'reused_equal_payload':True,'screen':payload['advance_screen']}
    output.mkdir(parents=True,exist_ok=False); new_json(json_path,payload)
    with csv_path.open('x',newline='',encoding='utf-8') as f: f.write(csv_text(payload))
    return {'output':str(output),'reused_equal_payload':False,'screen':payload['advance_screen']}

def csv_text(payload):
    import io
    buffer=io.StringIO(newline=''); writer=csv.writer(buffer,lineterminator='\n')
    writer.writerow(['condition','group','metric','seed_2026101301','seed_2026101302','seed_2026101303','mean','sample_std'])
    for condition,groups in payload['summaries'].items():
        for group,metrics in groups.items():
            for metric,value in metrics.items(): writer.writerow([condition,group,metric,*[value['per_seed'][str(seed)] for seed in SEEDS],value['mean'],value['sample_std']])
    return buffer.getvalue()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--evaluation',type=Path,required=True); p.add_argument('--protocol',type=Path,required=True); p.add_argument('--output',type=Path,required=True)
    a=p.parse_args(); print(json.dumps(aggregate(a.evaluation,a.protocol,a.output),ensure_ascii=False),flush=True)
if __name__=='__main__': main()
