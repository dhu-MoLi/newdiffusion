"""Inference-only 512-square geometry bridge; physical GT remains on the source canvas."""
from __future__ import annotations
from dataclasses import dataclass,asdict
import csv
import gc
import json
from pathlib import Path
from statistics import median
import time
import numpy as np
from PIL import Image,ImageDraw,ImageFont,ImageOps
import torch
from cadiff.data.geometry import GeometryMeta,letterbox_geometry,resize_pad_image,resize_pad_mask
from cadiff.provenance import sha256_file
from cadiff.v06_r.runner import write_new_json
from cadiff.v06_r.vae_ceiling import (
    SOURCE_PATHS as P0_SOURCES,_read_json,_samples,_identity,_verify_stage_code,
    _binary,_bbox,fixed_roi,metric_domains,reconstruction_metrics,
    validate_vae,code_manifest_data,verify_code_manifest,_rgb_image,
)

METHODS=('F','Z32','Q32','Z64','Q64')
MARGINS=(32,64)
PARAMETERS={
    'image_size':512,'roi_margins':[32,64],'predefined_next_margin':32,
    'forward_rgb':'original_resize_pad_image_PIL_RGB_uint8_LANCZOS',
    'forward_mask':'original_resize_pad_mask_NEAREST',
    'padding':'centered_black','size_rounding':'python_round',
    'inverse_resize':'PIL_F_LANCZOS_per_channel',
    'decode_clamp_before_inverse':True,'inverse_clamp_after_resize':True,
    'metric_roi_margin':32,'metrics_float_before_png_quantization':True,
    'edge_domain':'complete_3x3_gt','boundary_norm':'linf','boundary_radius':3,
    'vae_dtype':'float32','posterior':'mode','methods':list(METHODS),
    'vae_roundtrips':27,'baseline_repeat_absolute_tolerance':1e-6,
}
SOURCES=(*P0_SOURCES,'src/cadiff/v06_r/square_roi_bridge.py','tools/v06_r/check_square_roi_bridge.py')

@dataclass(frozen=True)
class SquareRoiGeometry:
    box:tuple[int,int,int,int]
    valid_box:tuple[int,int,int,int]
    margin:int
    letterbox:GeometryMeta

def square_roi_geometry(gt,valid,margin):
    if margin not in MARGINS or type(margin)!=int: raise ValueError('only preregistered margins32/64')
    gt,valid=_binary(gt,name='GT'),_binary(valid,name='valid')
    if gt.shape!=(1,1,512,512) or gt.shape!=valid.shape or torch.any(gt&~valid): raise ValueError('GT must be inside valid source512')
    vb=_bbox(valid); rectangle=torch.zeros_like(valid); rectangle[:,:,vb[1]:vb[3],vb[0]:vb[2]]=True
    if not torch.equal(rectangle,valid): raise ValueError('valid domain is not original rectangle')
    gb=_bbox(gt)
    box=(max(vb[0],gb[0]-margin),max(vb[1],gb[1]-margin),min(vb[2],gb[2]+margin),min(vb[3],gb[3]+margin))
    x0,y0,x1,y1=box
    if int(gt[:,:,y0:y1,x0:x1].sum())!=int(gt.sum()): raise ValueError('ROI loses physical GT')
    return SquareRoiGeometry(box,vb,margin,letterbox_geometry(x1-x0,y1-y0,512))

def forward_square(original,gt,geometry):
    """Reusable production geometry: source RGB uint8/GT to original512 square helpers."""
    if original.requires_grad: raise ValueError('PIL geometry is detached preprocessing, not a differentiable training projection')
    if original.dtype!=torch.float32 or original.shape!=(1,3,512,512) or not torch.isfinite(original).all() or torch.any(original<0) or torch.any(original>1):
        raise ValueError('source must be finite FP32 RGB512 in[0,1]')
    gt=_binary(gt,name='GT')
    x0,y0,x1,y1=geometry.box
    source=_rgb_image(original).crop(geometry.box)
    mask=Image.fromarray(gt[0,0].cpu().byte().mul(255).numpy()).crop(geometry.box)
    rgb,valid,meta=resize_pad_image(source,512)
    if meta!=geometry.letterbox: raise ValueError('original geometry helper differs from frozen metadata')
    square_mask=resize_pad_mask(mask,meta)
    pixels=torch.from_numpy(np.array(rgb,copy=True)).permute(2,0,1).unsqueeze(0).float()/255
    mapped=torch.from_numpy(np.array(square_mask,copy=True)).unsqueeze(0).unsqueeze(0).ge(128)
    legal=torch.from_numpy(np.array(valid,copy=True)).unsqueeze(0).unsqueeze(0).ge(128)
    if not mapped.any() or torch.any(mapped&~legal): raise ValueError('forward GT mapped outside squarevalid or vanished')
    return pixels,mapped,legal

def inverse_square(square,geometry):
    """Inference/deployment only: stored unpad, floating PIL LANCZOS, then clamp.

    This detached PIL operation is not a differentiable consistency projection.
    """
    if square.requires_grad: raise ValueError('PIL inverse is inference-only and cannot propagate training gradients')
    if square.shape!=(1,3,512,512) or square.dtype!=torch.float32 or not torch.isfinite(square).all():
        raise ValueError('inverse requires finite FP32 RGB square512')
    meta=geometry.letterbox
    rectangle=(meta.pad_left,meta.pad_top,meta.pad_left+meta.resized_width,meta.pad_top+meta.resized_height)
    x0,y0,x1,y1=geometry.box
    channels=[]
    for channel in square[0].clamp(0,1).detach().cpu().numpy():
        image=Image.fromarray(channel.astype(np.float32)).crop(rectangle)
        restored=image.resize((x1-x0,y1-y0),Image.Resampling.LANCZOS)
        channels.append(np.array(restored,dtype=np.float32,copy=True))
    return torch.from_numpy(np.stack(channels)).unsqueeze(0).clamp(0,1).to(square.device)

def paste_square_roi(original,roi,geometry):
    x0,y0,x1,y1=geometry.box
    if roi.shape!=(1,3,y1-y0,x1-x0): raise ValueError('inverse ROI dimensions differ')
    result=original.clone(); result[:,:,y0:y1,x0:x1]=roi
    outside=torch.ones((512,512),dtype=torch.bool,device=original.device); outside[y0:y1,x0:x1]=False
    if not torch.equal(result[:,:,outside],original[:,:,outside]): raise ValueError('outside ROI must remain exactly originalfloat')
    return result

@torch.inference_mode()
def roundtrip512(vae,rgb):
    validate_vae(vae)
    if rgb.dtype!=torch.float32 or rgb.shape!=(1,3,512,512) or not torch.isfinite(rgb).all() or torch.any(rgb<0) or torch.any(rgb>1):
        raise ValueError('VAE input must be FP32 square512 RGB[0,1]')
    scale=float(vae.config.scaling_factor)
    latent=vae.encode(rgb*2-1).latent_dist.mode()*scale
    if latent.shape!=(1,4,64,64) or latent.dtype!=torch.float32 or not torch.isfinite(latent).all(): raise ValueError('production latent must be finite FP32 4x64x64')
    raw=vae.decode(latent/scale).sample.float()/2+.5
    if raw.shape!=rgb.shape or not torch.isfinite(raw).all(): raise ValueError('invalid VAE output')
    fraction=float(((raw<0)|(raw>1)).float().mean())
    return raw.clamp(0,1),fraction

def protocol_check(path):
    data=_read_json(path)
    if data.get('schema_version')!='v06_r_square_roi_bridge_protocol_v1' or data.get('parameters')!=PARAMETERS:
        raise ValueError('registered bridge parameters differ')
    screen=data.get('numeric_screen',{})
    required={'near_zero_denominator':1e-8,'material_relative_regression_maximum':.10,'material_absolute_regression_maximum':1/255}
    if any(screen.get(k)!=v for k,v in required.items()): raise ValueError('registered screening thresholds differ')
    return data

def source_samples(root,p0):
    old=_read_json(p0/'inputs.json'); _verify_stage_code(p0,old)
    source,samples,valid=_samples(root)
    if [_identity(s,valid[s.sample_id]) for s in samples]!=old['rows']: raise ValueError('frozen nine train identities changed')
    return old,samples,valid

def prepare(*,root,output,protocol):
    protocol_check(protocol); p0=root/'vae_ceiling_p0'
    old,samples,valid=source_samples(root,p0)
    old_records=[json.loads(line) for line in (p0/'private/metrics_by_method.jsonl').read_text().splitlines() if line.strip()]
    baseline={r['identity']:r for r in old_records}
    rows=[]
    for sample in samples:
        identity=_identity(sample,valid[sample.sample_id]); gt=sample.mask[None]; legal=valid[sample.sample_id][None]
        geometries={str(m):json.loads(json.dumps(asdict(square_roi_geometry(gt,legal,m)))) for m in MARGINS}
        old_record=baseline[identity['identity']]
        full=p0/old_record['paths']['X3']['reconstruction']
        if sha256_file(full)!=old_record['paths']['X3']['reconstruction_sha256']: raise ValueError('old fullF sidecar changed')
        rows.append({**identity,'square_geometries':geometries,'old_F_metrics':old_record['metrics']['F'],
                     'old_F_png':str(full),'old_F_png_sha256':sha256_file(full)})
    if output.exists() or output.is_symlink(): raise FileExistsError('refuses to overwrite square bridge stage')
    output.mkdir(parents=True,exist_ok=False)
    with (output/'registered_protocol.json').open('xb') as f: f.write(protocol.read_bytes())
    frozen={'schema_version':'v06_r_square_roi_bridge_inputs_v1','root':str(root.resolve()),'p0':str(p0.resolve()),
            'p0_inputs_sha256':sha256_file(p0/'inputs.json'),'p0_metrics_sha256':sha256_file(p0/'private/metrics_by_method.jsonl'),
            'protocol_path':str(protocol.resolve()),'protocol_sha256':sha256_file(protocol),'sample_count':9,
            'vae_directory':old['vae_directory'],'vae_files_sha256':old['vae_files_sha256'],'rows':rows,
            'heldout_loaded':0,'coordinate_system':'original prepared512 letterbox, not native1342',
            'metric_domains':'same frozen old GTbbox32 for all methods and both margins'}
    write_new_json(output/'inputs.json',frozen)
    return {'stage':'square_bridge_cpu_frozen','samples':9,'roundtrips':27,'output':str(output)}

def freeze_code(output):
    if any((output/name).exists() for name in ('sidecars','domains','metrics.json','summary.json')): raise FileExistsError('cannot freeze after inference')
    frozen=_read_json(output/'inputs.json'); protocol_check(Path(frozen['protocol_path']))
    if sha256_file(Path(frozen['protocol_path']))!=frozen['protocol_sha256'] or sha256_file(output/'registered_protocol.json')!=frozen['protocol_sha256']:
        raise ValueError('bridge protocol changed before source freeze')
    worktree=Path(__file__).resolve().parents[3]
    data=code_manifest_data(worktree,source_paths=SOURCES)
    data['inputs_sha256']=sha256_file(output/'inputs.json'); data['protocol_sha256']=frozen['protocol_sha256']
    write_new_json(output/'code_manifest.json',data)
    return {'stage':'square_bridge_code_frozen','source_count':len(data['files'])}

def screen_records(records):
    result={'per_margin':{},'predefined_next_margin':32,'margin64_is_sensitivity_only':True,
            'automatic_training_authorized':False,'screen_is_engineering_eligibility_only':True}
    for margin in MARGINS:
        per_type={}
        for label in ('scratches','similar colour spot'):
            rows=[r for r in records if r['defect_type']==label]; q=f'Q{margin}'; regressions=[]; comparisons={}
            for metric in ('gt_rgb_mae','inner_band_3_rgb_mae','outer_band_3_rgb_mae','remaining_context_rgb_mae'):
                pairs=[(r['metrics']['F'][metric],r['metrics'][q][metric]) for r in rows]
                pairs=[(f,c) for f,c in pairs if f is not None and c is not None]
                fm,qm=(median(f for f,c in pairs),median(c for f,c in pairs)) if pairs else (None,None)
                delta=qm-fm if fm is not None else None
                stop=delta is not None and delta>.10*fm and delta>1/255
                comparisons[metric]={'F_median':fm,'Q_median':qm,'absolute_delta':delta,'material_regression':stop}
                if stop: regressions.append(metric)
            paired=[(r['metrics']['F']['internal_sobel_mae'],r['metrics'][q]['internal_sobel_mae']) for r in rows]
            paired=[(f,c) for f,c in paired if f is not None and c is not None and f>1e-8]
            fmed,qmed=(median(f for f,c in paired),median(c for f,c in paired)) if paired else (None,None)
            per_type[label]={'sample_count':len(rows),'material_boundary_regression':bool(regressions),'regressions':regressions,
                'comparisons':comparisons,'sobel':{'valid_pairs':len(paired),'NA_or_nearzero_excluded':len(rows)-len(paired),
                'F_median':fmed,'Q_median':qmed,'relative_median_improvement':(fmed-qmed)/fmed if fmed is not None else None,
                'direction_improved':sum(c<f for f,c in paired),'not_a_required_10percent_gate':True}}
        result['per_margin'][str(margin)]={'per_type':per_type,'engineering_eligible':not any(v['material_boundary_regression'] for v in per_type.values())}
    return result

def panel(identity,original,gt,results,box):
    columns=[('Real input',_rgb_image(original)),('GT (unchanged)',Image.fromarray(gt[0,0].cpu().byte().mul(255).numpy()).convert('RGB'))]+[(m,_rgb_image(results[m])) for m in METHODS]
    size,gap,label,top=256,12,24,32
    out=Image.new('RGB',(gap+len(columns)*(size+gap),top+2*(size+label+gap)),(18,22,30)); draw=ImageDraw.Draw(out)
    fontpath=Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'); font=ImageFont.truetype(str(fontpath),14) if fontpath.exists() else ImageFont.load_default()
    draw.text((gap,6),identity+' | geometry/reconstruction diagnosis only',font=font,fill='white')
    for i,(name,image) in enumerate(columns):
        x=gap+i*(size+gap); draw.text((x,top),name,font=font,fill='white')
        out.paste(ImageOps.pad(image,(size,size),method=Image.Resampling.NEAREST),(x,top+label))
        y=top+label+size+gap; draw.text((x,y),'Same fixed ROI32',font=font,fill='white')
        out.paste(ImageOps.pad(image.crop(box),(size,size),method=Image.Resampling.NEAREST),(x,y+label))
    return out

def run(*,output,device='cuda:0'):
    frozen=_read_json(output/'inputs.json'); protocol=Path(frozen['protocol_path']); protocol_check(protocol)
    if sha256_file(protocol)!=frozen['protocol_sha256'] or sha256_file(output/'registered_protocol.json')!=frozen['protocol_sha256']: raise ValueError('bridge protocol changed')
    code=_read_json(output/'code_manifest.json'); verify_code_manifest(code,worktree=Path(__file__).resolve().parents[3],source_paths=SOURCES)
    if code['inputs_sha256']!=sha256_file(output/'inputs.json') or code['protocol_sha256']!=frozen['protocol_sha256']: raise ValueError('bridge data/code freeze mismatch')
    p0=Path(frozen['p0'])
    if sha256_file(p0/'inputs.json')!=frozen['p0_inputs_sha256'] or sha256_file(p0/'private/metrics_by_method.jsonl')!=frozen['p0_metrics_sha256']: raise ValueError('old P0 source changed')
    old,samples,valid=source_samples(Path(frozen['root']),p0)
    vae_dir=Path(frozen['vae_directory'])
    if any(sha256_file(vae_dir/name)!=sha for name,sha in frozen['vae_files_sha256'].items()): raise ValueError('frozen local VAE changed')
    if any((output/name).exists() for name in ('sidecars','domains','metrics.json','summary.json')): raise FileExistsError('refuses to overwrite bridge results')
    from diffusers import AutoencoderKL
    torch.use_deterministic_algorithms(True); torch.backends.cudnn.benchmark=False
    vae=AutoencoderKL.from_pretrained(vae_dir,local_files_only=True,torch_dtype=torch.float32).to(device=device,dtype=torch.float32).eval().requires_grad_(False)
    validate_vae(vae)
    for name in ('sidecars','domains','rows'): (output/name).mkdir(exist_ok=False)
    if torch.device(device).type=='cuda': torch.cuda.reset_peak_memory_stats(device); torch.cuda.synchronize(device)
    start=time.monotonic(); records=[]; roundtrips=0
    for i,sample in enumerate(samples):
        expected=frozen['rows'][i]; gt=sample.mask[None].to(device); legal=valid[sample.sample_id][None].to(device)
        base=fixed_roi(gt,legal); domains=metric_domains(gt,legal,base)
        original=((sample.pixels+1)/2)[None].to(device=device,dtype=torch.float32)
        results={}; fractions={}; squares={}
        results['F'],fractions['F']=roundtrip512(vae,original); roundtrips+=1
        for margin in MARGINS:
            geometry=square_roi_geometry(gt,legal,margin)
            if json.loads(json.dumps(asdict(geometry)))!=expected['square_geometries'][str(margin)]: raise ValueError('ROI/letterbox geometry changed')
            square,mapped,squarevalid=forward_square(original,gt,geometry); square=square.to(device)
            squares[str(margin)]={'image':square,'GT':mapped,'valid':squarevalid}
            restored=inverse_square(square,geometry); results[f'Z{margin}']=paste_square_roi(original,restored,geometry); fractions[f'Z{margin}']=0.
            decoded,fraction=roundtrip512(vae,square); roundtrips+=1
            results[f'Q{margin}']=paste_square_roi(original,inverse_square(decoded,geometry),geometry); fractions[f'Q{margin}']=fraction
        metrics={m:{**reconstruction_metrics(original,results[m],domains),'preclamp_out_of_range_fraction_decode_grid':fractions[m]} for m in METHODS}
        repeat={k:{'old':expected['old_F_metrics'][k],'new':v,'absolute_difference':abs(v-expected['old_F_metrics'][k]) if v is not None else None}
                for k,v in metrics['F'].items() if k in expected['old_F_metrics']}
        repeated=all(v['old'] is None and v['new'] is None or v['absolute_difference'] is not None and v['absolute_difference']<=1e-6 for v in repeat.values())
        if not repeated: raise RuntimeError('fullF repeat differs from frozen P0 tolerance1e-6')
        folder=output/'sidecars'/expected['identity']; folder.mkdir()
        _rgb_image(original).save(folder/'real.png'); Image.fromarray(gt[0,0].cpu().byte().mul(255).numpy()).save(folder/'physical_gt_unchanged.png')
        paths={}
        for method in METHODS:
            path=folder/f'{method}.png'; _rgb_image(results[method]).save(path)
            paths[method]={'path':str(path.relative_to(output)),'sha256':sha256_file(path)}
        for margin,square in squares.items():
            _rgb_image(square['image']).save(folder/f'square-input-{margin}.png')
            for kind in ('GT','valid'): Image.fromarray(square[kind][0,0].byte().mul(255).numpy()).save(folder/f'square-{kind}-{margin}.png')
        domainfolder=output/'domains'/expected['identity']; domainfolder.mkdir()
        for name,value in domains.items(): Image.fromarray(value[0,0].cpu().byte().mul(255).numpy()).save(domainfolder/f'{name}.png')
        panel(expected['identity'],original,gt,results,base.box).save(output/'rows'/f'{expected["identity"]}.png')
        records.append({**expected,'metrics':metrics,'F_repeat':repeat,'F_repeat_passed':repeated,'paths':paths,
                        'domain_pixels':{k:int(v.sum()) for k,v in domains.items()},'outside_roi_exact':True,'diagnostic_only':True})
        print(json.dumps({'stage':'square_roi_bridge','completed':i+1,'total':9,'roundtrips':roundtrips,'identity':expected['identity']}),flush=True)
    if torch.device(device).type=='cuda': torch.cuda.synchronize(device)
    screen=screen_records(records)
    write_new_json(output/'metrics.json',{'records':records}); write_new_json(output/'numerical_screen.json',screen)
    with (output/'metrics.csv').open('x',newline='',encoding='utf-8') as f:
        names=list(records[0]['metrics']['F']); writer=csv.DictWriter(f,fieldnames=['identity','defect_type','method',*names]); writer.writeheader()
        for r in records:
            for m in METHODS: writer.writerow({'identity':r['identity'],'defect_type':r['defect_type'],'method':m,**r['metrics'][m]})
    summary={'schema_version':'v06_r_square_roi_bridge_results_v1','sample_count':9,'method_count':5,'vae_roundtrips':roundtrips,
        'vae_encode_decode_calls':roundtrips*2,'wall_seconds':time.monotonic()-start,'vae_dtype':'float32','posterior':'mode',
        'peak_gpu_bytes':torch.cuda.max_memory_allocated(device) if torch.device(device).type=='cuda' else None,
        'all_F_repeat_passed':True,'physical_gt_unchanged':True,'outside_roi_exact':True,'no_unet_or_adapter_loaded':True,
        'no_training':True,'no_test':True,'diagnostic_only':True,'predefined_next_margin':32,'margin64_is_sensitivity_only':True,
        'engineering_eligible_by_margin':{m:v['engineering_eligible'] for m,v in screen['per_margin'].items()},
        'numeric_screen_is_not_promotion':True,'protocol_sha256':frozen['protocol_sha256'],'output':str(output)}
    write_new_json(output/'summary.json',summary); del vae; gc.collect()
    return summary
