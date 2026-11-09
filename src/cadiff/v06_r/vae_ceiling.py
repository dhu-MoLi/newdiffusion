"""Fixed few-shot SD1.5 VAE reconstruction ceiling. No trainable renderer or A/H."""
from __future__ import annotations
from dataclasses import dataclass, asdict
import csv
import gc
import hashlib
import json
import math
from pathlib import Path
from importlib.metadata import version as package_version
import platform
import random
from statistics import median
from typing import Any
import numpy as np
import PIL
from PIL import Image, ImageDraw, ImageFont, ImageOps
import torch
import torch.nn.functional as F
from torch import Tensor
from cadiff.config import load_config
from cadiff.provenance import sha256_file
from cadiff.v06_c.visa_exact_mask_r import visa_p2_a1_config
from cadiff.v06_r.runner import load_train_manifest, write_new_json

METHODS = ("F", "Z", "N", "L")
LABELS = ("X1", "X2", "X3", "X4")
PRIMARY = ("scratches", "similar colour spot")
SOURCE_PATHS = (
    "src/cadiff/v06_r/vae_ceiling.py",
    "tools/v06_r/check_vae_ceiling.py",
    "src/cadiff/v06_r/__init__.py",
    "src/cadiff/v06_r/config.py",
    "src/cadiff/v06_r/runner.py",
    "src/cadiff/v06_c/visa_exact_mask_r.py",
    "src/cadiff/v06_c/visa_pilot.py",
    "src/cadiff/p2_a0/fewshot.py",
    "src/cadiff/p2_a0/config.py",
    "src/cadiff/p2_a0/fewshot_config.py",
    "src/cadiff/p2_a1/config.py",
    "src/cadiff/data/geometry.py",
    "src/cadiff/config.py",
    "src/cadiff/provenance.py",
)
PARAMETERS = {
    "image_size":512, "roi_margin":32, "padding_multiple":8,
    "padding_mode":"reflect", "padding_alignment":"symmetric",
    "local_long_edge":512, "resize_mode":"bilinear", "align_corners":False,
    "antialias":True, "size_rounding":"python_round", "vae_dtype":"float32",
    "posterior":"mode", "clamp_policy":"decoded_rgb_before_back_resize",
    "metrics_float_before_png_quantization":True, "boundary_norm":"linf",
    "boundary_radius":3, "edge_domain":"complete_3x3_gt", "blind_seed":2026101204,
    "blind_labels":list(LABELS), "methods":list(METHODS),
}

@dataclass(frozen=True)
class RoiGeometry:
    box: tuple[int,int,int,int]
    valid_box: tuple[int,int,int,int]
    native_hw: tuple[int,int]
    enlarged_hw: tuple[int,int]
    native_pad: tuple[int,int,int,int]
    enlarged_pad: tuple[int,int,int,int]
    scale_nominal: float
    scale_hw: tuple[float,float]

def _binary(value: Tensor, *, name: str) -> Tensor:
    if value.ndim != 4 or value.shape[0:2] != (1,1) or value.requires_grad or not torch.all((value==0)|(value==1)):
        raise ValueError(f"{name} must be detached binary 11HW")
    return value.bool()

def _bbox(mask: Tensor) -> tuple[int,int,int,int]:
    y,x = torch.nonzero(mask[0,0], as_tuple=True)
    if not len(x):
        raise ValueError("mask/domain must be nonempty")
    return int(x.min()),int(y.min()),int(x.max())+1,int(y.max())+1

def padding8(height: int, width: int) -> tuple[int,int,int,int]:
    dh,dw=(-height)%8,(-width)%8
    return dw//2,dw-dw//2,dh//2,dh-dh//2

def fixed_roi(ground_truth: Tensor, valid: Tensor) -> RoiGeometry:
    gt,legal=_binary(ground_truth,name="GT"),_binary(valid,name="valid")
    if gt.shape != legal.shape or torch.any(gt & ~legal):
        raise ValueError("complete GT must lie inside valid domain")
    vb=_bbox(legal)
    rect=torch.zeros_like(legal)
    rect[:,:,vb[1]:vb[3],vb[0]:vb[2]]=True
    if not torch.equal(rect,legal):
        raise ValueError("valid domain must be the original letterbox rectangle")
    gb=_bbox(gt)
    box=max(vb[0],gb[0]-32),max(vb[1],gb[1]-32),min(vb[2],gb[2]+32),min(vb[3],gb[3]+32)
    x0,y0,x1,y1=box
    if int(gt[:,:,y0:y1,x0:x1].sum()) != int(gt.sum()):
        raise ValueError("fixed clipped ROI must retain all GT")
    h,w=y1-y0,x1-x0
    scale=512/max(h,w)
    enlarged=max(1,int(round(h*scale))),max(1,int(round(w*scale)))
    return RoiGeometry(box,vb,(h,w),enlarged,padding8(h,w),padding8(*enlarged),scale,(enlarged[0]/h,enlarged[1]/w))

def crop_roi(image: Tensor, geometry: RoiGeometry) -> Tensor:
    x0,y0,x1,y1=geometry.box
    return image[:,:,y0:y1,x0:x1]

def pad_reflect(image: Tensor, padding: tuple[int,int,int,int]) -> Tensor:
    left,right,top,bottom=padding
    h,w=image.shape[-2:]
    if min(padding)<0 or max(left,right)>=w or max(top,bottom)>=h:
        raise ValueError("reflect padding must fit the original ROI dimensions")
    return F.pad(image,padding,mode="reflect") if any(padding) else image

def unpad(image: Tensor, padding: tuple[int,int,int,int]) -> Tensor:
    l,r,t,b=padding
    h,w=image.shape[-2:]
    return image[:,:,t:h-b if b else h,l:w-r if r else w]

def resize(image: Tensor, size: tuple[int,int]) -> Tensor:
    return F.interpolate(image,size=size,mode="bilinear",align_corners=False,antialias=True)

def enlarged_input(roi: Tensor, geometry: RoiGeometry) -> Tensor:
    return pad_reflect(resize(roi,geometry.enlarged_hw),geometry.enlarged_pad)

def enlarged_back(image: Tensor, geometry: RoiGeometry) -> Tensor:
    return resize(unpad(image,geometry.enlarged_pad),geometry.native_hw)

def resize_only_control(roi: Tensor, geometry: RoiGeometry) -> Tensor:
    return enlarged_back(enlarged_input(roi,geometry),geometry).clamp(0,1)

def validate_vae(vae: torch.nn.Module):
    floating=tuple(p for p in vae.parameters() if p.is_floating_point())
    if not floating or any(p.dtype != torch.float32 or p.requires_grad or p.grad is not None for p in floating) or any(m.training for m in vae.modules()):
        raise ValueError("VAE must be frozen gradient-free FP32 eval")
    scaling=float(vae.config.scaling_factor)
    if not math.isfinite(scaling) or scaling <= 0:
        raise ValueError("VAE scaling_factor must be positive finite")

@torch.inference_mode()
def vae_roundtrip(vae: Any, rgb: Tensor) -> tuple[Tensor,float]:
    validate_vae(vae)
    if rgb.dtype != torch.float32 or rgb.ndim != 4 or rgb.shape[:2] != (1,3) or not torch.isfinite(rgb).all() or torch.any(rgb<0) or torch.any(rgb>1):
        raise ValueError("roundtrip input must be finite FP32 RGB11 in [0,1]")
    if rgb.shape[-1]%8 or rgb.shape[-2]%8:
        raise ValueError("VAE spatial dimensions must be divisible by8")
    scaling=float(vae.config.scaling_factor)
    encoded=vae.encode(rgb*2-1).latent_dist.mode()
    latent=encoded*scaling
    raw=vae.decode(latent/scaling).sample.float()/2+.5
    if raw.shape != rgb.shape or not torch.isfinite(raw).all():
        raise ValueError("VAE reconstruction must be finite matching RGB")
    fraction=float(((raw<0)|(raw>1)).float().mean())
    return raw,fraction

def paste_roi(original: Tensor, roi: Tensor, geometry: RoiGeometry) -> Tensor:
    x0,y0,x1,y1=geometry.box
    if roi.shape[-2:] != geometry.native_hw:
        raise ValueError("restored ROI dimensions differ")
    result=original.clone()
    result[:,:,y0:y1,x0:x1]=roi
    return result

@torch.inference_mode()
def reconstruct_methods(vae: Any, original: Tensor, geometry: RoiGeometry):
    roi=crop_roi(original,geometry)
    full_raw,full_fraction=vae_roundtrip(vae,original)
    native_raw,native_fraction=vae_roundtrip(vae,pad_reflect(roi,geometry.native_pad))
    enlarged_raw,enlarged_fraction=vae_roundtrip(vae,enlarged_input(roi,geometry))
    # Clamp decoder RGB on its decode grid, then use the identical fixed
    # unpad/back-resize process as the VAE-free Z arm; no contrast enhancement.
    results={"F":full_raw.clamp(0,1),"Z":paste_roi(original,resize_only_control(roi,geometry),geometry),
        "N":paste_roi(original,unpad(native_raw.clamp(0,1),geometry.native_pad),geometry),
        "L":paste_roi(original,enlarged_back(enlarged_raw.clamp(0,1),geometry),geometry)}
    fractions={"F":full_fraction,"Z":0.,"N":native_fraction,"L":enlarged_fraction}
    raw_rois={"F":crop_roi(full_raw,geometry),"Z":crop_roi(results["Z"],geometry),
        "N":unpad(native_raw,geometry.native_pad),"L":unpad(enlarged_raw,geometry.enlarged_pad)}
    extra={m:{"preclamp_out_of_range_fraction_decode_grid":fractions[m],
              "preclamp_out_of_range_fraction_unpadded_roi":float(((raw_rois[m]<0)|(raw_rois[m]>1)).float().mean())} for m in METHODS}
    return results,extra

def metric_domains(gt: Tensor, valid: Tensor, geometry: RoiGeometry) -> dict[str,Tensor]:
    gt,valid=_binary(gt,name="GT"),_binary(valid,name="valid")
    if torch.any(gt & ~valid):
        raise ValueError("GT outside valid domain")
    gf=gt.float()
    dilated=F.max_pool2d(gf,7,stride=1,padding=3).bool()
    eroded=(F.conv2d(gf,gf.new_ones(1,1,7,7),padding=3)==49)
    edge=(F.conv2d(gf,gf.new_ones(1,1,3,3),padding=1)==9)
    box=torch.zeros_like(gt)
    x0,y0,x1,y1=geometry.box
    box[:,:,y0:y1,x0:x1]=True
    domain=box & valid
    return {"gt":gt,"internal_edge":edge,"inner_band_3":gt & ~eroded,
        "outer_band_3":dilated & ~gt & domain,"context":domain & ~gt,
        "remaining_context":domain & ~dilated,"roi":domain}

def _luminance(image: Tensor) -> Tensor:
    return image[:,0:1]*.299+image[:,1:2]*.587+image[:,2:3]*.114

def _sobel(image: Tensor) -> Tensor:
    gray=_luminance(image)
    k=image.new_tensor(((1,0,-1),(2,0,-2),(1,0,-1))).view(1,1,3,3)/8
    return torch.cat((F.conv2d(gray,k,padding=1),F.conv2d(gray,k.transpose(2,3),padding=1)),1)

def region_mean(error: Tensor, mask: Tensor) -> float|None:
    count=int(mask.sum())
    if not count:
        return None
    return float((error*mask).sum()/(count*error.shape[1]))

def reconstruction_metrics(original: Tensor, reconstruction: Tensor, domains: dict[str,Tensor]):
    if original.shape != reconstruction.shape or not torch.isfinite(reconstruction).all() or torch.any(reconstruction<0) or torch.any(reconstruction>1):
        raise ValueError("metrics require matching finite clamped RGB")
    error=(reconstruction-original).abs()
    return {"gt_rgb_mae":region_mean(error,domains["gt"]),
        "internal_sobel_mae":region_mean((_sobel(reconstruction)-_sobel(original)).abs(),domains["internal_edge"]),
        "gt_luminance_signed_bias":region_mean(_luminance(reconstruction)-_luminance(original),domains["gt"]),
        "inner_band_3_rgb_mae":region_mean(error,domains["inner_band_3"]),
        "outer_band_3_rgb_mae":region_mean(error,domains["outer_band_3"]),
        "context_rgb_mae":region_mean(error,domains["context"]),
        "remaining_context_rgb_mae":region_mean(error,domains["remaining_context"])}

def numerical_screen(records: list[dict]):
    output={"nearzero":1e-8,"median_sobel_improvement_required":.10,
            "boundary_relative_tolerance":.10,"boundary_absolute_tolerance":1/255,
            "per_type":{},"automatic_training_authorized":False,"human_review_required":True}
    for label,required in (("scratches",2),("similar colour spot",4)):
        rows=[r for r in records if r["defect_type"]==label]
        pairs=[(r["metrics"]["N"]["internal_sobel_mae"],r["metrics"]["L"]["internal_sobel_mae"]) for r in rows]
        valid=[(n,l) for n,l in pairs if n is not None and l is not None and n>1e-8]
        nmed,lmed=(median(n for n,l in valid),median(l for n,l in valid)) if valid else (None,None)
        relative=(nmed-lmed)/nmed if nmed is not None else None
        improved=sum(l<n for n,l in valid)
        regressions=[]
        for metric in ("inner_band_3_rgb_mae","outer_band_3_rgb_mae","remaining_context_rgb_mae"):
            for baseline in ("N","F"):
                pairs_region=[(r["metrics"][baseline][metric],r["metrics"]["L"][metric]) for r in rows]
                pairs_region=[(b,l) for b,l in pairs_region if b is not None and l is not None]
                if not pairs_region:
                    continue
                bm,lm=median(b for b,l in pairs_region),median(l for b,l in pairs_region)
                delta=lm-bm
                if delta>.10*bm and delta>1/255:
                    regressions.append({"metric":metric,"baseline":baseline,"baseline_median":bm,"L_median":lm,"absolute_increase":delta})
        output["per_type"][label]={"total_samples":len(rows),"valid_sobel_pairs":len(valid),
            "excluded_na_or_nearzero":len(rows)-len(valid),"N_sobel_median":nmed,"L_sobel_median":lmed,
            "relative_median_improvement":relative,"direction_improved":improved,"direction_required":required,
            "structure_screen":relative is not None and relative>=.10 and improved>=required,
            "boundary_or_context_stop":bool(regressions),"regressions":regressions}
    output["numeric_screen_passed"]=all(t["structure_screen"] and not t["boundary_or_context_stop"] for t in output["per_type"].values())
    return output

def _read_json(path:Path):
    return json.loads(path.read_text(encoding="utf-8"))

def runtime_versions():
    return {"torch":str(torch.__version__),"diffusers":package_version("diffusers"),
        "numpy":str(np.__version__),"pillow":str(PIL.__version__),
        "pyyaml":package_version("PyYAML"),"python":platform.python_version()}

def code_manifest_data(worktree:Path, *, source_paths=SOURCE_PATHS, versions=None):
    worktree=worktree.resolve(strict=True)
    files={}
    for relative in source_paths:
        rel=Path(relative)
        if rel.is_absolute() or ".." in rel.parts:
            raise ValueError("code source paths must be worktree-relative")
        path=(worktree/rel).resolve(strict=True)
        try:
            path.relative_to(worktree)
        except ValueError as error:
            raise ValueError("code source escapes worktree") from error
        if not path.is_file():
            raise ValueError("code source must be a regular file")
        files[relative]={"sha256":sha256_file(path),"size":path.stat().st_size}
    if not files or len(files)!=len(source_paths):
        raise ValueError("code source set must be nonempty and unique")
    return {"schema_version":"v06_r_vae_ceiling_code_manifest_v1","worktree":str(worktree),
        "versions":runtime_versions() if versions is None else versions,"files":files}

def verify_code_manifest(data, *, worktree:Path, source_paths=SOURCE_PATHS, versions=None):
    current=code_manifest_data(worktree,source_paths=source_paths,versions=versions)
    if data.get("schema_version")!=current["schema_version"] or data.get("worktree")!=current["worktree"]:
        raise ValueError("code manifest schema/worktree differs")
    if data.get("versions")!=current["versions"]:
        raise ValueError("frozen runtime library versions changed")
    if set(data.get("files",{}))!=set(current["files"]):
        raise ValueError("frozen code source set differs")
    if data["files"]!=current["files"]:
        raise ValueError("frozen code source SHA/size changed")

def _stage_has_results(output:Path):
    return any((output/name).exists() or (output/name).is_symlink()
        for name in ("private","reconstructions","diagnostics","blind","summary.json"))

def freeze_code_manifest(*, output:Path):
    """Append a source freeze only after data freeze and before any GPU result."""
    if _stage_has_results(output):
        raise FileExistsError("cannot freeze code after P0 result creation")
    frozen=_read_json(output/"inputs.json")
    protocol_path=Path(frozen["protocol_path"])
    _protocol(protocol_path)
    if sha256_file(protocol_path)!=frozen["protocol_sha256"] or sha256_file(output/"registered_protocol.json")!=frozen["protocol_sha256"]:
        raise ValueError("P0 protocol changed before code freeze")
    worktree=Path(_read_json(Path(frozen["root"])/"inputs.json")["worktree"]).resolve(strict=True)
    if Path(__file__).resolve().parents[3]!=worktree:
        raise ValueError("executing P0 module is outside the registered worktree")
    data=code_manifest_data(worktree)
    data.update({"protocol_sha256":frozen["protocol_sha256"],
        "frozen_inputs_sha256":sha256_file(output/"inputs.json"),"no_gpu_results_at_freeze":True})
    write_new_json(output/"code_manifest.json",data)
    return {"stage":"P0_code_frozen_cpu_only","source_count":len(data["files"]),
        "code_manifest_sha256":sha256_file(output/"code_manifest.json"),"output":str(output)}

def _verify_stage_code(output:Path,frozen):
    data=_read_json(output/"code_manifest.json")
    worktree=Path(_read_json(Path(frozen["root"])/"inputs.json")["worktree"]).resolve(strict=True)
    if Path(__file__).resolve().parents[3]!=worktree:
        raise ValueError("executing P0 module is outside the frozen worktree")
    verify_code_manifest(data,worktree=worktree)
    if data.get("protocol_sha256")!=frozen["protocol_sha256"] or data.get("frozen_inputs_sha256")!=sha256_file(output/"inputs.json"):
        raise ValueError("protocol/data freeze changed after code freeze")

def _protocol(path:Path):
    data=_read_json(path)
    if data.get("parameters") != PARAMETERS:
        raise ValueError("registered P0 protocol parameters differ from implementation")
    if data.get("schema_version") != "v06_r_vae_ceiling_registered_protocol_v1" or data.get("category") != "chewinggum" or data.get("primary_defect_types") != list(PRIMARY) or data.get("expected_counts") != {"scratches":3,"similar colour spot":6}:
        raise ValueError("registered P0 schema/category/primary counts differ")
    numeric=data.get("numeric_screen",{})
    expected={"gradient_median_relative_improvement_minimum":.10,
        "minimum_improved_samples":{"scratches":2,"similar colour spot":4},
        "near_zero_denominator":1e-8,"boundary_or_context_relative_regression_maximum":.10,
        "boundary_or_context_absolute_regression_maximum":1/255}
    if any(numeric.get(k)!=v for k,v in expected.items()):
        raise ValueError("registered numerical screening thresholds differ")
    flags={"human_review_required_after_numeric_screen":True,"automatic_scale_r_training":False,
        "formal_a_h_changed":False,"heldout_anomalies_allowed":False,
        "latent_residual_reinjection_allowed":False,"candidate_selection_allowed":False,"local_files_only":True}
    if any(data.get(k) is not v for k,v in flags.items()):
        raise ValueError("registered P0 scope constraints differ")
    return data

def _samples(root:Path):
    source=_read_json(root/"inputs.json")
    manifest=Path(source["training_manifest"])
    if sha256_file(manifest)!=source["training_manifest_sha256"]:
        raise ValueError("frozen few-shot manifest changed")
    config=visa_p2_a1_config(source["category"],tuple(source["defect_types"]))
    samples,valid,_=load_train_manifest(manifest,Path(source["dataset_root"]),config)
    primary=tuple(s for s in samples if s.defect_type in PRIMARY)
    if len(primary)!=9 or sum(s.defect_type=="scratches" for s in primary)!=3 or sum(s.defect_type=="similar colour spot" for s in primary)!=6:
        raise ValueError("P0 requires all frozen3/6 primary fewshots")
    return source,primary,valid

def _identity(sample,valid):
    geometry=fixed_roi(sample.mask.unsqueeze(0),valid.unsqueeze(0))
    domains=metric_domains(sample.mask.unsqueeze(0),valid.unsqueeze(0),geometry)
    return {"identity":sample.defect_type.replace(" ","_")+"-"+Path(sample.image_relpath).stem,
        "sample_id":sample.sample_id,"category":"chewinggum","defect_type":sample.defect_type,"role":sample.role,
        "image_relpath":sample.image_relpath,"mask_relpath":sample.mask_relpath,
        "image_sha256":sample.image_sha256,"mask_sha256":sample.mask_sha256,
        "geometry":json.loads(json.dumps(asdict(geometry))),"domain_pixels":{k:int(v.sum()) for k,v in domains.items()},
        "method_geometry":{"F":{"vae_hw":[512,512],"pad":[0,0,0,0]},
            "N":{"vae_hw":[geometry.native_hw[0]+geometry.native_pad[2]+geometry.native_pad[3],geometry.native_hw[1]+geometry.native_pad[0]+geometry.native_pad[1]],"pad":list(geometry.native_pad)},
            "Z_L_shared":{"resized_hw":list(geometry.enlarged_hw),"padded_hw":[geometry.enlarged_hw[0]+geometry.enlarged_pad[2]+geometry.enlarged_pad[3],geometry.enlarged_hw[1]+geometry.enlarged_pad[0]+geometry.enlarged_pad[1]],"pad":list(geometry.enlarged_pad)}}}

def prepare_vae_ceiling(*,root:Path,output:Path,paths_yaml:Path,protocol_path:Path|None=None):
    paths_yaml=paths_yaml.resolve(strict=True)
    protocol_path=protocol_path or root/"vae_ceiling_protocol.json"
    protocol=_protocol(protocol_path)
    source,samples,valid=_samples(root)
    if protocol.get("training_manifest_sha256") != source["training_manifest_sha256"] or protocol.get("root") != str(root):
        raise ValueError("registered protocol data root/manifest SHA differs")
    paths=load_config(paths_yaml)
    vae_dir=paths.model_root/"vae"
    if not (vae_dir/"config.json").is_file():
        raise FileNotFoundError("local VAE config is absent")
    fingerprints={p.name:sha256_file(p) for p in sorted(vae_dir.iterdir()) if p.is_file() and (p.name=="config.json" or p.suffix in (".safetensors",".bin"))}
    if len(fingerprints)<2:
        raise FileNotFoundError("local VAE weights are absent")
    frozen={"schema_version":"v06_r_vae_ceiling_inputs_v1","root":str(root),"protocol_path":str(protocol_path),
        "protocol_sha256":sha256_file(protocol_path),"paths_yaml":str(paths_yaml),"paths_sha256":sha256_file(paths_yaml),
        "training_manifest":source["training_manifest"],"training_manifest_sha256":source["training_manifest_sha256"],
        "vae_directory":str(vae_dir),"vae_files_sha256":fingerprints,
        "sample_count":9,"heldout_images_loaded":0,"rows":[_identity(s,valid[s.sample_id]) for s in samples],
        "metric_convention":"float clamped[0,1] before PNG round255; no contrast enhancement",
        "blind_display":"All four reconstruction columns show fixed ROI pasted into original RGB; F still encodes the entire512 image. Full reconstruction sidecars are separate.",
        "domain_convention":"GT RGB /3 channels; luminance Sobel /2 gradient directions at full3x3GT centers; square L∞radius3 bands; null for empty regions"}
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refuses to overwrite P0 stage: {output}")
    output.mkdir(parents=True,exist_ok=False)
    # Preserve the authority file's exact bytes, including its scope and screen.
    with (output/"registered_protocol.json").open("xb") as f:
        f.write(protocol_path.read_bytes())
    write_new_json(output/"inputs.json",frozen)
    return {"stage":"P0_protocol_frozen_cpu_only","sample_count":9,"output":str(output),"protocol_sha256":frozen["protocol_sha256"]}

def _rgb_image(tensor:Tensor):
    pixels=tensor[0].detach().cpu().clamp(0,1).mul(255).round().byte().permute(1,2,0).numpy()
    return Image.fromarray(pixels)

def _font(size=18):
    path=Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    return ImageFont.truetype(str(path),size) if path.is_file() else ImageFont.load_default()

def _visual_row(identity,original,mask,reconstructions,geometry,tile_size):
    normal=_rgb_image(original)
    gt=Image.fromarray(mask[0,0].cpu().byte().mul(255).numpy()).convert("RGB")
    columns=[("Real input",normal),("Official GT",gt)]+[(l,_rgb_image(reconstructions[l])) for l in LABELS]
    gap,top,label=12,36,28
    out=Image.new("RGB",(gap+len(columns)*(tile_size+gap),top+2*(tile_size+gap+label)),(18,22,30))
    draw=ImageDraw.Draw(out)
    draw.text((gap,6),identity+" | deterministic reconstruction only",font=_font(17),fill="white")
    for i,(name,image) in enumerate(columns):
        x=gap+i*(tile_size+gap)
        draw.text((x,top),name,font=_font(17),fill="white")
        out.paste(ImageOps.pad(image,(tile_size,tile_size),method=Image.Resampling.NEAREST,color=(18,22,30)),(x,top+label))
        y=top+label+tile_size+gap
        draw.text((x,y),"Fixed GT bbox +32",font=_font(17),fill="white")
        crop=image.crop(geometry.box)
        out.paste(ImageOps.pad(crop,(tile_size,tile_size),method=Image.Resampling.NEAREST,color=(18,22,30)),(x,y+label))
    return out

def run_vae_ceiling(*,output:Path,device:str="cuda:0"):
    frozen=_read_json(output/"inputs.json")
    _verify_stage_code(output,frozen)
    protocol_path=Path(frozen["protocol_path"])
    _protocol(protocol_path)
    if sha256_file(protocol_path)!=frozen["protocol_sha256"] or sha256_file(output/"registered_protocol.json")!=frozen["protocol_sha256"]:
        raise ValueError("P0 protocol changed after freeze")
    registered_paths=Path(frozen["paths_yaml"])
    if not registered_paths.is_absolute():
        registered_paths=Path(_read_json(Path(frozen["root"])/"inputs.json")["worktree"])/registered_paths
    if sha256_file(registered_paths)!=frozen["paths_sha256"]:
        raise ValueError("P0 paths config changed")
    root=Path(frozen["root"])
    source,samples,valid=_samples(root)
    if [_identity(s,valid[s.sample_id]) for s in samples]!=frozen["rows"]:
        raise ValueError("P0 fewshot identity/hash/geometry changed")
    vae_dir=Path(frozen["vae_directory"])
    if any(sha256_file(vae_dir/name)!=digest for name,digest in frozen["vae_files_sha256"].items()):
        raise ValueError("frozen local VAE files changed")
    result_names=("private","reconstructions","diagnostics","blind","summary.json")
    if _stage_has_results(output):
        raise FileExistsError("refuses to overwrite any existing P0 results")
    from diffusers import AutoencoderKL
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False
    vae=AutoencoderKL.from_pretrained(vae_dir,local_files_only=True,torch_dtype=torch.float32)
    vae=vae.to(device=torch.device(device),dtype=torch.float32).eval().requires_grad_(False)
    validate_vae(vae)
    for name in result_names[:-1]:
        (output/name).mkdir(exist_ok=False)
    (output/"blind"/"rows").mkdir()
    shuffled=list(METHODS)
    random.Random(PARAMETERS["blind_seed"]).shuffle(shuffled)
    mapping=dict(zip(LABELS,shuffled))
    write_new_json(output/"private"/"blind_key.json",{"seed":PARAMETERS["blind_seed"],"mapping":mapping})
    for label in LABELS:
        (output/"reconstructions"/label).mkdir()
        (output/"diagnostics"/label).mkdir()
    records,pieces=[],[]
    for index,sample in enumerate(samples,1):
        identity=frozen["rows"][index-1]
        gt=sample.mask.unsqueeze(0).to(device)
        legal=valid[sample.sample_id].unsqueeze(0).to(device)
        geometry=fixed_roi(gt,legal)
        domains=metric_domains(gt,legal,geometry)
        original=((sample.pixels+1)/2).unsqueeze(0).to(device=device,dtype=torch.float32)
        results,extra=reconstruct_methods(vae,original,geometry)
        metrics={m:{**reconstruction_metrics(original,results[m],domains),**extra[m]} for m in METHODS}
        blinded={l:paste_roi(original,crop_roi(results[m],geometry),geometry) for l,m in mapping.items()}
        paths={}
        for label in LABELS:
            full=output/"reconstructions"/label/(identity["identity"]+".png")
            diagnostic=output/"diagnostics"/label/(identity["identity"]+"-roi-pasted.png")
            _rgb_image(results[mapping[label]]).save(full)
            _rgb_image(blinded[label]).save(diagnostic)
            paths[label]={"reconstruction":str(full.relative_to(output)),"reconstruction_sha256":sha256_file(full),
                "roi_pasted_diagnostic":str(diagnostic.relative_to(output)),"diagnostic_sha256":sha256_file(diagnostic)}
        record={**identity,"metrics":metrics,"paths":paths,"diagnostic_only":True,"no_generated_label":True}
        records.append(record)
        row_path=output/"blind"/"rows"/(identity["identity"]+".png")
        _visual_row(identity["identity"],original,gt,blinded,geometry,384).save(row_path)
        pieces.append(_visual_row(identity["identity"],original,gt,blinded,geometry,256))
        with (output/"private"/"metrics_by_method.jsonl").open("a",encoding="utf-8") as f:
            f.write(json.dumps(record,ensure_ascii=False,allow_nan=False)+"\n")
        print(json.dumps({"stage":"vae_ceiling_p0","completed":index,"total":9,"identity":identity["identity"]}),flush=True)
    sheet=Image.new("RGB",(pieces[0].width,sum(p.height for p in pieces)),(18,22,30))
    y=0
    for piece in pieces:
        sheet.paste(piece,(0,y)); y+=piece.height
    sheet.save(output/"blind"/"all9-reconstruction.png")
    screen=numerical_screen(records)
    write_new_json(output/"private"/"numerical_screen.json",screen)
    metric_names=list(records[0]["metrics"]["F"])
    with (output/"private"/"metrics_by_method.csv").open("x",encoding="utf-8",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=["identity","defect_type","method",*metric_names])
        writer.writeheader()
        for record in records:
            for method in METHODS:
                writer.writerow({"identity":record["identity"],"defect_type":record["defect_type"],"method":method,**record["metrics"][method]})
    public={"schema_version":"v06_r_vae_ceiling_results_v1","sample_count":9,"reconstruction_count":36,
        "blind_labels":list(LABELS),"metrics":"private/metrics_by_method.csv","protocol_sha256":frozen["protocol_sha256"],
        "blind_panel":"blind/all9-reconstruction.png","row_panels":"blind/rows/","diagnostics":"diagnostics/",
        "diagnostic_only":True,"no_unet_or_adapter_loaded":True,"no_training":True,"no_new_mask":True,
        "vae_dtype":"float32","posterior":"mode","human_vae_ceiling_review":"pending",
        "numeric_screen_is_not_training_authorization":True,"output":str(output)}
    write_new_json(output/"summary.json",public)
    del vae
    gc.collect()
    if torch.device(device).type=="cuda":
        torch.cuda.empty_cache()
    return public
