"""True FP32 VAE with original FP16 latent DDIM; float output before physical composition."""
from dataclasses import replace
from pathlib import Path
import json
import numpy as np
import torch
from diffusers import AutoencoderKL,DDIMScheduler
from cadiff.diffusion.components import load_frozen_components
from cadiff.diffusion.ddim_inversion import encode_prompt
from cadiff.diffusion.vae_roundtrip import encode_mode,decode_latent
from cadiff.p2_a0.masking import build_latent_masks,build_masked_context
from cadiff.p2_a0.model import P2InfillNoisePredictor
from cadiff.p2_b0.generation import denoise_masked_infill
from cadiff.v06_c.visa_exact_mask_r import CHECKPOINT_SCHEMA
from cadiff.v06_c.can_exact_mask_r import load_can_r_modules
from cadiff.v06_r.runner import P2A1Config,validate_training_config
from cadiff.v06_r.vae_ceiling import validate_vae
from cadiff.v06_r.full_crop_rep2_training import VIEWS,PARAMETERS
from cadiff.v06_r.square_roi_bridge import square_roi_geometry,forward_square,inverse_square,paste_square_roi
from cadiff.v06_b.renderer import compose_causal
from cadiff.provenance import sha256_file

def checkpoint_config(checkpoint,view,*,allow_smoke=False):
    data=json.loads((checkpoint/'metadata.json').read_text())
    if data.get('schema_version')!=CHECKPOINT_SCHEMA or data.get('identity',{}).get('view')!=view or view not in VIEWS:
        raise ValueError('not a declared Full/Crop32 checkpoint')
    steps=data.get('completed_steps')
    if (not allow_smoke and steps!=1200) or (allow_smoke and (type(steps)!=int or not 1<=steps<=4)):
        raise ValueError('only fixed final1200 or explicit <=4 smoke checkpoint')
    exp=data['experiment']
    if exp['arm']!='latent_control' or exp['rgb_weight'] or exp['edge_weight'] or exp['save_every']!=100:
        raise ValueError('checkpoint does not use unchanged control loss')
    raw=data['config']; raw={**raw,'defect_types':tuple(raw['defect_types']),'eval_timesteps':tuple(raw['eval_timesteps'])}
    config=P2A1Config(**raw); validate_training_config(config,smoke=allow_smoke,forced_timestep=None)
    for key in ('seed','context_fill_seed','noise_seed','type_rank','learning_rate'):
        if getattr(config,key)!=PARAMETERS[key]: raise ValueError('checkpoint protocol differs')
    for filename in ('adapter.safetensors','conditioner.safetensors'):
        if sha256_file(checkpoint/filename)!=data['files'][filename]['sha256']: raise ValueError('checkpoint weights changed')
    return config,data

class FloatRenderer:
    def __init__(self,*,components,model,prompt,device,allowed_types):
        self.components,self.model,self.prompt=components,model,prompt
        self.device=torch.device(device); self.allowed_types=tuple(allowed_types)
        validate_vae(components.vae)
    @torch.inference_mode()
    def render_float(self,square,mask,defect_type,seed):
        if defect_type not in self.allowed_types or type(seed)!=int or seed<0: raise ValueError('invalid type/seed')
        if square.shape!=(1,3,512,512) or square.dtype!=torch.float32 or not torch.isfinite(square).all() or torch.any(square<0) or torch.any(square>1):
            raise ValueError('normal square must be finite FP32 RGB[0,1]')
        if mask.shape!=(1,1,512,512) or not torch.all((mask==0)|(mask==1)) or not mask.any() or mask.all(): raise ValueError('invalid mapped GT')
        pixels=square.to(self.device,dtype=torch.float32)*2-1
        normal_latent=encode_mode(self.components.vae,pixels).detach().to(dtype=torch.float16)
        if normal_latent.shape!=(1,4,64,64): raise ValueError('predictor latent must be4x64x64')
        latent_mask=build_latent_masks(mask.to(self.device,dtype=torch.float32),(64,64)).binary
        context=build_masked_context(normal_latent,latent_mask,torch.Generator(device=self.device).manual_seed(seed+10_000_000))
        noise=torch.randn(normal_latent.shape,generator=torch.Generator(device=self.device).manual_seed(seed),device=self.device,dtype=torch.float16)
        generated=denoise_masked_infill(model=self.model,scheduler=DDIMScheduler.from_config(self.components.scheduler.config),normal_latent=normal_latent,
            context=context,binary_mask=latent_mask,prompt_embeds=self.prompt,defect_type=defect_type,noise=noise,steps=50)
        return decode_latent(self.components.vae,generated.float()).float().clamp(0,1)

def load_float_renderer(checkpoint,*,view,model_root,device='cuda:0',allow_smoke=False):
    config,metadata=checkpoint_config(checkpoint,view,allow_smoke=allow_smoke)
    components=load_frozen_components(model_root,torch.device(device),torch.float16)
    # Load original FP32 weights directly; promoting quantized FP16 weights is not equivalent.
    vae=AutoencoderKL.from_pretrained(model_root,subfolder='vae',local_files_only=True,torch_dtype=torch.float32).to(device=device,dtype=torch.float32).eval().requires_grad_(False)
    components=replace(components,vae=vae)
    _,conditioner,adapter,_=load_can_r_modules(checkpoint,config=config,device=torch.device(device),expected_schema=CHECKPOINT_SCHEMA)
    model=P2InfillNoisePredictor(components.unet,conditioner,adapter).eval()
    prompt=encode_prompt(components,config.normal_prompt,torch.device(device)).detach().clone()
    return FloatRenderer(components=components,model=model,prompt=prompt,device=device,allowed_types=config.defect_types)

def quantize(image):
    return image[0].detach().cpu().clamp(0,1).mul(255).round().byte().permute(1,2,0).numpy().copy()

def render_physical(renderer,normal,physical_mask,row,view):
    if view not in VIEWS: raise ValueError('unknown view')
    original=torch.from_numpy(normal.copy()).permute(2,0,1)[None].float()/255
    gt=torch.from_numpy(physical_mask.copy())[None,None].float()
    legal=torch.zeros_like(gt); x0,y0,x1,y1=row['valid_rectangle']; legal[:,:,y0:y1,x0:x1]=1
    if torch.any(gt>legal): raise ValueError('physical H leaves valid domain')
    geometry=None
    if view=='crop32':
        geometry=square_roi_geometry(gt,legal,32); square,mapped,valid=forward_square(original,gt,geometry)
    else: square,mapped=original,gt
    decoded=renderer.render_float(square,mapped,row['defect_type'],row['renderer_seed'])
    if decoded.dtype!=torch.float32 or decoded.shape!=(1,3,512,512) or not torch.isfinite(decoded).all(): raise ValueError('float renderer output invalid')
    if view=='crop32': raw_float=paste_square_roi(original.to(decoded.device),inverse_square(decoded,geometry),geometry)
    else: raw_float=decoded.clamp(0,1)
    final_float=torch.where(gt.to(raw_float.device).bool(),raw_float,original.to(raw_float.device))
    raw=quantize(raw_float); final=compose_causal(normal,raw,physical_mask)
    if not np.array_equal(quantize(final_float),final): raise ValueError('float composition/physical byte composition differ')
    return raw,final,geometry
