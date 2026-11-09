"""Original FP32-VAE Full/Crop inference algorithm with pre-clamp finite checks."""
import torch
from diffusers import DDIMScheduler
from cadiff.diffusion.vae_roundtrip import encode_mode,decode_latent
from cadiff.p2_a0.masking import build_latent_masks,build_masked_context
from cadiff.p2_b0.generation import denoise_masked_infill
from cadiff.v06_r.full_crop_inference import FloatRenderer as LegacyFloatRenderer

def finite(value,label):
    if not torch.isfinite(value).all(): raise RuntimeError('nonfinite '+label)
    return value

class FloatRenderer(LegacyFloatRenderer):
    @torch.inference_mode()
    def render_float(self,square,mask,defect_type,seed):
        if defect_type not in self.allowed_types or type(seed)!=int or seed<0: raise ValueError('invalid type/seed')
        if square.shape!=(1,3,512,512) or square.dtype!=torch.float32 or not torch.isfinite(square).all() or torch.any(square<0) or torch.any(square>1): raise ValueError('normal square must be finite FP32 RGB[0,1]')
        if mask.shape!=(1,1,512,512) or not torch.all((mask==0)|(mask==1)) or not mask.any() or mask.all(): raise ValueError('invalid mapped GT')
        pixels=square.to(self.device,dtype=torch.float32)*2-1
        encoded=finite(encode_mode(self.components.vae,pixels).detach(),'FP32 encoded latent')
        normal_latent=finite(encoded.to(dtype=torch.float16),'FP16 cast latent')
        if normal_latent.shape!=(1,4,64,64): raise ValueError('predictor latent must be4x64x64')
        latent_mask=build_latent_masks(mask.to(self.device,dtype=torch.float32),(64,64)).binary
        context=build_masked_context(normal_latent,latent_mask,torch.Generator(device=self.device).manual_seed(seed+10_000_000))
        noise=torch.randn(normal_latent.shape,generator=torch.Generator(device=self.device).manual_seed(seed),device=self.device,dtype=torch.float16)
        generated=finite(denoise_masked_infill(model=self.model,scheduler=DDIMScheduler.from_config(self.components.scheduler.config),normal_latent=normal_latent,context=context,binary_mask=latent_mask,prompt_embeds=self.prompt,defect_type=defect_type,noise=noise,steps=50),'generated latent')
        decoded=finite(decode_latent(self.components.vae,generated.float()).float(),'preclamp decoded float')
        return decoded.clamp(0,1)
