"""Registered narrow scratch pilot, fresh Full/Crop paired training."""
import argparse,json
from pathlib import Path
from cadiff.v06_r.scratch_pilot_training import train_full_crop,load_pilot_config,VIEWS
from cadiff.v06_r.config import RExperimentConfig
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--view',choices=VIEWS,required=True)
    p.add_argument('--category',choices=('cashew','pcb1'),required=True)
    for name in ('protocol','paths-yaml','manifest','dataset-root','output'): p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--device',default='cuda:0'); p.add_argument('--smoke',action='store_true'); p.add_argument('--steps',type=int,default=1200)
    p.add_argument('--forced-timestep',type=int); p.add_argument('--resume',type=Path); p.add_argument('--stop-after',type=int)
    a=p.parse_args(); config=load_pilot_config(a.protocol,a.category,smoke=a.smoke,steps=a.steps)
    result=train_full_crop(view=a.view,category=a.category,protocol=a.protocol,paths_yaml=a.paths_yaml,manifest=a.manifest,dataset_root=a.dataset_root,output=a.output,config=config,experiment=RExperimentConfig('latent_control',.1,0.,0.,100),device=a.device,smoke=a.smoke,forced_timestep=a.forced_timestep,resume=a.resume,stop_after=a.stop_after)
    print(json.dumps(result,ensure_ascii=False),flush=True)
if __name__=='__main__': main()
