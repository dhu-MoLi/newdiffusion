"""Fresh matched single-view training; frozen reference supplies config only."""
import argparse
import json
from pathlib import Path
from cadiff.v06_r.full_crop_rep2_training import train_full_crop,load_reference_config,PARAMETERS,VIEWS
from cadiff.v06_r.config import RExperimentConfig
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--view',choices=VIEWS,required=True)
    for name in ('reference-checkpoint','protocol','paths-yaml','manifest','dataset-root','output'): p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--device',default='cuda:0'); p.add_argument('--smoke',action='store_true'); p.add_argument('--steps',type=int,default=1200)
    p.add_argument('--forced-timestep',type=int); p.add_argument('--resume',type=Path); p.add_argument('--stop-after',type=int)
    a=p.parse_args(); config=load_reference_config(a.reference_checkpoint,smoke=a.smoke,steps=a.steps)
    result=train_full_crop(view=a.view,reference_checkpoint=a.reference_checkpoint,protocol=a.protocol,
        paths_yaml=a.paths_yaml,manifest=a.manifest,dataset_root=a.dataset_root,output=a.output,config=config,
        experiment=RExperimentConfig('latent_control',.1,0.,0.,100),device=a.device,smoke=a.smoke,
        forced_timestep=a.forced_timestep,resume=a.resume,stop_after=a.stop_after)
    print(json.dumps(result,ensure_ascii=False),flush=True)
if __name__=='__main__': main()
