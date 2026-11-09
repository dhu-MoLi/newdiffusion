"""New Full/Crop32 detectors with exact previous plans and locked Real-FS references."""
import argparse,json
from pathlib import Path
from cadiff.v06_r.full_crop_rep2_downstream import prepare,run,CONDITIONS,SEEDS
def main():
    p=argparse.ArgumentParser(description=__doc__)
    mode=p.add_mutually_exclusive_group(required=True); mode.add_argument('--prepare',action='store_true'); mode.add_argument('--run',action='store_true')
    p.add_argument('--old-prepared',type=Path); p.add_argument('--identities',type=Path); p.add_argument('--protocol',type=Path); p.add_argument('--pool-root',type=Path)
    p.add_argument('--output',type=Path,required=True); p.add_argument('--condition',choices=CONDITIONS); p.add_argument('--seed',type=int,choices=SEEDS)
    p.add_argument('--device',default='cuda:0'); p.add_argument('--resume',action='store_true'); p.add_argument('--stop-after-epoch',type=int)
    a=p.parse_args()
    if a.prepare:
        if any(x is None for x in (a.old_prepared,a.identities,a.protocol,a.pool_root)): p.error('--prepare requires old-prepared/identities/protocol/pool-root')
        result=prepare(old_prepared=a.old_prepared,identities=a.identities,protocol=a.protocol,pool_root=a.pool_root,output=a.output)
    else:
        if a.condition is None or a.seed is None: p.error('--run requires condition/seed')
        result=run(a.output,a.condition,a.seed,device=a.device,resume=a.resume,stop_after_epoch=a.stop_after_epoch)
    print(json.dumps(result,ensure_ascii=False),flush=True)
if __name__=='__main__': main()
