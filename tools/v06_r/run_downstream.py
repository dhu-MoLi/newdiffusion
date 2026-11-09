"""Explicit preparation and one isolated downstream condition; no implicit training."""
import argparse
import json
from pathlib import Path
from cadiff.v06_r.downstream import prepare,run,smoke,CONDITIONS,SEEDS
def main():
    p=argparse.ArgumentParser(description=__doc__)
    modes=p.add_mutually_exclusive_group(required=True)
    modes.add_argument('--prepare',action='store_true'); modes.add_argument('--run',action='store_true')
    modes.add_argument('--smoke',action='store_true'); modes.add_argument('--benchmark',action='store_true')
    p.add_argument('--assets',type=Path); p.add_argument('--pool-root',type=Path); p.add_argument('--eval-kit',type=Path)
    p.add_argument('--identities',type=Path); p.add_argument('--protocol',type=Path)
    p.add_argument('--output',type=Path,required=True); p.add_argument('--condition',choices=CONDITIONS); p.add_argument('--seed',type=int,choices=SEEDS)
    p.add_argument('--device',default='cuda:0'); p.add_argument('--resume',action='store_true')
    p.add_argument('--smoke-output',type=Path); p.add_argument('--smoke-updates',type=int,choices=(1,2),default=2)
    p.add_argument('--stop-after-epoch',type=int,help='Engineering interruption check only; never evaluates test early')
    a=p.parse_args()
    if a.prepare:
        if any(x is None for x in (a.assets,a.identities,a.protocol,a.pool_root,a.eval_kit)): p.error('--prepare requires assets/identities/protocol/pool-root/eval-kit')
        result=prepare(a.assets,a.identities,a.protocol,a.pool_root,a.eval_kit,a.output)
    elif a.smoke or a.benchmark:
        if a.smoke_output is None: p.error('engineering checks require --smoke-output outside formal evaluation')
        result=smoke(a.output,a.smoke_output,device=a.device,updates=32 if a.benchmark else a.smoke_updates)
    else:
        if a.condition is None or a.seed is None: p.error('--run requires condition/seed')
        result=run(a.output,a.condition,a.seed,device=a.device,resume=a.resume,stop_after_epoch=a.stop_after_epoch)
    print(json.dumps(result,ensure_ascii=False,allow_nan=False),flush=True)
if __name__=='__main__': main()
