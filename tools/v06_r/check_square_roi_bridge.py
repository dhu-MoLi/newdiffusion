"""Freeze nine-shot square geometry, freeze sources, then explicitly run frozen VAE."""
import argparse
import json
from pathlib import Path
from cadiff.v06_r.square_roi_bridge import prepare,freeze_code,run,PARAMETERS
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path); p.add_argument('--output',type=Path,required=True); p.add_argument('--protocol',type=Path)
    modes=p.add_mutually_exclusive_group(required=True)
    modes.add_argument('--freeze-only',action='store_true'); modes.add_argument('--freeze-code',action='store_true'); modes.add_argument('--run',action='store_true'); modes.add_argument('--parameters',action='store_true')
    p.add_argument('--device',default='cuda:0'); a=p.parse_args()
    if a.parameters: result=PARAMETERS
    elif a.freeze_only:
        if a.root is None or a.protocol is None: p.error('--freeze-only requires --root and --protocol')
        result=prepare(root=a.root,output=a.output,protocol=a.protocol)
    elif a.freeze_code: result=freeze_code(a.output)
    else: result=run(output=a.output,device=a.device)
    print(json.dumps(result,ensure_ascii=False,sort_keys=True),flush=True)
if __name__=='__main__': main()
