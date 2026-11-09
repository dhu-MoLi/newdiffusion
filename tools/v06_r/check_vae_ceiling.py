"""Freeze the nine-shot CPU protocol first; run only the frozen local VAE later."""
import argparse
import json
from pathlib import Path
from cadiff.v06_r.vae_ceiling import prepare_vae_ceiling, run_vae_ceiling, freeze_code_manifest

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path('/home/dell/2T/lm/new_projects/category_anomaly_diffusion/runs/v06_r/visa/chewinggum/20261012-strong-r'))
    parser.add_argument('--output',type=Path)
    parser.add_argument('--paths-yaml',type=Path)
    parser.add_argument('--protocol',type=Path)
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--freeze-only',action='store_true')
    mode.add_argument('--freeze-code',action='store_true')
    mode.add_argument('--run',action='store_true')
    parser.add_argument('--device',default='cuda:0')
    args=parser.parse_args()
    output=args.output or args.root/'vae_ceiling_p0'
    if args.freeze_only:
        if args.paths_yaml is None:
            parser.error('--freeze-only requires --paths-yaml')
        result=prepare_vae_ceiling(root=args.root,output=output,paths_yaml=args.paths_yaml,protocol_path=args.protocol)
    elif args.freeze_code:
        result=freeze_code_manifest(output=output)
    else:
        result=run_vae_ceiling(output=output,device=args.device)
    print(json.dumps(result,ensure_ascii=False,sort_keys=True))

if __name__=='__main__':
    main()
