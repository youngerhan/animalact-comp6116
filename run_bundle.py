"""Portable bundle entry point. Run via the adjacent Windows command files."""
import argparse
import datetime
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parent
PROJECT=ROOT/'AnimalAct-SlowFast'


def run(args,cwd=PROJECT):
    subprocess.run([sys.executable,*map(str,args)],cwd=cwd,check=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['install','verify','predict','smoke','train32','train-full','blend'])
    parser.add_argument('--video',type=Path)
    args=parser.parse_args()
    if args.mode=='install':
        run(['-m','ensurepip','--upgrade'],ROOT)
        run(['-m','pip','install','--no-index','--find-links',ROOT/'wheelhouse','-r',ROOT/'requirements-windows.lock'],ROOT)
        run(['-m','pip','install','--no-index','--no-deps','--no-build-isolation','-e',ROOT/'mmaction2-main'],ROOT)
        run(['-m','pip','check'],ROOT)
        print('Offline installation completed. Next: VERIFY_RESULTS.cmd',flush=True)
        return
    # Refuse to silently use packages from the original author's machine.
    import torch
    import mmaction
    if not Path(torch.__file__).resolve().is_relative_to(ROOT/'runtime'):
        raise RuntimeError('Torch was not imported from this bundle runtime.')
    if not Path(mmaction.__file__).resolve().is_relative_to(ROOT/'mmaction2-main'):
        raise RuntimeError('MMAction2 was not imported from this bundle source.')
    print('Python:',sys.executable,'CUDA:',torch.cuda.is_available(),flush=True)
    if args.mode=='verify':
        run(['-m','unittest','-v','test_pipeline'])
        run(['verify.py'])
    elif args.mode=='predict':
        if not torch.cuda.is_available():raise RuntimeError('Video prediction requires a compatible NVIDIA GPU/driver; saved metric verification does not.')
        video=args.video or ROOT/'AnimalAct/videos/valid/FLBGMVYU.mp4'
        extra=[] if args.video else ['--start-frame','22','--end-frame','96']
        run(['predict.py',video,*extra])
    else:
        stamp=datetime.datetime.now().strftime('%Y%m%d-%H%M%S-%f')
        if args.mode=='blend':
            out=ROOT/'reproduced'/('blend-'+stamp)
            run(['blend.py','--base','runs/ensemble7','--new','runs/pilot32-whole224','--out',out,'--allow-legacy'])
            saved=json.loads((out/'test_metrics.json').read_text(encoding='utf-8'))
            expected=json.loads((PROJECT/'runs/pilot32-blend/test_metrics.json').read_text(encoding='utf-8'))
            for key in ('mAP','macro_F1'):
                if abs(saved[key]-expected[key])>1e-12:raise RuntimeError(f'Reproduced {key} disagrees')
            print({key:saved[key] for key in ('mAP','macro_F1','threshold')},flush=True)
        else:
            if not torch.cuda.is_available():raise RuntimeError('Training requires a compatible NVIDIA GPU/driver.')
            out=ROOT/'reproduced'/(args.mode+'-'+stamp)
            if args.mode=='smoke':
                run(['train.py','--smoke','--manifest','data/smoke.json','--out',out,'--epochs','1',
                     '--frames','8','--image-size','64','--batch-size','1','--workers','0','--log-every','4'])
            else:
                recipe='recipes/historical/pilot32-whole224.json' if args.mode=='train32' else 'recipes/whole224-clean.json'
                run(['reproduce.py',recipe,'--out',out,'--execute'])
        print('Output:',out,flush=True)


if __name__=='__main__':
    main()
