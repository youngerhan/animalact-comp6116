"""Regenerate the final ensemble from all eight best checkpoints into new output folders."""
import argparse
import datetime
import json
import subprocess
import sys
from pathlib import Path


def main():
    root=Path(__file__).resolve().parent; project=root/'AnimalAct-SlowFast'
    sys.path.insert(0,str(project))
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dry-run',action='store_true')
    args=parser.parse_args()
    from clips import read_json,write_json
    members=read_json(project/'runs/ensemble7/members.json')['members']
    names=[Path(name.replace('\\','/')).name for name in members]+['pilot32-whole224']
    variants={}
    for name in names:
        run=project/'runs'/name
        if not (run/'best.pth').is_file():raise FileNotFoundError(run/'best.pth')
        summary=run/'test_summary.json'
        variants[name]=read_json(summary)['selected'] if summary.exists() else 'training validation recipe'
    if args.dry_run:
        print(json.dumps(variants,indent=2)); return
    import argparse as ap
    import torch
    from mmaction.apis import init_recognizer
    from mmengine.config import Config
    from metrics import select_threshold,per_class_thresholds
    from runtime import run_report
    from evaluate_run import VARIANTS,scores
    if not torch.cuda.is_available():raise RuntimeError('Checkpoint evaluation requires a compatible NVIDIA GPU/driver.')
    torch.set_num_threads(2); torch.backends.cudnn.benchmark=False
    out=root/'reproduced'/('all-checkpoints-'+datetime.datetime.now().strftime('%Y%m%d-%H%M%S-%f'))
    out.mkdir(parents=True)
    mean=torch.tensor([123.675,116.28,103.53],device='cuda').view(1,3,1,1,1)
    std=torch.tensor([58.395,57.12,57.375],device='cuda').view(1,3,1,1,1)
    for name in names:
        run=project/'runs'/name; settings=ap.Namespace(**read_json(run/'settings.json'))
        manifest=read_json(Path(settings.manifest)); classes=manifest['classes']
        best=torch.load(run/'best.pth',map_location='cpu',weights_only=True)
        if best['classes']!=classes:raise ValueError('Checkpoint class mapping differs from manifest')
        config=Config.fromfile(settings.config); config.model.cls_head.num_classes=len(classes)
        model=init_recognizer(config,checkpoint=None,device='cuda:0'); model.load_state_dict(best['state_dict'])
        variant=(next(v for v in VARIANTS if v['name']==variants[name]) if variants[name]!='training validation recipe'
                 else dict(name='plain',zoom=settings.eval_zoom,tta=False,crops=(.5,),windows=((0.,1.),)))
        target=out/name; target.mkdir()
        print('Evaluating',name,variant['name'],flush=True)
        valid=scores(model,manifest['splits']['valid'],settings,mean,std,variant)
        threshold=select_threshold(*valid); thresholds=per_class_thresholds(*valid)
        run_report(target,'valid',valid,classes,threshold,thresholds,manifest['splits']['valid'])
        test=scores(model,manifest['splits']['test'],settings,mean,std,variant)
        run_report(target,'test',test,classes,threshold,thresholds,manifest['splits']['test'])
        write_json(target/'evaluation.json',dict(source=str(run),variant=variant['name'],threshold=threshold))
        del model,best; torch.cuda.empty_cache()
    subprocess.run([sys.executable,str(project/'ensemble.py'),'--runs',*[str(out/name) for name in names[:-1]],
                    '--out',str(out/'ensemble7')],cwd=project,check=True)
    subprocess.run([sys.executable,str(project/'blend.py'),'--base',str(out/'ensemble7'),'--new',str(out/names[-1]),
                    '--out',str(out/'final-blend')],cwd=project,check=True)
    measured=read_json(out/'final-blend/test_metrics.json')
    reference=read_json(project/'runs/pilot32-blend/test_metrics.json')
    comparison={key:dict(reproduced=measured[key],archived=reference[key],delta_pp=100*(measured[key]-reference[key]))
                for key in ('mAP','macro_F1')}
    write_json(out/'comparison.json',comparison)
    print(json.dumps(comparison,indent=2)); print('Output:',out)


if __name__=='__main__':main()
