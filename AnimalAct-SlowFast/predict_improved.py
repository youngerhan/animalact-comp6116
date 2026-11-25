"""Run the actual improved mixed-model ensemble on one video interval."""
import argparse
import io
import json
from pathlib import Path

import decord
import numpy as np
import torch

from clips import read_json,write_json
from runtime import PROJECT,DEFAULT_MODEL_CONFIG


def local_run(value):
    value=Path(str(value).replace('\\','/'))
    candidate=PROJECT/'runs'/value.name
    return candidate if candidate.is_dir() else value


def components(run,coefficient):
    if not np.any(coefficient):return []
    run=local_run(run)
    if run.name=='pilot32-blend':
        alpha=read_json(PROJECT/'results/pilot32/summary.json')['selected_blend']['new_weight']
        return components(PROJECT/'runs/ensemble7',(1-alpha)*coefficient)+components(PROJECT/'runs/pilot32-whole224',alpha*coefficient)
    if (run/'members.json').exists():
        member=read_json(run/'members.json')
        if 'swin' in member:
            new=np.asarray(member['selected']['new_weights'])
            swin_run=local_run(member['swin'])
            swin_eval=swin_run.with_name(swin_run.name+'-evaluation')
            return components(member['base'],coefficient*(1-new))+components(swin_eval if swin_eval.exists() else swin_run,coefficient*new)
        weights=member.get('selected',{}).get('weights') or member.get('weights')
        if weights is None:weights=[1/len(member['members'])]*len(member['members'])
        result=[]
        for path,weight in zip(member['members'],weights):result+=components(path,coefficient*weight)
        return result
    flip=False
    if (run/'selection.json').exists():
        choice=read_json(run/'selection.json'); run=local_run(choice['source_run'])
        flip=choice.get('flip',choice.get('selected',{}).get('variant')=='flip')
    settings=read_json(run/'settings.json')
    if settings.get('architecture')=='torchvision-swin3d-t':
        variant=dict(tta=flip)
        kind='swin'
    else:
        from evaluate_run import VARIANTS
        if (run/'test_summary.json').exists():
            selected=read_json(run/'test_summary.json')['selected']
            variant=next(v for v in VARIANTS if v['name']==selected)
        else:
            variant=dict(name='flip' if flip else 'plain',zoom=settings.get('eval_zoom',1.),tta=flip,
                         crops=(.5,),windows=((0.,1.),))
        kind='slowfast'
    return [dict(run=run,weights=coefficient,kind=kind,variant=variant)]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('video',type=Path)
    parser.add_argument('--run',type=Path,default=PROJECT/'runs/improved-swin-final-20261125')
    parser.add_argument('--start-frame',type=int,default=0)
    parser.add_argument('--end-frame',type=int)
    parser.add_argument('--threshold-plan',type=Path,
                        default=PROJECT/'results/improvement-20261125/calibration-final/selection-frozen.json')
    parser.add_argument('--dry-run',action='store_true')
    parser.add_argument('--output',type=Path,help='Save full class scores and decisions to JSON.')
    parser.add_argument('--match-evaluation-batch',action='store_true',
                        help='Repeat this clip to reproduce the recorded AMP evaluation batch shape.')
    args=parser.parse_args(); classes=read_json(PROJECT/'data/main.json')['classes']
    specs=components(args.run,np.ones(len(classes)))
    if not np.allclose(sum(s['weights'] for s in specs),1):raise ValueError('Ensemble coefficients do not sum to one')
    if args.dry_run:
        print(json.dumps([dict(run=str(s['run']),kind=s['kind'],weights=s['weights'].tolist()) for s in specs],indent=2)); return
    if not torch.cuda.is_available():raise RuntimeError('A CUDA GPU is required')
    torch.set_num_threads(2)
    video=args.video.resolve(); reader=decord.VideoReader(io.BytesIO(video.read_bytes()),num_threads=1)
    end=len(reader) if args.end_frame is None else args.end_frame
    if not 0<=args.start_frame<end<=len(reader):parser.error('Invalid frame interval')
    row=dict(video_path=str(video),start_frame=args.start_frame,end_frame=end,labels=[])
    probabilities=np.zeros(len(classes))
    component_scores=[]
    for index,spec in enumerate(specs,1):
        print(f'Evaluating model {index}/{len(specs)}: {spec["run"].name}',flush=True)
        settings_data=read_json(spec['run']/'settings.json')
        overrides=spec['run']/'inference_overrides.json'
        if overrides.exists():settings_data.update(read_json(overrides)['settings'])
        settings=argparse.Namespace(**settings_data)
        batch=settings.batch_size if args.match_evaluation_batch else 1
        rows=[row]*batch
        state=torch.load(spec['run']/'best.pth',map_location='cpu',weights_only=True)
        if state['classes']!=classes:raise ValueError('Checkpoint class mapping mismatch')
        if spec['kind']=='swin':
            from torchvision.models.video import swin3d_t
            from swin_experiment import dataset,predict
            from torch.utils.data import DataLoader
            model=swin3d_t(weights=None)
            model.head=torch.nn.Sequential(torch.nn.Dropout(.3),torch.nn.Linear(768,len(classes)))
            model.cuda().load_state_dict(state['state_dict'])
            _,score=predict(model,DataLoader(dataset(settings,rows),batch_size=batch,num_workers=0),spec['variant']['tta'])
        else:
            from mmaction.apis import init_recognizer
            from mmengine.config import Config
            from evaluate_run import scores
            path=Path(settings.config); config=Config.fromfile(str(path if path.exists() else DEFAULT_MODEL_CONFIG))
            config.model.cls_head.num_classes=len(classes)
            model=init_recognizer(config,checkpoint=None,device='cuda:0'); model.load_state_dict(state['state_dict'])
            settings.batch_size=batch; settings.workers=0
            mean=torch.tensor([123.675,116.28,103.53],device='cuda').view(1,3,1,1,1)
            std=torch.tensor([58.395,57.12,57.375],device='cuda').view(1,3,1,1,1)
            _,score=scores(model,rows,settings,mean,std,spec['variant'])
        probabilities+=spec['weights']*score[0]
        component_scores.append(dict(run=spec['run'].name,kind=spec['kind'],variant=spec['variant'],
                                     weights=spec['weights'].tolist(),scores=score[0].tolist()))
        del model,state; torch.cuda.empty_cache()
    if args.threshold_plan.exists():
        thresholds=np.asarray(read_json(args.threshold_plan)['thresholds']); policy='regularized class thresholds'
    else:
        thresholds=np.full(len(classes),read_json(args.run/'test_metrics.json')['threshold']); policy='global validation threshold'
    if thresholds.shape!=probabilities.shape:raise ValueError('Threshold class count mismatch')
    ranked=sorted([dict(**label,score=float(probabilities[i]),threshold=float(thresholds[i]))
                   for i,label in enumerate(classes)],key=lambda x:-x['score'])
    result=dict(video=str(video),start_frame=args.start_frame,end_frame_exclusive=end,
                         models=len(specs),threshold_policy=policy,matched_evaluation_batch=args.match_evaluation_batch,
                         predicted_actions=[r for r in ranked if r['score']>=r['threshold']],top_scores=ranked[:5])
    if args.output:write_json(args.output,dict(**result,all_scores=ranked,component_scores=component_scores))
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
