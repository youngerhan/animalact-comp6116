"""Frozen-weight validation probe: real temporal clips versus one repeated middle frame."""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    project=args.project.resolve(); sys.path.insert(0,str(project))
    from clips import read_json,write_json
    from train_strong import ClipDataset
    from runtime import video_logits
    from metrics import evaluate,select_threshold
    from mmaction.apis import init_recognizer
    from mmengine.config import Config
    run=project/'runs/pilot32-whole224'
    settings=read_json(run/'settings.json'); manifest=read_json(Path(settings['manifest']))
    best=torch.load(run/'best.pth',map_location='cpu',weights_only=True)
    config=Config.fromfile(settings['config']); config.model.cls_head.num_classes=len(best['classes'])
    model=init_recognizer(config,checkpoint=None,device='cuda:0')
    model.load_state_dict(best['state_dict']); model.eval()
    torch.set_num_threads(2); torch.manual_seed(42); torch.backends.cudnn.benchmark=False
    rows=manifest['splits']['valid']; classes=manifest['classes']
    dataset=ClipDataset(Path(settings['dataset']),rows,len(classes),32,224,False,crop='whole',eval_zoom=settings['eval_zoom'])
    loader=DataLoader(dataset,batch_size=8,num_workers=2,pin_memory=True)
    mean=torch.tensor([123.675,116.28,103.53],device='cuda').view(1,3,1,1,1)
    std=torch.tensor([58.395,57.12,57.375],device='cuda').view(1,3,1,1,1)
    true,normal,static=[],[],[]
    with torch.inference_mode():
        for step,(video,target) in enumerate(loader,1):
            normal.append(video_logits(model,video,mean,std).sigmoid().cpu().numpy())
            frozen=video[:,:,16:17,:,:].expand(-1,-1,32,-1,-1)
            static.append(video_logits(model,frozen,mean,std).sigmoid().cpu().numpy())
            true.append(target.numpy().astype(bool))
            if step%20==0: print(f'validation probe {step}/{len(loader)} batches',flush=True)
    targets,normal,static=map(np.concatenate,(true,normal,static))
    args.out.mkdir(exist_ok=True,parents=True)
    np.savez_compressed(args.out/'motion-probe.npz',targets=targets,normal=normal,static=static)
    threshold=best['threshold']
    normal_report=evaluate(targets,normal,classes,threshold)
    static_report=evaluate(targets,static,classes,threshold)
    with np.load(run/'valid_predictions.npz') as saved:
        difference=float(np.max(np.abs(normal-saved['scores'])))
    report=dict(run=str(run),split='valid',clips=len(rows),threshold=threshold,
                normal=normal_report,static_middle_frame=static_report,
                normal_scores_max_abs_difference_from_saved=difference,
                static_retuned_valid_threshold=select_threshold(targets,static),
                warning='Diagnostic input intervention; no training or test-set selection. Single model, not the final ensemble.')
    rng=np.random.default_rng(42); groups={}
    for i,row in enumerate(rows): groups.setdefault(row['video_id'],[]).append(i)
    videos=list(groups); deltas=[]
    for _ in range(300):
        ids=np.concatenate([groups[videos[i]] for i in rng.integers(0,len(videos),len(videos))])
        a=evaluate(targets[ids],normal[ids],classes,threshold)['mAP']
        b=evaluate(targets[ids],static[ids],classes,threshold)['mAP']
        deltas.append(100*(a-b))
    report['normal_minus_static_mAP_pp']=100*(normal_report['mAP']-static_report['mAP'])
    report['paired_video_bootstrap_delta_95_interval_pp']=np.percentile(deltas,[2.5,97.5]).tolist()
    write_json(args.out/'motion-probe.json',report)
    print({key:value for key,value in report.items() if key not in ('normal','static_middle_frame')},flush=True)
    print('normal',normal_report['mAP'],normal_report['macro_F1'],'static',static_report['mAP'],static_report['macro_F1'],flush=True)


if __name__=='__main__':
    main()
