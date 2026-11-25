"""Validation-only choice of Swin TTA/fusion, then a fixed test evaluation."""
import argparse
import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from clips import read_json,write_json
from metrics import evaluate,per_class_thresholds,select_threshold
from runtime import PROJECT,run_report
from swin_experiment import build,dataset,predict

OUT=PROJECT/'results/improvement-20261125'
RUN=PROJECT/'runs/improve-swin-t'


def infer(split,flip):
    settings=argparse.Namespace(**read_json(RUN/'settings.json'))
    manifest=read_json(Path(settings.manifest))
    checkpoint=torch.load(RUN/'best.pth',map_location='cpu',weights_only=True)
    model=build(Path(settings.weights),len(settings.classes)); model.load_state_dict(checkpoint['state_dict'])
    loader=DataLoader(dataset(settings,manifest['splits'][split]),batch_size=settings.batch_size,
                      num_workers=settings.workers,pin_memory=True)
    pairs=predict(model,loader,flip)
    del model,checkpoint; torch.cuda.empty_cache()
    return pairs


def main():
    torch.set_num_threads(2)
    manifest=read_json(PROJECT/'data/main.json'); classes=manifest['classes']
    base=PROJECT/'runs/improved-final-20261125'
    if not base.exists():base=PROJECT/'runs/pilot32-blend'
    base_report=read_json(base/'valid_metrics.json')
    with np.load(base/'valid_predictions.npz') as saved:targets,base_scores=saved['targets'],saved['scores']
    with np.load(RUN/'valid_predictions.npz') as saved:plain=saved['targets'],saved['scores']
    print('Swin validation flip averaging',flush=True); flipped=infer('valid',True)
    if not np.array_equal(plain[0],targets) or not np.array_equal(flipped[0],targets):raise ValueError('Validation order mismatch')
    table=[]
    for flip,pairs in [(False,plain),(True,flipped)]:
        for alpha in (0,.125,.25,.5,.75,1):
            scores=(1-alpha)*base_scores+alpha*pairs[1]
            threshold=select_threshold(targets,scores); report=evaluate(targets,scores,classes,threshold)
            table.append(dict(flip=flip,new_weight=alpha,threshold=threshold,mAP=report['mAP'],
                              macro_F1=report['macro_F1'],utility=(report['mAP']+report['macro_F1'])/2))
    eligible=[r for r in table if r['mAP']>=base_report['mAP']-1e-12 and r['macro_F1']>=base_report['macro_F1']-1e-12]
    selected=max(eligible,key=lambda r:r['utility'])
    plan=dict(frozen_before_test=True,time=datetime.datetime.now().isoformat(),base=str(base),swin=str(RUN),
              candidates=table,selected=selected,rule='Nondecreasing validation mAP and Macro-F1, maximum mean; predefined weights and TTA candidates.')
    write_json(OUT/'swin-selection-frozen.json',plan); print('Swin frozen fusion',selected,flush=True)
    alpha=selected['new_weight']; chosen=flipped if selected['flip'] else plain
    valid_scores=(1-alpha)*base_scores+alpha*chosen[1]
    with np.load(base/'test_predictions.npz') as saved:test_targets,base_test=saved['targets'],saved['scores']
    if alpha:
        print('Swin selected test evaluation',flush=True); tested=infer('test',selected['flip'])
        if not np.array_equal(tested[0],test_targets):raise ValueError('Test order mismatch')
        test_scores=(1-alpha)*base_test+alpha*tested[1]
        raw_threshold=select_threshold(*chosen)
        run_report(RUN,'test',tested,classes,raw_threshold,per_class_thresholds(*chosen),manifest['splits']['test'])
    else:test_scores=base_test
    final=PROJECT/'runs/improved-swin-final-20261125'; final.mkdir(exist_ok=False)
    thresholds=per_class_thresholds(targets,valid_scores)
    run_report(final,'valid',(targets,valid_scores),classes,selected['threshold'],thresholds,manifest['splits']['valid'])
    report=run_report(final,'test',(test_targets,test_scores),classes,selected['threshold'],thresholds,manifest['splits']['test'])
    write_json(final/'members.json',plan)
    old=read_json(PROJECT/'runs/pilot32-blend/test_metrics.json')
    summary=dict(out=str(final),selected=selected,test={k:report[k] for k in ('mAP','macro_F1','threshold')},
                 old_test={k:old[k] for k in ('mAP','macro_F1','threshold')},
                 delta_pp={k:100*(report[k]-old[k]) for k in ('mAP','macro_F1')},
                 exploratory_evaluation_note='Swin comparison was specified before the three-variant test result. Two preplanned stages evaluated test; no claim of an untouched single-use test set.')
    rng=np.random.default_rng(42); groups={}
    for i,row in enumerate(manifest['splits']['test']):groups.setdefault(row['video_id'],[]).append(i)
    groups=list(groups.values()); deltas=[]
    with np.load(PROJECT/'runs/pilot32-blend/test_predictions.npz') as saved:old_scores=saved['scores']
    for _ in range(300):
        ids=np.concatenate([groups[i] for i in rng.integers(0,len(groups),len(groups))])
        a=evaluate(test_targets[ids],old_scores[ids],classes,old['threshold'])
        b=evaluate(test_targets[ids],test_scores[ids],classes,selected['threshold'])
        deltas.append([100*(b[k]-a[k]) for k in ('mAP','macro_F1')])
    summary['paired_video_bootstrap_95_interval_pp']=dict(zip(('mAP','macro_F1'),np.percentile(deltas,[2.5,97.5],axis=0).T.tolist()))
    write_json(OUT/'summary-swin.json',summary); print(summary,flush=True)


if __name__=='__main__':main()
