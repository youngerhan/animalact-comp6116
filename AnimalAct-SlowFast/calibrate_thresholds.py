"""Video-group cross-validated shrinkage of class thresholds; no model-score changes."""
import argparse
import copy
import datetime
import hashlib
import json
import sys
from pathlib import Path

import numpy as np


def macro(targets, predictions):
    tp=(targets & predictions).sum(0); fp=(~targets & predictions).sum(0); fn=(targets & ~predictions).sum(0)
    denominator=2*tp+fp+fn
    return float(np.divide(2*tp,denominator,out=np.zeros(len(tp)),where=denominator>0).mean())


def fit(targets, scores, regularizer, select_threshold, per_class_thresholds):
    global_threshold=select_threshold(targets,scores)
    if regularizer is None:
        return np.full(scores.shape[1],global_threshold)
    raw=np.asarray(per_class_thresholds(targets,scores))
    counts=targets.sum(0)
    weight=np.divide(counts,counts+regularizer,out=np.zeros(len(counts)),where=(counts+regularizer)>0)
    logit=lambda x:np.log(x/(1-x))
    mixed=weight*logit(raw)+(1-weight)*logit(global_threshold)
    return 1/(1+np.exp(-mixed))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project',type=Path,required=True)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args(); sys.path.insert(0,str(args.project))
    from clips import read_json,write_json
    from metrics import evaluate,select_threshold,per_class_thresholds,confusion_counts,f1_scores
    manifest=read_json(args.project/'data/main.json'); rows=manifest['splits']['valid']
    with np.load(args.run/'valid_predictions.npz') as stored:
        targets,scores=stored['targets'],stored['scores']
    folds=np.array([int(hashlib.sha256(('42:'+r['video_id']).encode()).hexdigest()[:8],16)%5 for r in rows])
    table=[]
    for k in (None,0,10,30,100,300):
        prediction=np.zeros_like(targets)
        for fold in range(5):
            train=folds!=fold; held=~train
            thresholds=fit(targets[train],scores[train],k,select_threshold,per_class_thresholds)
            prediction[held]=scores[held]>=thresholds
        table.append(dict(regularizer=k,OOF_macro_F1=macro(targets,prediction)))
    selected=max(table,key=lambda x:x['OOF_macro_F1'])
    thresholds=fit(targets,scores,selected['regularizer'],select_threshold,per_class_thresholds)
    args.out.mkdir(exist_ok=False,parents=True)
    plan=dict(frozen_before_test=True,time=datetime.datetime.now().isoformat(),run=str(args.run),
              policy='Five validation folds grouped by video; shrink class thresholds toward the global threshold in log-odds space. Select regularizer by aggregate held-out F1. No score or AP changes.',
              candidates=table,selected=selected,thresholds=thresholds.tolist(),
              global_reference_threshold=select_threshold(targets,scores))
    write_json(args.out/'selection-frozen.json',plan)
    # Only now open test predictions. Uniform-threshold results stay the main comparison.
    with np.load(args.run/'test_predictions.npz') as stored:
        test_targets,test_scores=stored['targets'],stored['scores']
    uniform=evaluate(test_targets,test_scores,manifest['classes'],plan['global_reference_threshold'])
    calibrated=copy.deepcopy(uniform)
    tp,fp,fn,tn=confusion_counts(test_targets,test_scores,thresholds)
    f1=f1_scores(tp,fp,fn)
    calibrated.pop('threshold')
    calibrated.update(threshold_policy='class_specific_regularized',thresholds=thresholds.tolist(),macro_F1=float(f1.mean()))
    for i,row in enumerate(calibrated['per_class']):
        row.update(F1=float(f1[i]),TP=int(tp[i]),FP=int(fp[i]),FN=int(fn[i]),TN=int(tn[i]),threshold=float(thresholds[i]))
    write_json(args.out/'test-uniform-metrics.json',uniform)
    write_json(args.out/'test-calibrated-metrics.json',calibrated)
    summary=dict(OOF_candidates=table,selected=selected,
                 uniform_test_macro_F1=uniform['macro_F1'],calibrated_test_macro_F1=calibrated['macro_F1'],
                 mAP_unchanged=uniform['mAP'],F1_delta_pp=100*(calibrated['macro_F1']-uniform['macro_F1']),
                 TP=int(tp.sum()),FP=int(fp.sum()),FN=int(fn.sum()))
    write_json(args.out/'summary.json',summary)
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
