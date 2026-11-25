"""Grouped validation fitting of shrinkage class weights; never reads test targets."""
import hashlib
import numpy as np
from metrics import average_precision, evaluate, select_threshold, confusion_counts, f1_scores

ALPHAS=(0,.125,.25,.5,.75,1)


def global_fit(targets,base,new,classes):
    candidates=[]
    reference=evaluate(targets,base,classes,select_threshold(targets,base))
    for alpha in ALPHAS:
        scores=(1-alpha)*base+alpha*new
        threshold=select_threshold(targets,scores)
        report=evaluate(targets,scores,classes,threshold)
        if report['mAP']>=reference['mAP'] and report['macro_F1']>=reference['macro_F1']:
            candidates.append((.5*(report['mAP']+report['macro_F1']),alpha,threshold))
    _,alpha,threshold=max(candidates,key=lambda row:row[0])
    return alpha,threshold


def fit(targets,base,new,classes,regularizer):
    alpha,_=global_fit(targets,base,new,classes)
    weights=np.full(base.shape[1],alpha,dtype=float)
    if regularizer is not None:
        counts=targets.sum(0)
        for column,n in enumerate(counts):
            if not n:continue
            values=[(average_precision(targets[:,column],(1-a)*base[:,column]+a*new[:,column]),a) for a in ALPHAS]
            # Prefer the global coefficient when class AP ties.
            _,local=max(values,key=lambda pair:(pair[0],-abs(pair[1]-alpha)))
            shrink=n/(n+regularizer) if n+regularizer else 0
            weights[column]=(1-shrink)*alpha+shrink*local
    scores=(1-weights)*base+weights*new
    return weights,select_threshold(targets,scores)


def cross_validate(targets,base,new,classes,video_ids):
    folds=np.array([int(hashlib.sha256(('42:'+v).encode()).hexdigest()[:8],16)%5 for v in video_ids])
    table=[]
    for regularizer in (None,10,30,100):
        all_scores=np.zeros_like(base); decisions=np.zeros_like(targets)
        for fold in range(5):
            train=folds!=fold; held=~train
            weights,threshold=fit(targets[train],base[train],new[train],classes,regularizer)
            scores=(1-weights)*base[held]+weights*new[held]
            all_scores[held]=scores; decisions[held]=scores>=threshold
        report=evaluate(targets,all_scores,classes,.5)
        tp=(targets & decisions).sum(0); fp=(~targets & decisions).sum(0); fn=(targets & ~decisions).sum(0)
        f1=float(f1_scores(tp,fp,fn).mean())
        table.append(dict(regularizer=regularizer,OOF_mAP=report['mAP'],OOF_macro_F1=f1,
                          utility=(report['mAP']+f1)/2))
    eligible=[r for r in table if r['OOF_mAP']>=table[0]['OOF_mAP'] and r['OOF_macro_F1']>=table[0]['OOF_macro_F1']]
    selected=max(eligible,key=lambda r:r['utility'])
    return selected,table
