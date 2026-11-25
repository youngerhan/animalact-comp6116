"""Verify legacy checkpoint preprocessing against every archived prediction."""
import argparse
from pathlib import Path
import numpy as np
import torch
from clips import read_json,write_json
from evaluate_run import scores,VARIANTS
from train_strong import build_model
from runtime import PROJECT


def main():
    run=PROJECT/'runs/ft256'
    settings=read_json(run/'settings.json'); settings.update(read_json(run/'inference_overrides.json')['settings'])
    settings=argparse.Namespace(**settings);settings.config=Path(settings.config)
    torch.set_num_threads(2)
    model,_,_=build_model(settings,31,settings.weights)
    model.load_state_dict(torch.load(run/'best.pth',map_location='cpu',weights_only=True)['state_dict'])
    mean=torch.tensor([123.675,116.28,103.53],device='cuda').view(1,3,1,1,1)
    std=torch.tensor([58.395,57.12,57.375],device='cuda').view(1,3,1,1,1)
    variant=next(v for v in VARIANTS if v['name']==read_json(run/'test_summary.json')['selected'])
    result={}
    for split in ('valid','test'):
        rows=read_json(PROJECT/'data/main.json')['splits'][split]
        y,a=scores(model,rows,settings,mean,std,variant)
        with np.load(run/f'{split}_predictions.npz') as old:
            assert np.array_equal(y,old['targets'])
            error=float(np.max(np.abs(a-old['scores'])))
            result[split]=dict(clips=len(rows),max_absolute_error=error,passed=error<1e-6)
        print(split,result[split],flush=True)
    write_json(PROJECT/'results/improvement-20261125/legacy-inference-verification.json',result)
    if not all(v['passed'] for v in result.values()):raise RuntimeError('Legacy prediction verification failed')


if __name__=='__main__':main()
