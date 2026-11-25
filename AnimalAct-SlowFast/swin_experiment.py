"""Complementary Video Swin-T: frozen-feature head fitting, then partial fine-tuning.

Official model/weights: torchvision 0.22, KINETICS400_V1.
https://docs.pytorch.org/vision/0.22/models/generated/torchvision.models.video.swin3d_t.html
"""
import argparse
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision.models.video import swin3d_t

from clips import DEFAULT_DATASET, read_json, write_json
from metrics import evaluate, per_class_thresholds, select_threshold
from provenance import record_run
from runtime import PROJECT, run_report, save_checkpoint
from train_strong import ClipDataset, sample_weights


def build(weights, classes):
    model=swin3d_t(weights=None)
    model.load_state_dict(torch.load(weights,map_location='cpu',weights_only=True))
    model.head=nn.Sequential(nn.Dropout(.3),nn.Linear(model.head.in_features,classes))
    nn.init.normal_(model.head[1].weight,std=.01); nn.init.zeros_(model.head[1].bias)
    return model.cuda()


def features(model, x):
    x=model.pos_drop(model.patch_embed(x)); x=model.norm(model.features(x))
    return model.avgpool(x.permute(0,4,1,2,3)).flatten(1)


def normalize(video):
    value=video.cuda(non_blocking=True).float()/255
    mean=torch.tensor([.485,.456,.406],device='cuda').view(1,3,1,1,1)
    std=torch.tensor([.229,.224,.225],device='cuda').view(1,3,1,1,1)
    return (value-mean)/std


def dataset(options,rows,train=False):
    return ClipDataset(options.dataset,rows,len(options.classes),16,options.image_size,train,crop='whole')


@torch.inference_mode()
def cached_features(model,loader,label):
    model.eval(); xs=[]; ys=[]
    for step,(video,target) in enumerate(loader,1):
        with torch.autocast('cuda',dtype=torch.float16):
            xs.append(features(model,normalize(video)).float().cpu().numpy())
        ys.append(target.numpy().astype(bool))
        if step%100==0 or step==len(loader):print(f'{label} features {step}/{len(loader)}',flush=True)
    return np.concatenate(xs),np.concatenate(ys)


@torch.inference_mode()
def predict(model,loader,flip=False):
    model.eval(); ys=[]; scores=[]
    for video,target in loader:
        with torch.autocast('cuda',dtype=torch.float16):
            x=normalize(video); current=model(x).float().sigmoid()
            if flip:current=(current+model(x.flip(-1)).float().sigmoid())/2
        scores.append(current.cpu().numpy()); ys.append(target.numpy().astype(bool))
    return np.concatenate(ys),np.concatenate(scores)


@torch.inference_mode()
def head_scores(model,x):
    model.head.eval()
    with torch.autocast('cuda',dtype=torch.float16):
        return model.head(x).float().sigmoid().cpu().numpy()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',type=Path,default=DEFAULT_DATASET)
    parser.add_argument('--manifest',type=Path,default=PROJECT/'data/main.json')
    parser.add_argument('--weights',type=Path,default=PROJECT.parent/'swin3d_t-7615ae03.pth')
    parser.add_argument('--out',type=Path,default=PROJECT/'runs/improve-swin-t')
    parser.add_argument('--image-size',type=int,default=224)
    parser.add_argument('--batch-size',type=int,default=2)
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--head-epochs',type=int,default=40)
    parser.add_argument('--epochs',type=int,default=3)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--extra-epochs',type=int,default=0)
    options=parser.parse_args()
    if options.resume and not (options.out/'last.pth').is_file():
        parser.error('--resume requires last.pth optimizer state. Release packages contain best weights only; start a new output directory for training.')
    torch.set_num_threads(2); torch.manual_seed(options.seed)
    random.seed(options.seed); np.random.seed(options.seed)
    torch.backends.cudnn.benchmark=False
    manifest=read_json(options.manifest); options.classes=manifest['classes']
    rows=manifest['splits']['train']; valid_rows=manifest['splits']['valid']
    if options.smoke:rows=rows[:8]; valid_rows=valid_rows[:8]
    options.out.mkdir(parents=True,exist_ok=options.resume)
    settings={k:str(v.resolve()) if isinstance(v,Path) else v for k,v in vars(options).items() if k not in {'resume','extra_epochs'}}
    settings.update(architecture='torchvision-swin3d-t',frames=16,crop='whole')
    if options.resume:
        if read_json(options.out/'settings.json')!=settings:raise ValueError('Resume settings differ')
    else:write_json(options.out/'settings.json',settings)
    record_run(options.out,options)
    model=build(options.weights,len(options.classes))
    valid_loader=DataLoader(dataset(options,valid_rows),batch_size=options.batch_size,
                            num_workers=options.workers,pin_memory=True)
    counts=np.zeros(len(options.classes))
    for row in rows: counts[row["labels"]]+=1
    # A smoke subset may omit a class; finite weights avoid dividing by zero.
    weights=sample_weights(rows,np.maximum(counts,1))
    if not (options.out/'head-best.pth').exists():
        cache_path=options.out/'features.npz'
        if cache_path.exists():
            with np.load(cache_path) as saved:
                train_x,train_y,valid_x,valid_y=[saved[k] for k in ('train_x','train_y','valid_x','valid_y')]
        else:
            train_loader=DataLoader(dataset(options,rows),batch_size=max(options.batch_size,4),
                                    num_workers=options.workers,pin_memory=True)
            train_x,train_y=cached_features(model,train_loader,'train')
            valid_x,valid_y=cached_features(model,valid_loader,'valid')
            np.savez_compressed(cache_path,train_x=train_x,train_y=train_y,valid_x=valid_x,valid_y=valid_y)
        x=torch.tensor(train_x,device='cuda'); y=torch.tensor(train_y,device='cuda',dtype=torch.float32)
        vx=torch.tensor(valid_x,device='cuda')
        optimizer=torch.optim.AdamW(model.head.parameters(),lr=.003,weight_decay=.05)
        best=-1.; last_improved=0; head_history=[]
        for epoch in range(1,options.head_epochs+1):
            model.head.train(); generator=torch.Generator().manual_seed(options.seed+epoch)
            indices=torch.multinomial(torch.as_tensor(weights),len(rows),replacement=True,generator=generator).cuda()
            for subset in indices.split(256):
                optimizer.zero_grad(set_to_none=True)
                loss=nn.functional.binary_cross_entropy_with_logits(model.head(x[subset]),y[subset])
                loss.backward(); optimizer.step()
            scores=head_scores(model,vx); threshold=select_threshold(valid_y,scores)
            report=evaluate(valid_y,scores,options.classes,threshold)
            row=dict(epoch=epoch,mAP=report['mAP'],macro_F1=report['macro_F1'])
            head_history.append(row); print('Swin head',row,flush=True)
            if report['mAP']>best:
                best=report['mAP']; last_improved=epoch
                save_checkpoint(options.out/'head-best.pth',dict(head=model.head.state_dict(),epoch=epoch))
            if epoch-last_improved>=6:break
        write_json(options.out/'head-history.json',head_history)
        del x,y,vx,optimizer
    head=torch.load(options.out/'head-best.pth',map_location='cpu',weights_only=True)
    model.head.load_state_dict(head['head'])
    pairs=predict(model,valid_loader); threshold=select_threshold(*pairs)
    thresholds=per_class_thresholds(*pairs)
    initial=run_report(options.out,'initial_valid',pairs,options.classes,threshold,thresholds,valid_rows)
    best_map=initial['mAP']; history=[dict(epoch=0,valid_mAP=initial['mAP'],valid_macro_F1=initial['macro_F1'],train_loss=None)]
    if not options.resume:
        save_checkpoint(options.out/'best.pth',dict(state_dict=model.state_dict(),classes=options.classes,
                        architecture='torchvision-swin3d-t',epoch=0,threshold=threshold,thresholds=thresholds))
    model.requires_grad_(False)
    model.features[4:].requires_grad_(True); model.norm.requires_grad_(True); model.head.requires_grad_(True)
    backbone=[p for name,p in model.named_parameters() if p.requires_grad and not name.startswith('head.')]
    optimizer=torch.optim.AdamW([dict(params=model.head.parameters(),lr=.0001),dict(params=backbone,lr=.00001)],weight_decay=.01)
    scaler=torch.amp.GradScaler('cuda'); first=1; last_improved=0
    if options.resume and (options.out/'last.pth').exists():
        saved=torch.load(options.out/'last.pth',map_location='cpu',weights_only=True)
        model.load_state_dict(saved['state_dict']); optimizer.load_state_dict(saved['optimizer']); scaler.load_state_dict(saved['scaler'])
        history=saved['history']; best_map=saved['best_map']; first=saved['epoch']+1; last_improved=saved['last_improved']
    if options.extra_epochs:
        if not options.resume or options.extra_epochs < 0:
            parser.error("--extra-epochs requires --resume and a positive count")
        last_improved = first - 1
        write_json(options.out/'continuation.json',dict(original_epochs=options.epochs,
                   extra_epochs=options.extra_epochs,head_lr=.00005,backbone_lr=.000005,
                   reason="Predeclared validation rule: last initial epoch best and >=2pp gain over frozen features."))
    for epoch in range(first,options.epochs+options.extra_epochs+1):
        torch.manual_seed(options.seed+epoch); random.seed(options.seed+epoch); np.random.seed(options.seed+epoch)
        generator=torch.Generator().manual_seed(options.seed+epoch)
        sampler=WeightedRandomSampler(weights,len(rows),replacement=True,generator=generator)
        loader=DataLoader(dataset(options,rows,True),batch_size=options.batch_size,sampler=sampler,
                          num_workers=options.workers,pin_memory=True)
        model.eval(); model.head.train(); optimizer.zero_grad(set_to_none=True)
        examples=0; total_loss=0.; started=time.monotonic()
        extended = epoch > options.epochs
        phase_epoch = epoch-options.epochs if extended else epoch
        phase_epochs = options.extra_epochs if extended else options.epochs
        ratio=.05+.95*.5*(1+math.cos(math.pi*(phase_epoch-1)/max(phase_epochs,1)))
        optimizer.param_groups[0]['lr']=(.00005 if extended else .0001)*ratio
        optimizer.param_groups[1]['lr']=(.000005 if extended else .00001)*ratio
        accumulate=max(1,8//options.batch_size)
        for step,(video,target) in enumerate(loader,1):
            group=min(accumulate,len(loader)-((step-1)//accumulate)*accumulate)
            with torch.autocast('cuda',dtype=torch.float16):
                logits=model(normalize(video)); loss=nn.functional.binary_cross_entropy_with_logits(logits.float(),target.cuda())
            if not torch.isfinite(loss):raise FloatingPointError('Non-finite Swin loss')
            scaler.scale(loss/group).backward()
            if step%accumulate==0 or step==len(loader):
                scaler.unscale_(optimizer); nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],5)
                scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True)
            total_loss+=float(loss.detach())*len(target); examples+=len(target)
            if step==1 or step%200==0 or step==len(loader):
                print(f'Swin epoch {epoch} batch {step}/{len(loader)} loss={total_loss/examples:.4f} {examples/(time.monotonic()-started):.1f} clips/s',flush=True)
        pairs=predict(model,valid_loader); threshold=select_threshold(*pairs); thresholds=per_class_thresholds(*pairs)
        report=evaluate(*pairs,options.classes,threshold)
        row=dict(epoch=epoch,train_loss=total_loss/examples,valid_mAP=report['mAP'],valid_macro_F1=report['macro_F1'])
        history.append(row); print('Swin validation',row,flush=True)
        if report['mAP']>best_map:
            best_map=report['mAP']; last_improved=epoch
            save_checkpoint(options.out/'best.pth',dict(state_dict=model.state_dict(),classes=options.classes,
                architecture='torchvision-swin3d-t',epoch=epoch,threshold=threshold,thresholds=thresholds))
        save_checkpoint(options.out/'last.pth',dict(state_dict=model.state_dict(),optimizer=optimizer.state_dict(),scaler=scaler.state_dict(),
                        epoch=epoch,history=history,best_map=best_map,last_improved=last_improved))
        write_json(options.out/'history.json',history)
        if epoch-last_improved>=2:break
    best=torch.load(options.out/'best.pth',map_location='cpu',weights_only=True); model.load_state_dict(best['state_dict'])
    pairs=predict(model,valid_loader)
    report=run_report(options.out,'valid',pairs,options.classes,best['threshold'],best['thresholds'],valid_rows)
    print('Swin selected validation',report['mAP'],report['macro_F1'],'epoch',best['epoch'],flush=True)


if __name__=='__main__':main()
