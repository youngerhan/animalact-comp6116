"""Read-only error audit; generate deterministic validation review cases and score summaries."""
import argparse
import csv
import hashlib
import html
import io
import json
import sys
from collections import Counter
from pathlib import Path

import cv2
import decord
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def overlap(actions, name, start, end):
    intervals = sorted((max(start, a['start_frame']), min(end, a['end_frame']))
                       for a in actions if a['action'] == name)
    total, last = 0, start
    for left, right in intervals:
        if right > left:
            total += max(0, right-max(last, left))
            last = max(last, right)
    return total / (end-start)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    project, out = args.project.resolve(), args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    dataset = project.parent / 'AnimalAct'
    sys.path.insert(0,str(project))
    from clips import annotation_intervals
    manifest = read(project/'data/main.json')
    classes = manifest['classes']
    names = [c['name'] for c in classes]
    threshold = read(project/'runs/pilot32-blend/valid_metrics.json')['threshold']
    annotations = {}
    summary = dict(threshold=threshold, review_split='valid', selection='targeted, not a random prevalence sample', splits={})
    detailed, pairs_by_split = [], {}
    for split in ('valid', 'test'):
        rows = manifest['splits'][split]
        official = {r['video_id']:r for r in read(dataset/f'splits/{split}.json')['samples']}
        with np.load(project/f'runs/pilot32-blend/{split}_predictions.npz') as stored:
            targets, scores = stored['targets'], stored['scores']
            for key, field in [('video_ids','video_id'), ('start_frames','start_frame'), ('end_frames','end_frame')]:
                if not np.array_equal(stored[key], [r[field] for r in rows]):
                    raise ValueError(f'{split}: {field} mismatch')
        pairs_by_split[split] = (targets, scores)
        pred = scores >= threshold
        missed, extra = targets & ~pred, ~targets & pred
        cross = Counter()
        temporal = Counter()
        for index, row in enumerate(rows):
            key = split, row['video_id']
            if key not in annotations:
                annotations[key] = read(dataset/f'annotations/{split}/{row["video_id"]}.json')
            sample = official[row['video_id']]
            intervals = annotation_intervals(annotations[key],sample['nb_frames'],sample['fps'],Counter())
            actions = [dict(start_frame=a,end_frame=b,action=name) for a,b,name in intervals]
            for a in np.flatnonzero(missed[index]):
                for b in np.flatnonzero(extra[index]):
                    cross[names[a], names[b]] += 1
            for column in np.flatnonzero(missed[index] | extra[index]):
                kind = 'FN' if targets[index,column] else 'FP'
                fraction = overlap(actions, names[column], row['start_frame'], row['end_frame'])
                exists = any(a['action']==names[column] for a in actions)
                if kind == 'FP':
                    temporal['any_temporal_overlap' if fraction > 0 else 'zero_temporal_overlap'] += 1
                    temporal['label_elsewhere_in_video' if exists else 'label_absent_from_video_annotations'] += 1
                detailed.append(dict(split=split, index=index, video_id=row['video_id'], kind=kind,
                    action=names[column], score=float(scores[index,column]), duration=row['duration_seconds'],
                    annotation_overlap=fraction, annotated_elsewhere=exists))
        stats = {k:int(v) for k,v in dict(TP=(targets&pred).sum(),FP=extra.sum(),FN=missed.sum()).items()}
        stats['error_clips'] = int((missed.any(1)|extra.any(1)).sum())
        stats['clips'] = len(rows)
        stats['FP_annotation_context'] = dict(temporal)
        stats['cooccurring_FN_FP'] = [dict(missed=a, extra=b, clips=n) for (a,b),n in cross.most_common(20)]
        stats['duration_groups'] = []
        for lower,upper in [(0,1),(1,3),(3,10),(10,float('inf'))]:
            mask = np.array([lower<r['duration_seconds']<=upper for r in rows])
            positives = int(targets[mask].sum())
            stats['duration_groups'].append(dict(min_exclusive=lower, max_inclusive=upper if upper!=float('inf') else None,
                clips=int(mask.sum()), recall=float((targets[mask]&pred[mask]).sum()/positives) if positives else None))
        summary['splits'][split] = stats
    with (out/'all-errors.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(detailed[0])); writer.writeheader(); writer.writerows(detailed)
    valid = [r for r in detailed if r['split']=='valid']
    selected, used = [], set()
    # One false negative and one false positive per target class where available.
    focus = ['Carrying In Mouth','Being Eaten','Shaking','Climbing','Grooming','Attacking',
             'Attending','Sensing','Keeping still','Eating','Chirping','Walking']
    for action in focus:
        for kind in ('FN','FP'):
            pool = [r for r in valid if r['action']==action and r['kind']==kind and r['video_id'] not in used]
            if pool:
                case = sorted(pool,key=lambda r:r['score'], reverse=kind=='FP')[0].copy()
                selected.append(case); used.add(case['video_id'])
    targets,scores = pairs_by_split['valid']
    for action in ('Drinking','Spitting Venom','Walking'):
        col = names.index(action)
        pool = [i for i,row in enumerate(manifest['splits']['valid']) if targets[i,col] and scores[i,col]>=threshold and row['video_id'] not in used]
        if pool:
            index=max(pool,key=lambda i:scores[i,col]); row=manifest['splits']['valid'][index]
            selected.append(dict(split='valid',index=index,video_id=row['video_id'],kind='TP control',action=action,score=float(scores[index,col])))
            used.add(row['video_id'])
    font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',16)
    small=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',14)
    cards, pages = [], []
    for number,case in enumerate(selected,1):
        row=manifest['splits']['valid'][case['index']]
        annotation=annotations['valid',row['video_id']]
        video=dataset/row['video_path']
        reader=decord.VideoReader(io.BytesIO(video.read_bytes()),num_threads=1)
        indices=np.linspace(row['start_frame'],row['end_frame']-1,8).round().astype(int)
        frames=reader.get_batch(indices.tolist()).asnumpy()
        true_names=[names[i] for i in row['labels']]
        prediction=[f'{names[i]} {scores[case["index"],i]:.2f}' for i in np.argsort(-scores[case['index']]) if scores[case['index'],i]>=threshold]
        case.update(case_id=number,start_frame=row['start_frame'],end_frame=row['end_frame'],
            duration=row['duration_seconds'],animals=annotation['list_animal'],targets=true_names,predictions=prediction,
            width=int(frames.shape[2]),height=int(frames.shape[1]),sampled_frames=indices.tolist(),
            unique_32_frame_indices=len(set(np.linspace(row['start_frame'],row['end_frame']-1,32).round().astype(int))),
            original_video=str(video), annotation=annotation)
        card=Image.new('RGB',(1280,210),'white'); draw=ImageDraw.Draw(card)
        draw.text((8,2),f'#{number} {case["kind"]}: {case["action"]} score={case["score"]:.3f} | {row["video_id"]} [{row["start_frame"]},{row["end_frame"]}) {row["duration_seconds"]:.2f}s',font=font,fill='black')
        draw.text((8,24),'GT: '+', '.join(true_names),font=small,fill='#165d26')
        draw.text((8,44),'Pred: '+', '.join(prediction)[:170],font=small,fill='#a12a20')
        for j,frame in enumerate(frames):
            card.paste(Image.fromarray(frame).resize((160,120)),(j*160,68))
            draw.text((j*160+4,190),f'frame {indices[j]}',font=small,fill='black')
        cards.append(card)
        poster=f'case-{number:02d}.jpg'; card.save(out/poster,quality=93)
        start,end=row['start_frame']/annotation['fps'],row['end_frame']/annotation['fps']
        pages.append(f'<article><h2>#{number} {html.escape(case["kind"]+": "+case["action"])}</h2><img src="{poster}"><video controls preload="none" src="{html.escape(video.as_uri())}#t={start:.3f},{end:.3f}"></video><details><summary>原始标注与完整检查记录</summary><pre>{html.escape(json.dumps(case,ensure_ascii=False,indent=2))}</pre></details></article>')
    for page,start in enumerate(range(0,len(cards),5),1):
        sheet=Image.new('RGB',(1280,210*len(cards[start:start+5])), '#dddddd')
        for j,card in enumerate(cards[start:start+5]): sheet.paste(card,(0,j*210))
        sheet.save(out/f'contact-{page:02d}.jpg',quality=94)
    save(out/'summary.json',summary); save(out/'cases.json',selected)
    (out/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>AnimalAct 错误检查</title><style>body{font:16px system-ui;max-width:1320px;margin:30px auto;background:#f4f6f8}article{background:white;padding:16px;margin:20px 0}img{width:100%}video{width:640px;max-width:100%}pre{white-space:pre-wrap}</style><h1>AnimalAct 典型错误与正确对照</h1><p>验证集定向抽样；不代表错误原因的总体比例。视频链接指向本机原文件。GT 指当前标注，不代表人工已确认其绝对正确。</p>'+''.join(pages),encoding='utf-8')
    print(json.dumps(dict(cases=len(selected),summary=summary),ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
