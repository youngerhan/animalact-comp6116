"""Validate the released AnimalAct files, fixed partitions and class statistics."""
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPLITS = ('train', 'valid', 'test')


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):h.update(block)
    return h.hexdigest()


def main():
    manifest = json.loads((ROOT/'manifest.json').read_text())
    policy = json.loads((ROOT/'splits/split_manifest.json').read_text())
    vocab = json.loads((ROOT/'action_classes.json').read_text())
    main_classes = json.loads((ROOT/'main_evaluation_classes.json').read_text())
    samples = manifest['samples'];lookup = {s['video_id']:s for s in samples}
    assert len(samples)==len(lookup)==manifest['count']==vocab['dataset_statistics']['video_count']
    assert len(vocab['classes'])==119
    assert [c['class_id'] for c in vocab['classes']]==list(range(119))
    assert sha(ROOT/'manifest.json')==policy['manifest_sha256']==vocab['source_manifest_sha256']
    assert manifest['split_version']==policy['split_version']==main_classes['split_version']==vocab['split_version']
    assert Counter(s['split'] for s in samples)==policy['target_video_counts']
    sets={split:set() for split in SPLITS};group_splits=defaultdict(set)
    video_hash_splits=defaultdict(set);counts=defaultdict(Counter);intervals=defaultdict(Counter);groups=defaultdict(lambda:defaultdict(set))
    for s in samples:
        split=s['split'];vid=s['video_id'];sets[split].add(vid);group_splits[s['split_group_id']].add(split)
        for kind,key,hashkey,suffix in [('videos','video_path','video_sha256','.mp4'),('annotations','annotation_path','annotation_sha256','.json')]:
            path=ROOT/s[key];assert path==ROOT/kind/split/(vid+suffix)
            assert sha(path)==s[hashkey],f'Checksum mismatch: {path}'
        video_hash_splits[s['video_sha256']].add(split)
        ann=json.loads((ROOT/s['annotation_path']).read_text());assert ann['video_id']==vid and ann['video']==s['video']
        assert abs(ann['fps']-s['fps'])<0.05, f'Annotation fps mismatch: {vid}'
        for animal in ann['annotations']:
            for a in animal['actions']:
                assert 0<=a['start']<a['end']<=s['duration']+1/s['fps']+1e-4, vid
                for field,seconds in [('start_frame',a['start']),('end_frame',a['end'])]:
                    if field in a:assert abs(a[field]-seconds*s['fps'])<=1.05, (vid,field)
        labels=[a['action'] for animal in ann['annotations'] for a in animal['actions']]
        assert len(labels)==s['action_count']
        for label in labels:intervals[label][split]+=1
        for label in set(labels):counts[label][split]+=1;groups[label][split].add(s['split_group_id'])
    assert all(len(s)==1 for s in group_splits.values())
    assert all(len(s)==1 for s in video_hash_splits.values())
    assert len(group_splits)==policy['group_count']
    for split in SPLITS:
        ids=(ROOT/f'splits/{split}.txt').read_text().splitlines();assert len(ids)==len(set(ids)) and set(ids)==sets[split]
        sub=json.loads((ROOT/f'splits/{split}.json').read_text());assert sub['count']==len(ids)
        assert sub['samples']==[lookup[v] for v in ids]
        for kind,suffix in [('videos','.mp4'),('annotations','.json')]:
            assert {p.stem for p in (ROOT/kind/split).glob('*'+suffix)}==sets[split]
            assert not list((ROOT/kind).glob('*'+suffix))
    with (ROOT/'splits/assignments.csv').open() as f:
        rows=list(csv.DictReader(f));assert len(rows)==len(samples)
        for r in rows:assert all(str(lookup[r['video_id']][k])==value for k,value in r.items())
    for r in policy['assignments']:
        assert all(lookup[r['video_id']][k]==value for k,value in r.items())
    audit=json.loads((ROOT/'splits/duplicate_audit.json').read_text());fp_splits=defaultdict(set)
    for fp in audit['frame_fingerprints']:fp_splits[fp['sha256']].add(lookup[fp['video_id']]['split'])
    assert all(len(s)==1 for s in fp_splits.values())
    for link in audit['links']:assert lookup[link['left']]['split_group_id']==lookup[link['right']]['split_group_id']
    expected_main=[]
    for c in vocab['classes']:
        name=c['name'];assert sum(counts[name].values())==c['video_count'];assert sum(intervals[name].values())==c['interval_count']
        for s in SPLITS:
            assert counts[name][s]==c[s+'_video_count'];assert intervals[name][s]==c[s+'_interval_count'];assert len(groups[name][s])==c[s+'_group_count']
        total=sum(len(groups[name][s]) for s in SPLITS);assert total==c['group_count'] and len(groups[name]['train'])>=1
        if total>=3:assert all(groups[name][s] for s in SPLITS)
        eligible=total>=20 and len(groups[name]['valid'])>=3 and len(groups[name]['test'])>=3
        assert c['main_evaluation']==eligible
        if eligible:expected_main.append(c)
    assert main_classes['classes']==expected_main
    assert main_classes['class_ids']==[c['class_id'] for c in expected_main]
    assert main_classes['class_names']==[c['name'] for c in expected_main]
    assert main_classes['num_classes']==len(expected_main)==policy['main_evaluation_class_count']
    interval_total=sum(sum(c.values()) for c in intervals.values())
    assert interval_total==vocab['dataset_statistics']['action_interval_count']
    print(json.dumps(dict(status='passed',videos=len(samples),classes=119,intervals=interval_total,
                         splits={s:len(ids) for s,ids in sets.items()},groups=len(group_splits),
                         main_classes=len(expected_main),checks='All video and annotation SHA-256 hashes; paths; disjoint partitions; grouping evidence; all class statistics; fixed lists'),indent=2))


if __name__=='__main__':main()
