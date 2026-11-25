"""Run bounded local improvements; freeze validation choices before evaluating test."""
import argparse
import datetime
import itertools
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

from clips import read_json, write_json
from metrics import evaluate, per_class_thresholds, select_threshold
from runtime import PROJECT, run_report

RESULTS = PROJECT / 'results' / 'improvement-20261125'
VARIANTS = [
    ('context32', ['--min-context-seconds', '.75']),
    ('motion32', ['--spatial-view', 'motion']),
    ('asymmetric32', ['--loss', 'asymmetric']),
]


def progress(**values):
    write_json(RESULTS / 'progress.json', dict(time=datetime.datetime.now().isoformat(), **values))


def train(name, extra):
    out = PROJECT / 'runs' / ('improve-' + name)
    if (out / 'evaluation-ready.json').exists():
        return out
    command = [sys.executable, '-u', str(PROJECT / 'train_strong.py'),
               '--initial', str(PROJECT / 'runs/pilot32-whole224/best.pth'), '--out', str(out),
               '--crop', 'whole', '--frames', '32', '--image-size', '224', '--epochs', '4',
               '--schedule-epochs', '4', '--patience', '2', '--warmup', '0',
               '--batch-size', '8', '--workers', '6', '--head-lr', '.0001', '--backbone-lr', '.00001',
               '--keep-initial', *extra]
    if (out / 'last.pth').exists():
        command.append('--resume')
    elif out.exists():
        raise RuntimeError(f'Incomplete output without resumable state: {out}')
    progress(stage='training', variant=name, command=command)
    env = os.environ.copy()
    env.update(PYTHONIOENCODING='utf-8', PYTHONDONTWRITEBYTECODE='1')
    with (RESULTS / (name + '.log')).open('a', encoding='utf-8') as log:
        child = subprocess.Popen(command, cwd=PROJECT, env=env, stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace')
        for line in child.stdout:
            log.write(line); log.flush(); print(line, end='', flush=True)
        if child.wait():
            raise RuntimeError(f'Training failed: {name}; inspect {log.name}')
    write_json(out / 'evaluation-ready.json', dict(completed=True))
    return out


def model_for(run):
    from mmaction.apis import init_recognizer
    from mmengine.config import Config
    settings = argparse.Namespace(**read_json(run / 'settings.json'))
    best = torch.load(run / 'best.pth', map_location='cpu', weights_only=True)
    config = Config.fromfile(settings.config)
    config.model.cls_head.num_classes = len(best['classes'])
    model = init_recognizer(config, checkpoint=None, device='cuda:0')
    model.load_state_dict(best['state_dict'])
    return model, settings, best


def inference(run, rows, flip=False):
    from evaluate_run import scores
    model, settings, best = model_for(run)
    variant = dict(name='flip' if flip else 'plain', zoom=settings.eval_zoom, tta=flip,
                   crops=(.5,), windows=((0., 1.),))
    mean = torch.tensor([123.675, 116.28, 103.53], device='cuda').view(1, 3, 1, 1, 1)
    std = torch.tensor([58.395, 57.12, 57.375], device='cuda').view(1, 3, 1, 1, 1)
    pairs = scores(model, rows, settings, mean, std, variant)
    del model, best
    torch.cuda.empty_cache()
    return pairs


def valid_variants(run, manifest):
    out = run.with_name(run.name + '-evaluation')
    if (out / 'selection.json').exists():
        return out
    out.mkdir(exist_ok=False)
    classes = manifest['classes']; rows = manifest['splits']['valid']
    with np.load(run / 'valid_predictions.npz') as saved:
        plain = saved['targets'], saved['scores']
    progress(stage='validation_tta', run=str(run))
    mirrored = inference(run, rows, flip=True)
    if not np.array_equal(plain[0], mirrored[0]):
        raise ValueError('Validation clip order differs')
    table = []
    for name, pairs in [('plain', plain), ('flip', mirrored)]:
        threshold = select_threshold(*pairs)
        report = evaluate(*pairs, classes, threshold)
        table.append(dict(variant=name, mAP=report['mAP'], macro_F1=report['macro_F1'],
                          threshold=threshold, utility=(report['mAP'] + report['macro_F1']) / 2))
    eligible = [r for r in table if r['mAP'] >= table[0]['mAP'] - .001]
    selected = max(eligible, key=lambda r: r['utility'])
    pairs = plain if selected['variant'] == 'plain' else mirrored
    run_report(out, 'valid', pairs, classes, selected['threshold'],
               per_class_thresholds(*pairs), rows)
    best = torch.load(run / 'best.pth', map_location='cpu', weights_only=True)
    write_json(out / 'selection.json', dict(source_run=str(run), selected=selected,
               variants=table, best_epoch=best['epoch']))
    print('Validation TTA:', run.name, selected, flush=True)
    return out


def choose_fusion(directories, manifest):
    base = PROJECT / 'runs/pilot32-blend'
    with np.load(base / 'valid_predictions.npz') as saved:
        targets, baseline_scores = saved['targets'], saved['scores']
    baseline = read_json(base / 'valid_metrics.json')
    arrays = [baseline_scores]
    for directory in directories:
        with np.load(directory / 'valid_predictions.npz') as saved:
            if not np.array_equal(saved['targets'], targets):
                raise ValueError('Fusion target mismatch')
            arrays.append(saved['scores'])
    table = []
    for raw in itertools.product((0, 1, 2, 3), repeat=len(directories)):
        if sum(raw) > 3:
            continue  # Preserve at least 25% of the existing ensemble.
        weights = [1 - sum(raw) / 4] + [r / 4 for r in raw]
        combined = sum(w * a for w, a in zip(weights, arrays))
        threshold = select_threshold(targets, combined)
        report = evaluate(targets, combined, manifest['classes'], threshold)
        table.append(dict(weights=weights, threshold=threshold, mAP=report['mAP'],
                          macro_F1=report['macro_F1'], utility=(report['mAP'] + report['macro_F1']) / 2))
    eligible = [r for r in table if r['mAP'] >= baseline['mAP'] - 1e-12
                and r['macro_F1'] >= baseline['macro_F1'] - 1e-12]
    selected = max(eligible, key=lambda r: r['utility'])
    plan = dict(frozen_before_test=True, time=datetime.datetime.now().isoformat(),
                members=[str(base)] + [str(r) for r in directories],
                candidates=table, selected=selected, baseline=baseline,
                selection_rule='Nondecreasing validation mAP AND Macro-F1, then maximum mean of both; weight grid .25, baseline weight >=.25')
    write_json(RESULTS / 'selection-frozen.json', plan)
    return plan, targets, sum(w * a for w, a in zip(selected['weights'], arrays))


def final_evaluation(directories, manifest, plan, targets, valid_scores):
    selected = plan['selected']; base = PROJECT / 'runs/pilot32-blend'
    classes = manifest['classes']; rows = manifest['splits']['test']
    with np.load(base / 'test_predictions.npz') as saved:
        test_targets, test_base = saved['targets'], saved['scores']
    final_scores = selected['weights'][0] * test_base
    for weight, directory in zip(selected['weights'][1:], directories):
        if not weight:
            continue
        detail = read_json(directory / 'selection.json')
        progress(stage='final_test', run=detail['source_run'])
        pairs = inference(Path(detail['source_run']), rows,
                          flip=detail['selected']['variant'] == 'flip')
        if not np.array_equal(pairs[0], test_targets):
            raise ValueError('Test target mismatch')
        run_report(directory, 'test', pairs, classes, detail['selected']['threshold'],
                   read_json(directory / 'valid_metrics.json')['per_class_thresholds'], rows)
        final_scores += weight * pairs[1]
    out = PROJECT / 'runs/improved-final-20261125'
    out.mkdir(exist_ok=False)
    thresholds = per_class_thresholds(targets, valid_scores)
    run_report(out, 'valid', (targets, valid_scores), classes, selected['threshold'],
               thresholds, manifest['splits']['valid'])
    report = run_report(out, 'test', (test_targets, final_scores), classes,
                        selected['threshold'], thresholds, rows)
    write_json(out / 'members.json', plan)
    before = read_json(base / 'test_metrics.json')
    counts = lambda r: {key: sum(c[key] for c in r['per_class']) for key in ('TP', 'FP', 'FN', 'TN')}
    summary = dict(out=str(out), validation_selected=selected,
                   test={k: report[k] for k in ('mAP', 'macro_F1', 'threshold')},
                   baseline_test={k: before[k] for k in ('mAP', 'macro_F1', 'threshold')},
                   delta_pp={k: 100 * (report[k] - before[k]) for k in ('mAP', 'macro_F1')},
                   before_counts=counts(before), after_counts=counts(report))
    rng = np.random.default_rng(42); groups = {}
    for i, row in enumerate(rows):
        groups.setdefault(row['video_id'], []).append(i)
    groups = list(groups.values()); deltas = []
    for _ in range(300):
        ids = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        old = evaluate(test_targets[ids], test_base[ids], classes, before['threshold'])
        new = evaluate(test_targets[ids], final_scores[ids], classes, selected['threshold'])
        deltas.append([100 * (new[k] - old[k]) for k in ('mAP', 'macro_F1')])
    summary['paired_video_bootstrap_95_interval_pp'] = dict(zip(('mAP', 'macro_F1'),
                                                              np.percentile(deltas, [2.5, 97.5], axis=0).T.tolist()))
    write_json(RESULTS / 'summary.json', summary)
    progress(stage='completed', summary=summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('all', 'train', 'evaluate'), default='all')
    args = parser.parse_args()
    torch.set_num_threads(2)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU is required')
    RESULTS.mkdir(exist_ok=True, parents=True)
    write_json(RESULTS / 'plan.json', dict(variants=VARIANTS, maximum_epochs_each=4,
               early_stop_patience=2, initial='runs/pilot32-whole224/best.pth',
               seed=42, labels_unchanged=True, test_used_for_selection=False))
    directories = []
    for name, extra in VARIANTS:
        if args.stage != 'evaluate':
            run = train(name, extra)
        else:
            run = PROJECT / 'runs' / ('improve-' + name)
        directories.append(run)
    if args.stage == 'train':
        return
    manifest = read_json(PROJECT / 'data/main.json')
    evaluations = [valid_variants(run, manifest) for run in directories]
    plan, targets, valid_scores = choose_fusion(evaluations, manifest)
    if not any(plan['selected']['weights'][1:]):
        progress(stage='no_validation_improvement', selection=plan['selected'])
        print('No new model improves both validation metrics; original model retained.', flush=True)
        return
    final_evaluation(evaluations, manifest, plan, targets, valid_scores)


if __name__ == '__main__':
    main()
