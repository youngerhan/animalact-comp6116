"""Select the new model's blend weight on validation, then evaluate saved test scores."""
import argparse
from pathlib import Path

from clips import read_json, write_json
from metrics import evaluate, per_class_thresholds, select_threshold
from predictions import combine
from runtime import PROJECT, run_report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--new', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, default=PROJECT / 'data' / 'main.json')
    parser.add_argument('--candidates', type=float, nargs='+', default=[0, .125, .25, .5, 1])
    parser.add_argument('--allow-legacy', action='store_true')
    args = parser.parse_args()
    if any(not 0 <= alpha <= 1 for alpha in args.candidates):
        parser.error('Candidate weights must be between 0 and 1.')
    manifest = read_json(args.manifest)
    classes = manifest['classes']
    runs = [args.base, args.new]
    candidates = []
    for alpha in args.candidates:
        pairs = combine(runs, 'valid', [1-alpha, alpha], args.allow_legacy)
        threshold = select_threshold(*pairs)
        measured = evaluate(*pairs, classes, threshold)
        candidates.append(dict(new_weight=alpha, valid_mAP=measured['mAP'], threshold=threshold))
    chosen = max(candidates, key=lambda row: row['valid_mAP'])
    weights = [1-chosen['new_weight'], chosen['new_weight']]
    valid = combine(runs, 'valid', weights, args.allow_legacy)
    thresholds = per_class_thresholds(*valid)
    args.out.mkdir(parents=True, exist_ok=False)
    for split in ('valid', 'test'):
        pairs = valid if split == 'valid' else combine(runs, split, weights, args.allow_legacy)
        run_report(args.out, split, pairs, classes, chosen['threshold'], thresholds,
                   manifest['splits'][split])
    write_json(args.out / 'members.json', dict(members=[str(path) for path in runs],
               weights=weights, candidates=candidates, selected=chosen,
               legacy_identity_unverified=args.allow_legacy))


if __name__ == '__main__':
    main()
