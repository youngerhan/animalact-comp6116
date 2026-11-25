"""Check source syntax, data reconstruction and saved metrics without changing results."""
import argparse
import ast
import tempfile
from pathlib import Path

import numpy as np

from clips import prepare, read_json
from metrics import evaluate
from runtime import PROJECT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=PROJECT.parent / 'AnimalAct')
    parser.add_argument('--skip-data', action='store_true')
    args = parser.parse_args()
    for path in PROJECT.glob('*.py'):
        ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
    manifest = read_json(PROJECT / 'data' / 'main.json')
    seen = set()
    for split, rows in manifest['splits'].items():
        videos = {row['video_id'] for row in rows}
        if seen & videos:
            raise ValueError(f'Video leakage into {split}')
        seen |= videos
    if not args.skip_data:
        with tempfile.TemporaryDirectory() as temp:
            rebuilt = Path(temp) / 'main.json'
            prepare(args.dataset, rebuilt)
            if read_json(rebuilt) != manifest:
                raise ValueError('Rebuilt manifest differs from saved main.json')
        print('PASS: rebuilt manifest equals main.json; no video split leakage')
    for name in ('baseline-stride2', 'ensemble7', 'pilot32-whole224', 'pilot32-blend'):
        run = PROJECT / 'runs' / name
        for split in ('valid', 'test'):
            saved = read_json(run / f'{split}_metrics.json')
            with np.load(run / f'{split}_predictions.npz', allow_pickle=False) as data:
                actual = evaluate(data['targets'], data['scores'], manifest['classes'], saved['threshold'])
                expected = np.zeros_like(data['targets'], dtype=bool)
                for index, row in enumerate(manifest['splits'][split]):
                    expected[index, row['labels']] = True
                if not np.array_equal(expected, data['targets']):
                    raise ValueError(f'{name}/{split}: labels differ from manifest')
            for key in ('mAP', 'macro_F1'):
                if not np.isclose(actual[key], saved[key], atol=1e-12, rtol=0):
                    raise ValueError(f'{name}/{split}: saved {key} disagrees with scores')
        print(f'PASS: {name} validation/test metrics and labels')


if __name__ == '__main__':
    main()
