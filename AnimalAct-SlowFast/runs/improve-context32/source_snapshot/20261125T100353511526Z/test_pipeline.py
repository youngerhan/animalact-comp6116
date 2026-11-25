"""Regression checks for score identity and training/inference preprocessing."""
import tempfile
import unittest
from pathlib import Path

import numpy as np

from clips import AnimalActClips, read_json
from metrics import average_precision, evaluate
from predictions import combine
from predict import prediction_dataset
from train_strong import ClipDataset
from runtime import run_report


class PipelineTests(unittest.TestCase):
    def test_ap_ties(self):
        self.assertAlmostEqual(average_precision(np.array([True, False]), np.array([.5, .5])), .5)

    def test_identity_rejects_reordered_equal_targets(self):
        with tempfile.TemporaryDirectory() as temp:
            runs = [Path(temp) / str(i) for i in range(2)]
            for i, run in enumerate(runs):
                run.mkdir()
                np.savez(run / 'valid_predictions.npz', targets=np.ones((2, 1), dtype=bool),
                         scores=np.ones((2, 1)), global_class_ids=[1], video_ids=['a', 'b'],
                         start_frames=[0, 10], end_frames=[10, 20+i])
            with self.assertRaisesRegex(ValueError, 'end_frames'):
                combine(runs, 'valid')

    def test_report_files_agree(self):
        with tempfile.TemporaryDirectory() as temp:
            classes = [dict(global_id=1, name='Walking')]
            pairs = np.array([[True], [False]]), np.array([[.9], [.1]])
            rows = [dict(video_id=str(i), start_frame=0, end_frame=10, duration_seconds=1) for i in range(2)]
            report = run_report(Path(temp), 'valid', pairs, classes, .5, [.5], rows)
            self.assertEqual(report['mAP'], 1)
            self.assertEqual(read_json(Path(temp) / 'valid_metrics.json'), report)
            with np.load(Path(temp) / 'valid_predictions.npz') as stored:
                self.assertEqual(evaluate(stored['targets'], stored['scores'], classes, .5)['mAP'], report['mAP'])

    def test_predict_uses_training_sampling(self):
        classes, clip = [{}], dict(video_path='unused.mp4', start_frame=0, end_frame=100, labels=[])
        baseline = prediction_dataset(Path('.'), clip, classes, dict(frames=16, stride=2, image_size=128))
        self.assertIsInstance(baseline, AnimalActClips)
        self.assertEqual(baseline.sampling, 'fixed')
        strong = prediction_dataset(Path('.'), clip, classes,
                                    dict(frames=32, image_size=224, eval_zoom=1.15, crop='whole'))
        self.assertIsInstance(strong, ClipDataset)
        self.assertEqual(strong._frame_indices(0, 100)[-1], 99)
        self.assertEqual(strong.crop, 'whole')


if __name__ == '__main__':
    unittest.main()
