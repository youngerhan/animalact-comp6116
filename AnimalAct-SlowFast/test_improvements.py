"""Check temporal boundaries, label-independent ROI fallback and loss numerics."""
import unittest
import numpy as np
import torch
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from losses import AsymmetricLoss
from video_ops import context_interval, motion_box


class ImprovementTests(unittest.TestCase):
    def test_evaluation_never_reads_outside_original_clip(self):
        from train_strong import ClipDataset
        class Reader:
            def __len__(self): return 100
            def get_avg_fps(self): return 24
            def get_batch(self, indices):
                self.indices = indices
                return SimpleNamespace(asnumpy=lambda: np.zeros((len(indices),8,8,3),dtype=np.uint8))
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); (root/'video.mp4').write_bytes(b'fake')
            row=dict(video_path='video.mp4',start_frame=40,end_frame=43,labels=[0])
            for train in (False,True):
                reader=Reader()
                with patch('decord.VideoReader',return_value=reader):
                    dataset=ClipDataset(root,[row],1,32,8,train,crop='whole',min_context_seconds=.75,
                                        flip=0,colour=0)
                    dataset[0]
                if train:
                    self.assertTrue(min(reader.indices)<40 or max(reader.indices)>=43)
                else:
                    self.assertGreaterEqual(min(reader.indices),40)
                    self.assertLess(max(reader.indices),43)

    def test_historical_center_crop_rounding(self):
        from train_strong import ClipDataset
        nearest=ClipDataset('.',[],31,16,256,False)
        legacy=ClipDataset('.',[],31,16,256,False,center_rounding='floor')
        self.assertEqual(nearest._eval_crop(360,640)[1][0],134)
        self.assertEqual(legacy._eval_crop(360,640)[1][0],133)

    def test_context_preserves_original_and_bounds(self):
        for start, end in [(0, 3), (97, 100), (40, 43), (20, 80)]:
            left, right = context_interval(start, end, 24, 100, 1)
            self.assertLessEqual(left, start)
            self.assertGreaterEqual(right, end)
            self.assertGreaterEqual(left, 0)
            self.assertLessEqual(right, 100)
            self.assertGreaterEqual(right - left, 24)

    def test_static_roi_falls_back(self):
        self.assertEqual(motion_box(np.zeros((8, 90, 160, 3), dtype=np.uint8)), (0, 0, 160, 90))

    def test_local_motion_roi_contains_signal(self):
        frames = np.zeros((8, 90, 160, 3), dtype=np.uint8)
        for i in range(8):
            frames[i, 40:50, 70+i:80+i] = 255
        left, top, right, bottom = motion_box(frames)
        self.assertLessEqual(left, 70)
        self.assertGreaterEqual(right, 87)
        self.assertLessEqual(top, 40)
        self.assertGreaterEqual(bottom, 50)

    def test_zero_gamma_and_clip_matches_bce(self):
        logits = torch.tensor([[-100., -1., 0., 1., 100.]], requires_grad=True)
        targets = torch.tensor([[1., 0., 1., 0., 0.]])
        actual = AsymmetricLoss(0, 0, 0)(logits, targets)
        expected = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets)
        self.assertTrue(torch.allclose(actual, expected))
        actual.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_asymmetric_extremes_are_finite(self):
        logits = torch.tensor([[-100., -1., 0., 1., 100.]], requires_grad=True)
        loss = AsymmetricLoss()(logits, torch.tensor([[1., 0., 1., 0., 0.]]))
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(logits.grad).all())


if __name__ == '__main__':
    unittest.main()
