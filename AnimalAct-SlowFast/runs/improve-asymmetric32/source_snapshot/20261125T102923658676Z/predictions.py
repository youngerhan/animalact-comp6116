"""Validate saved score arrays before averaging; never silently reorder clips."""
import numpy as np

IDENTITY_KEYS = ("global_class_ids", "video_ids", "start_frames", "end_frames")


def load_predictions(run, split):
    with np.load(run / f"{split}_predictions.npz", allow_pickle=False) as stored:
        targets, scores = stored["targets"], stored["scores"]
        metadata = {key: stored[key] for key in IDENTITY_KEYS if key in stored.files}
    if targets.ndim != 2 or targets.shape != scores.shape or targets.dtype != np.bool_:
        raise ValueError(f"Invalid prediction shapes or target type: {run}")
    if not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
        raise ValueError(f"Expected finite probabilities in [0, 1]: {run}")
    return targets, scores, metadata


def combine(runs, split, weights=None, allow_legacy=False):
    arrays = [load_predictions(run, split) for run in runs]
    targets, _, identity = arrays[0]
    for other_targets, _, other in arrays:
        if not np.array_equal(targets, other_targets):
            raise ValueError("Ensemble targets or clip order differ.")
        for key in IDENTITY_KEYS:
            if key not in identity or key not in other:
                if not allow_legacy:
                    raise ValueError(f"Missing {key}; only historical files may use --allow-legacy.")
            elif not np.array_equal(identity[key], other[key]):
                raise ValueError(f"Ensemble identity mismatch: {key}")
    if weights is not None:
        weights = np.asarray(weights, dtype=float)
        if len(weights) != len(runs) or not np.isfinite(weights).all() or (weights < 0).any() or weights.sum() <= 0:
            raise ValueError("Expected one nonnegative weight per run, with positive total.")
    return targets, np.average([item[1] for item in arrays], axis=0, weights=weights).astype(arrays[0][1].dtype)
