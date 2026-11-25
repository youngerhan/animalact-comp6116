"""Label-independent temporal context and motion-region proposals for video clips."""
import cv2
import numpy as np


def context_interval(start, end, fps, frame_count, minimum_seconds):
    """Expand only the input interval, keeping the original clip's target unchanged."""
    length = min(frame_count, max(end - start, round(fps * minimum_seconds)))
    first = min(max(0, (start + end - length) // 2), frame_count - length)
    return first, first + length


def motion_box(frames):
    """Propose a crop from temporal differences; it is not an animal detector.

    Small/static/near-global proposals fall back to the full frame. No labels,
    species identities, manual boxes or learned detector weights are consulted.
    """
    height, width = frames.shape[1:3]
    whole = (0, 0, width, height)
    if len(frames) < 2:
        return whole
    indices = np.linspace(0, len(frames) - 1, min(8, len(frames))).round().astype(int)
    gray = np.stack([cv2.GaussianBlur(cv2.cvtColor(cv2.resize(frames[i], (160, 90)),
                    cv2.COLOR_RGB2GRAY), (3, 3), 0) for i in indices]).astype(np.float32)
    energy = np.abs(np.diff(gray, axis=0)).mean(axis=0)
    median = float(np.median(energy))
    mad = float(np.median(np.abs(energy - median)))
    mask = (energy > max(2.5, median + 3 * mad)).astype(np.uint8)
    if mask.mean() > .35:
        return whole
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if count <= 1:
        return whole
    component = stats[1:][np.argmax(stats[1:, cv2.CC_STAT_AREA])]
    x, y, w, h, area = map(int, component)
    if area < 12 or w * h > .7 * 160 * 90:
        return whole
    cx, cy = (x + w / 2) / 160, (y + h / 2) / 90
    crop_w = min(width, max(round(width * .4), round(w / 160 * width * 1.8)))
    crop_h = min(height, max(round(height * .4), round(h / 90 * height * 1.8)))
    left = min(max(0, round(cx * width - crop_w / 2)), width - crop_w)
    top = min(max(0, round(cy * height - crop_h / 2)), height - crop_h)
    return left, top, left + crop_w, top + crop_h
