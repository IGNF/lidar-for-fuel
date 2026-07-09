"""
Count LiDAR points per LAS classification code, for pixel-level QA channels.

"""

import numpy as np

_TRACKED_CLASSES = [1, 2, 3, 4, 5, 6, 9, 17, 18, 64, 66, 67]


def compute_class_counts(classification: np.ndarray) -> dict[str, int]:
    """Count points per LAS classification code, plus the pixel total.

    Args:
        classification (np.ndarray): LAS classification code, one per point
            (after the temporal filter, before any vegetation/ground subsetting).

    Returns:
        dict[str, int]: `Class_{code}` count for each tracked code, plus
        `Total`, the number of points of any classification (can exceed the
        sum of the tracked `Class_*` counters if untracked codes are present).
    """
    counts = {f"Class_{code}": int(np.sum(classification == code)) for code in _TRACKED_CLASSES}
    counts["Total"] = len(classification)
    return counts
