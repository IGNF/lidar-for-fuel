"""
Count LiDAR points per LAS classification code and the total.

"""

import numpy as np



def compute_class_counts(classification: np.ndarray, keep_classes: list) -> dict[str, int]:
    """Count points per LAS classification code and the total.

    Args:
        classification (np.ndarray): LAS classification code, one per point
            (after the temporal filter, before any vegetation/ground subsetting).
        keep_classes (list): Classes to keep for counting points

    Returns:
        dict[str, int]: `Class_{code}` count for each tracked code, and
        `Total`, the number of points of any classification (can exceed the
        sum of the tracked `Class_*` counters if untracked codes are present).
    """
    counts = {f"Class_{code}": int(np.sum(classification == code)) for code in keep_classes}
    counts["Total"] = len(classification)
    return counts
