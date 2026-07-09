import numpy as np

from lidar_for_fuel.commons.compute_class_counts import compute_class_counts

_TRACKED_CLASSES = [1, 2, 3, 4, 5, 6, 9, 17, 18, 64, 66, 67]


def test_compute_class_counts_counts_each_tracked_class():
    classification = np.array([1, 1, 2, 3, 4, 5, 6, 9, 17, 18, 64, 66, 67], dtype=np.float64)

    result = compute_class_counts(classification)

    assert result["Class_1"] == 2
    for code in _TRACKED_CLASSES[1:]:
        assert result[f"Class_{code}"] == 1
    assert result["Total"] == len(classification)


def test_compute_class_counts_total_includes_untracked_classes():
    """`Total` counts every point regardless of classification, even codes not
    individually tracked (e.g. 7 = low noise), so `Total` can exceed the sum of
    the per-class counters."""
    classification = np.array([1, 7, 7, 12], dtype=np.float64)

    result = compute_class_counts(classification)

    assert result["Class_1"] == 1
    assert result["Total"] == 4
    assert result["Total"] > sum(result[f"Class_{code}"] for code in _TRACKED_CLASSES)


def test_compute_class_counts_all_keys_present_even_when_zero():
    classification = np.zeros(3, dtype=np.float64)

    result = compute_class_counts(classification)

    assert set(result) == {f"Class_{code}" for code in _TRACKED_CLASSES} | {"Total"}
    assert all(result[f"Class_{code}"] == 0 for code in _TRACKED_CLASSES)
    assert result["Total"] == 3
