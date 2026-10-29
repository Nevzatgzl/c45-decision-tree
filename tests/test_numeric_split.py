
from c45.tree import _best_numeric_split


def test_best_numeric_split_finds_useful_threshold():
    values = [18, 22, 35, 45]
    labels = ["Hayır", "Hayır", "Evet", "Evet"]

    threshold, ratio = _best_numeric_split(values, labels)

    assert threshold == 28.5
    assert ratio > 0


def test_numeric_split_returns_no_threshold_for_same_class():
    values = [18, 22, 35, 45]
    labels = ["Evet", "Evet", "Evet", "Evet"]

    threshold, ratio = _best_numeric_split(values, labels)

    assert threshold is None
    assert ratio == 0.0
