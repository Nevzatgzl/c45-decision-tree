
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from c45.metrics import (
    entropy,
    information_gain,
    split_information,
    gain_ratio,
)


def test_entropy():
    assert entropy(["Evet", "Evet", "Hayır", "Hayır"]) == 1.0
    assert entropy(["Evet", "Evet", "Evet"]) == 0.0
    assert entropy([]) == 0.0


def test_split_information():
    partitions = [
        ["Evet", "Evet"],
        ["Hayır", "Hayır"],
    ]

    assert math.isclose(split_information(partitions), 1.0)


def test_information_gain():
    parent = ["Evet", "Evet", "Hayır", "Hayır"]
    partitions = [
        ["Evet", "Evet"],
        ["Hayır", "Hayır"],
    ]

    assert math.isclose(information_gain(parent, partitions), 1.0)


def test_gain_ratio():
    parent = ["Evet", "Evet", "Hayır", "Hayır"]
    partitions = [
        ["Evet", "Evet"],
        ["Hayır", "Hayır"],
    ]

    assert math.isclose(gain_ratio(parent, partitions), 1.0)


def test_gain_ratio_for_zero_split_information():
    parent = ["Evet", "Hayır"]
    partitions = [parent]

    assert gain_ratio(parent, partitions) == 0.0
