
import math
from collections import Counter


def entropy(labels):
    """
    Calculate the entropy of a collection of class labels.

    Entropy measures the impurity or uncertainty in a dataset.
    A value of 0 means all samples belong to the same class.
    """
    total = len(labels)

    if total == 0:
        return 0.0

    counts = Counter(labels)
    result = 0.0

    for count in counts.values():
        probability = count / total
        result -= probability * math.log2(probability)

    return result


def information_gain(parent_labels, partitions):
    """
    Calculate information gain after splitting a dataset.

    parent_labels: labels before splitting.
    partitions: a list of label lists, one for each partition.
    """
    total = len(parent_labels)

    if total == 0:
        return 0.0

    parent_entropy = entropy(parent_labels)
    weighted_entropy = 0.0

    for partition in partitions:
        if partition:
            weight = len(partition) / total
            weighted_entropy += weight * entropy(partition)

    return parent_entropy - weighted_entropy


def split_information(partitions):
    """
    Calculate the split information of a partition.

    This measures how broadly the samples are distributed
    across the resulting groups.
    """
    total = sum(len(partition) for partition in partitions)

    if total == 0:
        return 0.0

    result = 0.0

    for partition in partitions:
        if partition:
            probability = len(partition) / total
            result -= probability * math.log2(probability)

    return result


def gain_ratio(parent_labels, partitions):
    """
    Calculate gain ratio, the main attribute-selection measure
    used by the C4.5 decision tree algorithm.
    """
    gain = information_gain(parent_labels, partitions)
    split_info = split_information(partitions)

    if split_info == 0:
        return 0.0

    return gain / split_info
