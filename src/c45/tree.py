
from collections import Counter
from dataclasses import dataclass, field
from numbers import Real

from c45.metrics import gain_ratio, entropy


@dataclass
class TreeNode:
    prediction: str
    feature: str | None = None
    threshold: float | None = None
    branches: dict = field(default_factory=dict)


def _best_numeric_split(values, labels):
    """Find the numeric threshold with the highest gain ratio."""
    pairs = sorted(zip(values, labels), key=lambda pair: pair[0])
    candidates = []

    for i in range(len(pairs) - 1):
        value, label = pairs[i]
        next_value, next_label = pairs[i + 1]

        if value != next_value and label != next_label:
            candidates.append((value + next_value) / 2)

    best_threshold = None
    best_ratio = 0.0

    for threshold in candidates:
        left = [
            label for value, label in pairs if value <= threshold
        ]
        right = [
            label for value, label in pairs if value > threshold
        ]

        ratio = gain_ratio(labels, [left, right])

        if ratio > best_ratio:
            best_ratio = ratio
            best_threshold = threshold

    return best_threshold, best_ratio


class C45DecisionTree:
    def __init__(self, min_samples_split=2):
        self.root = None
        self.min_samples_split = min_samples_split
        self.feature_names = []

    def fit(self, X, y, feature_names=None):
        if len(X) != len(y):
            raise ValueError("X and y must contain the same number of samples.")
        if not X:
            raise ValueError("Training data cannot be empty.")
        if not y:
            raise ValueError("Target labels cannot be empty.")

        if any(len(row) != len(X[0]) for row in X):
            raise ValueError("All rows in X must have the same number of features.")

        self.feature_names = (
            list(feature_names)
            if feature_names is not None
            else [f"feature_{i}" for i in range(len(X[0]))]
        )

        if len(self.feature_names) != len(X[0]):
            raise ValueError("Number of feature names must match X.")

        self.root = self._build_tree(
            X, y, list(range(len(self.feature_names)))
        )
        return self

    def _build_tree(self, X, y, available_features):
        majority_class = Counter(y).most_common(1)[0][0]

        if len(set(y)) == 1:
            return TreeNode(prediction=majority_class)

        if not available_features or len(X) < self.min_samples_split:
            return TreeNode(prediction=majority_class)

        parent_entropy = entropy(y)
        feature_candidates = []
        best_feature = None
        best_ratio = 0.0
        best_threshold = None
        best_groups = None

        
          
        for feature_index in available_features:
            values = [row[feature_index] for row in X]

            is_numeric = all(
                isinstance(value, Real) and not isinstance(value, bool)
                for value in values
            )

            if is_numeric:
                threshold, ratio = _best_numeric_split(values, y)

                if threshold is None:
                    continue

                left_labels = [
                    label for value, label in zip(values, y)
                    if value <= threshold
                ]
                right_labels = [
                    label for value, label in zip(values, y)
                    if value > threshold
                ]

                partitions = [left_labels, right_labels]
                information_gain_value = parent_entropy - (
                    len(left_labels) / len(y) * entropy(left_labels)
                    + len(right_labels) / len(y) * entropy(right_labels)
                )

            else:
                groups = {}

                for value, label in zip(values, y):
                    groups.setdefault(value, []).append(label)

                partitions = list(groups.values())

                information_gain_value = parent_entropy - sum(
                    len(partition) / len(y) * entropy(partition)
                    for partition in partitions
                )

                ratio = gain_ratio(y, partitions)

            feature_candidates.append({
                "feature_index": feature_index,
                "threshold": threshold if is_numeric else None,
                "ratio": ratio,
                "information_gain": information_gain_value,
                "partitions": partitions,
            })

        
        if feature_candidates:
            average_gain = sum(
                candidate["information_gain"]
                for candidate in feature_candidates
            ) / len(feature_candidates)

            eligible_candidates = [
                candidate for candidate in feature_candidates
                if candidate["information_gain"] >= average_gain
            ]

            if eligible_candidates:
                best_candidate = max(
                    eligible_candidates,
                    key=lambda candidate: candidate["ratio"],
                )

                best_feature = best_candidate["feature_index"]
                best_ratio = best_candidate["ratio"]
                best_threshold = best_candidate["threshold"]

                if best_threshold is not None:
                    values = [row[best_feature] for row in X]
                    best_groups = {
                        "<=": [
                            label for value, label in zip(values, y)
                            if value <= best_threshold
                        ],
                        ">": [
                            label for value, label in zip(values, y)
                            if value > best_threshold
                        ],
                    }
                else:
                    best_groups = {}
                    values = [row[best_feature] for row in X]

                    for value, label in zip(values, y):
                        best_groups.setdefault(value, []).append(label)

        if best_feature is None or best_ratio <= 0 or parent_entropy == 0:
            return TreeNode(prediction=majority_class)

        node = TreeNode(
            prediction=majority_class,
            feature=self.feature_names[best_feature],
            threshold=best_threshold,
        )

        remaining_features = [
            index for index in available_features if index != best_feature
        ]

        if best_threshold is not None:
            left_X = [
                row for row in X if row[best_feature] <= best_threshold
            ]
            left_y = [
                label for row, label in zip(X, y)
                if row[best_feature] <= best_threshold
            ]

            right_X = [
                row for row in X if row[best_feature] > best_threshold
            ]
            right_y = [
                label for row, label in zip(X, y)
                if row[best_feature] > best_threshold
            ]

            node.branches["<="] = self._build_tree(
                left_X, left_y, remaining_features
            )
            node.branches[">"] = self._build_tree(
                right_X, right_y, remaining_features
            )

        else:
            for value in best_groups:
                subset_X = [
                    row for row in X if row[best_feature] == value
                ]
                subset_y = [
                    label for row, label in zip(X, y)
                    if row[best_feature] == value
                ]

                node.branches[value] = self._build_tree(
                    subset_X, subset_y, remaining_features
                )

        return node

    def predict_one(self, row):
        if self.root is None:
            raise ValueError("Model has not been trained. Call fit() first.")

        node = self.root

        while node.feature is not None:
            feature_index = self.feature_names.index(node.feature)
            value = row[feature_index]

            if node.threshold is not None:
                branch = "<=" if value <= node.threshold else ">"
                node = node.branches[branch]
            else:
                if value not in node.branches:
                    return node.prediction

                node = node.branches[value]

        return node.prediction

    def predict(self, X):
        return [self.predict_one(row) for row in X]
    
    def print_tree(self):
        if self.root is None:
            raise ValueError("Model has not been trained. Call fit() first.")

        def display(node, indent=""):
            if node.feature is None:
                print(f"{indent}Tahmin: {node.prediction}")
                return

            if node.threshold is not None:
                print(
                    f"{indent}{node.feature} <= {node.threshold}"
                )
                display(node.branches["<="], indent + "  ")

                print(
                    f"{indent}{node.feature} > {node.threshold}"
                )
                display(node.branches[">"], indent + "  ")

            else:
                for value, child in node.branches.items():
                    print(f"{indent}{node.feature} = {value}")
                    display(child, indent + "  ")

        display(self.root)
    
    
    def get_depth(self):
        if self.root is None:
            raise ValueError("Model has not been trained. Call fit() first.")

        def depth(node):
            if node.feature is None:
                return 0

            return 1 + max(
                depth(child) for child in node.branches.values()
            )

        return depth(self.root)

    
    def get_leaf_count(self):
        if self.root is None:
            raise ValueError("Model has not been trained. Call fit() first.")

        def count_leaves(node):
            if node.feature is None:
                return 1

            return sum(
                count_leaves(child)
                for child in node.branches.values()
            )

        return count_leaves(self.root)
    
    def prune_to_depth(self, max_depth):
        if self.root is None:
            raise ValueError("Model has not been trained. Call fit() first.")

        if max_depth < 0:
            raise ValueError("max_depth cannot be negative.")

        def prune(node, current_depth):
            if node.feature is None:
                return

            if current_depth >= max_depth:
                node.feature = None
                node.threshold = None
                node.branches = {}
                return

            for child in node.branches.values():
                prune(child, current_depth + 1)

        prune(self.root, 0)
        return self

