
from c45.tree import C45DecisionTree


def test_tree_learns_simple_categorical_data():
    X = [
        ["Güneşli", "Sıcak"],
        ["Güneşli", "Soğuk"],
        ["Yağmurlu", "Sıcak"],
        ["Yağmurlu", "Soğuk"],
    ]
    y = ["Hayır", "Hayır", "Evet", "Evet"]

    model = C45DecisionTree()
    model.fit(X, y, feature_names=["Hava", "Sıcaklık"])

    predictions = model.predict(X)

    assert predictions == y


def test_tree_predicts_majority_for_unseen_value():
    X = [
        ["Güneşli"],
        ["Güneşli"],
        ["Yağmurlu"],
    ]
    y = ["Hayır", "Hayır", "Evet"]

    model = C45DecisionTree()
    model.fit(X, y, feature_names=["Hava"])

    prediction = model.predict_one(["Bulutlu"])

    assert prediction == "Hayır"


def test_tree_rejects_empty_training_data():
    model = C45DecisionTree()

    try:
        model.fit([], [])
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError for empty training data")


def test_tree_rejects_mismatched_sample_counts():
    model = C45DecisionTree()

    try:
        model.fit([["Güneşli"]], [])
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError for mismatched sample counts")
    
    

def test_tree_learns_numeric_data():
    X = [[18], [22], [35], [45]]
    y = ["Hayır", "Hayır", "Evet", "Evet"]

    model = C45DecisionTree()
    model.fit(X, y, feature_names=["Yaş"])

    assert model.predict([[20], [40]]) == ["Hayır", "Evet"]
    
    

def test_tree_predicts_with_multiple_features():
    X = [
        ["Güneşli", 18],
        ["Güneşli", 22],
        ["Yağmurlu", 35],
        ["Yağmurlu", 45],
    ]
    y = ["Hayır", "Hayır", "Evet", "Evet"]

    model = C45DecisionTree()
    model.fit(X, y, feature_names=["Hava", "Yaş"])

    assert model.predict(X) == y



def test_tree_predicts_numeric_and_categorical_data():
    X = [
        ["Güneşli", 18],
        ["Güneşli", 22],
        ["Yağmurlu", 35],
        ["Yağmurlu", 45],
    ]
    y = ["Hayır", "Hayır", "Evet", "Evet"]

    model = C45DecisionTree()
    model.fit(X, y, feature_names=["Hava", "Yaş"])

    assert model.predict(X) == y
    assert model.predict_one(["Güneşli", 20]) == "Hayır"
    assert model.predict_one(["Yağmurlu", 40]) == "Evet"


def test_print_tree_shows_learned_feature(capsys):
    X = [[18], [22], [35], [45]]
    y = ["Hayır", "Hayır", "Evet", "Evet"]

    model = C45DecisionTree()
    model.fit(X, y, feature_names=["Yaş"])
    model.print_tree()

    output = capsys.readouterr().out

    assert "Yaş" in output
    assert "Tahmin: Hayır" in output
    assert "Tahmin: Evet" in output
    

def test_tree_depth():
    X = [[18], [22], [35], [45]]
    y = ["Hayır", "Hayır", "Evet", "Evet"]

    model = C45DecisionTree()
    model.fit(X, y, feature_names=["Yaş"])

    assert model.get_depth() == 1




def test_tree_leaf_count():
    X = [[18], [22], [35], [45]]
    y = ["Hayır", "Hayır", "Evet", "Evet"]

    model = C45DecisionTree()
    model.fit(X, y, feature_names=["Yaş"])

    assert model.get_leaf_count() == 2
    

def test_prune_to_depth_reduces_tree():
    X = [[18], [22], [35], [45]]
    y = ["Hayır", "Hayır", "Evet", "Evet"]

    model = C45DecisionTree()
    model.fit(X, y, feature_names=["Yaş"])

    original_leaf_count = model.get_leaf_count()

    model.prune_to_depth(0)

    assert model.get_depth() == 0
    assert model.get_leaf_count() == 1
    assert original_leaf_count == 2
    assert model.predict([[20], [40]]) == ["Hayır", "Hayır"]



import pytest


def test_prune_to_depth_rejects_negative_depth():
    model = C45DecisionTree()
    model.fit([[18], [22], [35], [45]],
              ["Hayır", "Hayır", "Evet", "Evet"])

    with pytest.raises(ValueError, match="max_depth cannot be negative"):
        model.prune_to_depth(-1)
