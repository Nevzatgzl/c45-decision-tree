# C4.5 Decision Tree Algorithm from Scratch

**A from-scratch implementation of the C4.5 decision tree algorithm using Python.**

This project implements a decision tree classifier with a focus on understanding the core concepts behind the C4.5 algorithm, including Entropy, Information Gain, and Gain Ratio.

## Features

- Entropy calculation
- Information Gain calculation
- Split Information and Gain Ratio
- Categorical feature handling
- Numeric feature splitting using thresholds
- Recursive decision tree construction
- Predictions for individual samples and batches
- Majority-class prediction for unseen categorical values
- Tree visualization in the terminal
- Tree depth and leaf count calculation
- Depth-limiting functionality
- Automated tests using `pytest`

## Project Structure

```text
c45-decision-tree/
├── data/
├── docs/
├── examples/
│   └── demo.py
├── src/
│   └── c45/
│       ├── __init__.py
│       ├── metrics.py
│       └── tree.py
├── tests/
│   ├── test_metrics.py
│   ├── test_numeric_split.py
│   └── test_tree.py
├── .gitignore
├── pyproject.toml
└── README.md
```

## Installation

Python 3.10 or newer is required.

Clone the repository:

```bash
git clone https://github.com/Nevzatgzl/c45-decision-tree.git
cd c45-decision-tree
```

Create and activate a virtual environment:

**Windows PowerShell**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install the project:

```bash
python -m pip install -e .
```

Install the testing dependency:

```bash
python -m pip install pytest
```

## Usage

Run the example:

```bash
python examples/demo.py
```

Example:

```python
from c45.tree import C45DecisionTree

X = [
    ["Sunny", 18],
    ["Sunny", 22],
    ["Rainy", 35],
    ["Rainy", 45],
    ["Cloudy", 28],
    ["Cloudy", 32],
]

y = ["No", "No", "Yes", "Yes", "Yes", "Yes"]

model = C45DecisionTree()
model.fit(X, y, feature_names=["Weather", "Age"])

predictions = model.predict([
    ["Sunny", 20],
    ["Rainy", 40],
])

print(predictions)
model.print_tree()
```

## Running Tests

Run the automated test suite from the project root:

```bash
python -m pytest -v
```

The current test suite contains **19 passing tests** covering metrics, numeric splits, predictions, tree structure, and input validation.

## Algorithm Overview

The implementation uses the following concepts:

1. **Entropy:** Measures the uncertainty in the target labels.
2. **Information Gain:** Measures the reduction in entropy after a split.
3. **Split Information:** Measures how the samples are distributed across the resulting partitions.
4. **Gain Ratio:** Normalizes Information Gain using Split Information.
5. **Recursive Tree Construction:** Selects features and creates branches until stopping conditions are reached.

Numeric features are split using threshold values, while categorical features are handled through their distinct values.

## Current Limitations

- The implementation is designed for educational purposes and is not a replacement for mature machine-learning libraries.
- Pruning currently limits the maximum tree depth; it does not implement the classical C4.5 error-based pruning procedure.
- Further validation against reference implementations and larger datasets is needed to assess full algorithmic compatibility with the original C4.5 method.

## Türkçe

Bu proje, C4.5 karar ağacı algoritmasının temel mantığını anlamak amacıyla Python ile sıfırdan geliştirilmiştir.

Proje kapsamında:

- Entropy, Information Gain, Split Information ve Gain Ratio hesaplamaları uygulanmıştır.
- Kategorik ve sayısal özelliklerle çalışabilen karar ağacı yapısı geliştirilmiştir.
- Tek bir örnek veya birden fazla örnek için tahmin yapılabilmektedir.
- Ağacın derinliği ve yaprak sayısı hesaplanabilmektedir.
- Otomatik testlerle temel işlevler doğrulanmıştır.

Projenin mevcut sürümünde budama işlemi ağacın derinliğini sınırlandırır; klasik C4.5 hata tahminine dayalı budama algoritmasını uygulamaz.

## Technologies

- Python
- pytest
- Git and GitHub

## License

No license has been specified yet.
