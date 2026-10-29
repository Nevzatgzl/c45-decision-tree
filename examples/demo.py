
from c45.tree import C45DecisionTree

# Eğitim verileri
X = [
    ["Güneşli", 18],
    ["Güneşli", 22],
    ["Yağmurlu", 35],
    ["Yağmurlu", 45],
    ["Bulutlu", 28],
    ["Bulutlu", 32],
]

y = ["Hayır", "Hayır", "Evet", "Evet", "Evet", "Evet"]

# Modeli eğit
model = C45DecisionTree()
model.fit(X, y, feature_names=["Hava", "Yaş"])

# Yeni örnekler üzerinde tahmin yap
test_samples = [
    ["Güneşli", 20],
    ["Yağmurlu", 40],
    ["Bulutlu", 30],
]


predictions = model.predict(test_samples)

for sample, prediction in zip(test_samples, predictions):
    print(f"Girdi: {sample} -> Tahmin: {prediction}")

print("\nÖğrenilen karar ağacı:")
model.print_tree()