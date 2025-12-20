import pandas as pd

f1 = pd.read_csv("./model_predictions/ensemble_5_fold-test.csv")
f2 = pd.read_csv("./model_predictions/ensemble_5_fold-test-horizontal-mirror.csv")

ids1 = set(f1["img_id"].unique())
ids2 = set(f2["img_id"].unique())

print("Missing in mirror CSV:", ids1 - ids2)
print("Missing in normal CSV:", ids2 - ids1)
