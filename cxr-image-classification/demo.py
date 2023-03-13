from configs import config
import pandas as pd
import os
from dataset import preprocessor
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

df = preprocessor(config["base_root"], config["csv_name"])
labels = df.iloc[:, 6:]
kf = MultilabelStratifiedKFold(
    n_splits=5, shuffle=True, random_state=42)
iterator = iter(kf.split(range(df.shape[0]), labels))
train_idx, val_idx = next(iterator)
# print(df.iloc[val_idx, 6:].sum())

labels = df.iloc[val_idx, 6:]
kf = MultilabelStratifiedKFold(
    n_splits=2, shuffle=True, random_state=42)
iterator = iter(kf.split(val_idx, labels))
_val_idx, _test_idx = next(iterator)
val_idx, test_idx = val_idx[_val_idx], val_idx[_test_idx]
print(df.iloc[test_idx, 6:].sum())