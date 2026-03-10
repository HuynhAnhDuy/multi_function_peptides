import pandas as pd
import numpy as np

df_train = pd.read_csv("Liver_train_cd40_onehot_esm.csv")
df_test  = pd.read_csv("Liver_test_cd40_onehot_esm.csv")

# y lấy từ file label riêng:
y_train = pd.read_csv("Liver_train_cd40_y.csv")["Label"].values
y_test  = pd.read_csv("Liver_test_cd40_y.csv")["Label"].values

# X: bỏ Sequence + Label (nếu Label có trong file features)
drop_cols = [c for c in ["Sequence", "Label"] if c in df_train.columns]
X_train = df_train.drop(columns=drop_cols).values
X_test  = df_test.drop(columns=drop_cols).values

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
