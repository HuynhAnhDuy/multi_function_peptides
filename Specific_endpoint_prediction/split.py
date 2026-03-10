import pandas as pd
from sklearn.model_selection import train_test_split

# ================= CẤU HÌNH =================
# Mỗi phần tử: (tên_data, file_input_balanced)
DATASETS = [
    ("Liver", "Liver_balanced_dataset_cd40.csv"),
    ("Lung",  "Lung_balanced_dataset_cd40.csv"),
    ("Colon", "Colon_balanced_dataset_cd40.csv"),
]

SEQ_COL = "Sequence"
LABEL_COL = "Label"   # chú ý đúng tên cột trong file balanced
TEST_SIZE = 0.2
RANDOM_STATE = 42  # để tái lập kết quả


def split_dataset(name, infile):
    print(f"\n===== Splitting dataset: {name} ({infile}) =====")
    df = pd.read_csv(infile)

    if SEQ_COL not in df.columns or LABEL_COL not in df.columns:
        raise ValueError(
            f"File {infile} must contain columns '{SEQ_COL}' and '{LABEL_COL}'"
        )

    # X, y
    X = df[SEQ_COL].astype(str)
    y = df[LABEL_COL].astype(int)

    # Train/test split với stratify theo label
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    # Tên file output theo chuẩn:
    # x_train: {name}_train_cd40.csv
    # y_train: {name}_train_cd40_y.csv
    # x_test : {name}_test_cd40.csv
    # y_test : {name}_test_cd40_y.csv
    x_train_file = f"{name}_train_cd40.csv"
    y_train_file = f"{name}_train_cd40_y.csv"
    x_test_file  = f"{name}_test_cd40.csv"
    y_test_file  = f"{name}_test_cd40_y.csv"

    # Lưu X_train, X_test (Sequence)
    pd.DataFrame({SEQ_COL: X_train}).to_csv(x_train_file, index=False)
    pd.DataFrame({SEQ_COL: X_test}).to_csv(x_test_file, index=False)

    # Lưu y_train, y_test (Label)
    pd.DataFrame({LABEL_COL: y_train}).to_csv(y_train_file, index=False)
    pd.DataFrame({LABEL_COL: y_test}).to_csv(y_test_file, index=False)

    print(f"[INFO] Total samples: {len(df)}")
    print(f"[INFO] Train size: {len(X_train)} ({(len(X_train)/len(df))*100:.1f} %)")
    print(f"[INFO] Test  size: {len(X_test)} ({(len(X_test)/len(df))*100:.1f} %)")
    print(f"[INFO] Saved for {name}:")
    print(f"       x_train -> {x_train_file}")
    print(f"       y_train -> {y_train_file}")
    print(f"       x_test  -> {x_test_file}")
    print(f"       y_test  -> {y_test_file}")

    # Trả về để dùng tiếp trong code (x_train, x_test, y_train, y_test)
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    # Lưu splits vào dict nếu cần dùng trong cùng script
    splits = {}

    for name, infile in DATASETS:
        X_train, X_test, Y_train, Y_test = split_dataset(name, infile)

        # Truy cập sau này: splits["Liver"]["X_train"], ...
        splits[name] = {
            "X_train": X_train,
            "X_test": X_test,
            "y_train": Y_train,
            "y_test": Y_test,
        }

    print("\n[✓] Finished splitting all datasets.")
