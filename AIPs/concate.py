import pandas as pd

# Cấu hình chung cho Liver
PREFIX = "AIP"
SPLITS = ["x_train", "x_test"]  # thực hiện cho cả train và test

for split in SPLITS:
    print(f"\n===== Processing {PREFIX}_{split} =====")

    # File input
    onehot_csv = f"{PREFIX}_{split}_onehot_candidate.csv"
    esm_csv    = f"{PREFIX}_{split}_esm_candidate.csv"

    # Load file one-hot (phải có cột Sequence)
    df1 = pd.read_csv(onehot_csv)

    # Load file ESM (bỏ cột Sequence nếu có)
    df2 = pd.read_csv(esm_csv)
    if "Sequence" in df2.columns:
        df2 = df2.drop(columns=["Sequence"])

    # Kiểm tra số dòng phải khớp
    if df1.shape[0] != df2.shape[0]:
        raise ValueError(
            f"Số dòng không khớp giữa {onehot_csv} ({df1.shape[0]}) "
            f"và {esm_csv} ({df2.shape[0]}). Kiểm tra lại!"
        )

    # Gộp 2 file theo chiều ngang (axis=1)
    result = pd.concat([df1, df2], axis=1)

    # Đảm bảo cột 'Sequence' nằm ở vị trí đầu tiên
    if "Sequence" in result.columns:
        cols = ["Sequence"] + [c for c in result.columns if c != "Sequence"]
        result = result[cols]
    else:
        raise ValueError(f"⚠️ File {onehot_csv} không có cột 'Sequence' — kiểm tra lại input!")

    # Lưu file đầu ra
    output_path = f"{PREFIX}_{split}_onehot_esm_candidate.csv"
    result.to_csv(output_path, index=False)

    print(f"✅ File đã lưu: {output_path}")
    print(f"🔹 {split} - số dòng: {result.shape[0]}")
    print(f"🔹 {split} - số cột: {result.shape[1]}")
    print(f"🔹 {split} - các cột đầu tiên: {result.columns[:10].tolist()}")
