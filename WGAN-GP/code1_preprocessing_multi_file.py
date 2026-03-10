import pandas as pd
import numpy as np

SEQ_COL = "sequence"   # tên cột chứa peptide sequence trong các file
VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")  # 20 amino acid chuẩn

# ===== KHAI BÁO CÁC FILE CẦN XỬ LÝ =====
# Nếu file nằm cùng thư mục script: chỉ cần để tên file như bên dưới.
# Nếu nằm chỗ khác: thay bằng full path, ví dụ "/home/andy/xxx/CancerPPD 2.0_Colon.csv"
FILES = [
    ("CancerPPD 2.0_Colon.csv", "CancerPPD_Colon_cleaned.csv"),
    ("CancerPPD 2.0_Lung.csv",  "CancerPPD_Lung_cleaned.csv"),
    ("CancerPPD 2.0_Liver.csv", "CancerPPD_Liver_cleaned.csv"),
]

def norm_seq(seq: str) -> str:
    return str(seq).upper().replace(" ", "").replace("\t", "").strip()

def has_invalid_aa(seq: str) -> bool:
    return any(ch not in VALID_AA for ch in seq)

def process_file(infile: str, outfile: str, seq_col: str = SEQ_COL):
    print(f"\n===== XỬ LÝ FILE: {infile} =====")
    df = pd.read_csv(infile, encoding="latin1")
    print(f"[INFO] Tổng số dòng ban đầu: {len(df)}")

    if seq_col not in df.columns:
        raise ValueError(f"Không tìm thấy cột '{seq_col}' trong file {infile}. Hãy chỉnh SEQ_COL cho đúng.")

    # 1. Chuẩn hoá chuỗi: UPPERCASE + strip
    df["seq_raw"] = (
        df[seq_col]
        .astype(str)
        .str.upper()
        .str.strip()
    )

    # 2. Chỉ giữ ký tự A–Z
    df["seq_clean"] = df["seq_raw"].str.replace(r"[^A-Z]", "", regex=True)

    # 3. Bỏ chuỗi rỗng sau khi làm sạch
    before_empty_filter = len(df)
    df = df[df["seq_clean"].str.len() > 0].copy()
    print(f"[INFO] Số dòng bị loại vì seq rỗng sau làm sạch: {before_empty_filter - len(df)}")
    print(f"[INFO] Số dòng còn lại sau bước làm sạch ký tự: {len(df)}")

    # 4. Lọc chỉ giữ 20 amino acid chuẩn
    df["has_invalid_aa"] = df["seq_clean"].apply(has_invalid_aa)
    invalid_count = df["has_invalid_aa"].sum()
    print(f"[INFO] Số dòng có amino acid lạ (không thuộc 20 aa chuẩn): {invalid_count}")

    df = df[~df["has_invalid_aa"]].copy()
    df.drop(columns=["has_invalid_aa"], inplace=True)
    print(f"[INFO] Số dòng còn lại sau khi bỏ amino acid lạ: {len(df)}")

    # 5. Loại trùng lặp 100% theo seq_clean
    before_dedup = len(df)
    df = df.drop_duplicates(subset=["seq_clean"]).reset_index(drop=True)
    print(f"[INFO] Số dòng trùng lặp bị loại (100% giống nhau): {before_dedup - len(df)}")
    print(f"[INFO] Số dòng còn lại sau khi bỏ trùng: {len(df)}")

    # 6. Tạo cột id và Sequence để dùng với CD-HIT
    df["id"] = [f"pep_{i+1}" for i in range(len(df))]
    df["Sequence"] = df["seq_clean"]

    # 7. Lưu file chỉ còn 2 cột id, Sequence
    df[["id", "Sequence"]].to_csv(outfile, index=False)
    print(f"[✓] Đã lưu file cho CD-HIT: {outfile}, số chuỗi cuối cùng = {len(df)}")

# ===== CHẠY CHO TẤT CẢ FILE =====
for infile, outfile in FILES:
    process_file(infile, outfile)
