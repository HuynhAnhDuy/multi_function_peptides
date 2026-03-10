import pandas as pd

# ================== CẤU HÌNH FILE ==================
MAIN_FILE   = "ACPs_generated_general_actives_full_screening.csv"

LIVER_FILE  = "Liver_candidate_probability.csv"
LUNG_FILE   = "Lung_candidate_probability.csv"
COLON_FILE  = "Colon_candidate_probability.csv"

OUTPUT_FILE = "ACPs_generated_screen_active_all_endpoint.csv"


# ================== HÀM TRỢ GIÚP ==================
def find_seq_col(df, name_hint=""):
    """
    Tìm cột sequence trong DataFrame.
    Ưu tiên: 'sequence', 'Sequence'. Nếu không có thì báo lỗi rõ ràng.
    """
    if "sequence" in df.columns:
        return "sequence"
    if "Sequence" in df.columns:
        return "Sequence"
    raise ValueError(f"Không tìm thấy cột 'sequence' hoặc 'Sequence' trong file {name_hint}")


def normalize_sequence_series(s):
    """
    Chuẩn hóa chuỗi peptide để join chắc chắn:
    - ép string
    - upper
    - bỏ khoảng trắng (space, tab)
    """
    return (
        s.astype(str)
         .str.upper()
         .str.replace(r"\s+", "", regex=True)
         .str.strip()
    )


# ================== MAIN ==================
def main():
    # ----- 1. Đọc file chính (A) -----
    df_main = pd.read_csv(MAIN_FILE)
    seq_col_main = find_seq_col(df_main, MAIN_FILE)

    # Tạo key chuẩn hóa để join
    df_main["sequence_key"] = normalize_sequence_series(df_main[seq_col_main])

    print(f"[INFO] Main file: {MAIN_FILE}")
    print(f"       Rows: {len(df_main)}, seq col: {seq_col_main}")

    # ----- 2. Hàm đọc & chuẩn hóa từng file candidate -----
    def load_candidate(file_path, endpoint_name):
        df = pd.read_csv(file_path)
        seq_col = find_seq_col(df, file_path)

        # Chuẩn hóa sequence để join
        df["sequence_key"] = normalize_sequence_series(df[seq_col])

        # Kiểm tra cột predicted_label
        if "predicted_label" not in df.columns:
            raise ValueError(
                f"File {file_path} (endpoint {endpoint_name}) không có cột 'predicted_label'"
            )

        # Chỉ giữ key + predicted_label để merge
        df_small = df[["sequence_key", "predicted_label"]].copy()
        new_col_name = f"{endpoint_name}_predicted_label"
        df_small = df_small.rename(columns={"predicted_label": new_col_name})

        print(f"[INFO] Loaded {endpoint_name} candidate file: {file_path}")
        print(f"       Rows: {len(df)}, unique sequences: {df_small['sequence_key'].nunique()}")
        return df_small

    # ----- 3. Đọc 3 file endpoint -----
    df_liver = load_candidate(LIVER_FILE, "Liver")
    df_lung  = load_candidate(LUNG_FILE,  "Lung")
    df_colon = load_candidate(COLON_FILE, "Colon")

    # ----- 4. Merge lần lượt vào df_main (LEFT JOIN) -----
    df_merged = df_main.merge(df_liver, on="sequence_key", how="left")
    df_merged = df_merged.merge(df_lung,  on="sequence_key", how="left")
    df_merged = df_merged.merge(df_colon, on="sequence_key", how="left")

    # Nếu muốn, có thể fillna cho predicted_label (vd: 0 hoặc -1). Ở đây giữ NaN là "không có dự đoán".
    # df_merged["Liver_predicted_label"] = df_merged["Liver_predicted_label"].fillna(-1).astype(int)
    # df_merged["Lung_predicted_label"]  = df_merged["Lung_predicted_label"].fillna(-1).astype(int)
    # df_merged["Colon_predicted_label"] = df_merged["Colon_predicted_label"].fillna(-1).astype(int)

    # Bỏ cột sequence_key trung gian
    df_merged = df_merged.drop(columns=["sequence_key"])

    # ----- 5. Ghi file output -----
    df_merged.to_csv(OUTPUT_FILE, index=False)
    print(f"[✓] Saved merged file to: {OUTPUT_FILE}")
    print(f"    Final rows: {len(df_merged)}")
    print(f"    Columns: {df_merged.columns.tolist()}")


if __name__ == "__main__":
    main()
