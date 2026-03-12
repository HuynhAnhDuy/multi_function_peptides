import pandas as pd
import subprocess
import os
import time

# ===== CẤU HÌNH DANH SÁCH FILE CẦN CHẠY CD-HIT =====
# Mỗi phần tử: (csv_input, fasta_tmp, fasta_cdhit, csv_output)
JOBS = [
    (
        "CancerPPD_Colon_cleaned.csv",
        "CancerPPD_Colon_cleaned.fasta",
        "CancerPPD_Colon_cdhit.fasta",
        "CancerPPD_Colon_cleaned_cdhit.csv",
    ),
    (
        "CancerPPD_Lung_cleaned.csv",
        "CancerPPD_Lung_cleaned.fasta",
        "CancerPPD_Lung_cdhit.fasta",
        "CancerPPD_Lung_cleaned_cdhit.csv",
    ),
    (
        "CancerPPD_Liver_cleaned.csv",
        "CancerPPD_Liver_cleaned.fasta",
        "CancerPPD_Liver_cdhit.fasta",
        "CancerPPD_Liver_cleaned_cdhit.csv",
    ),
]

CDHIT_THRESHOLD = 0.9   


# ===== STEP 1: CSV -> FASTA =====
def csv_to_fasta(csv_file, fasta_file):
    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"Input file '{csv_file}' does not exist.")
    
    df = pd.read_csv(csv_file)

    # Kiểm tra cột cần thiết
    required_columns = {"id", "Sequence"}
    if not required_columns.issubset(df.columns):
        raise ValueError(f"CSV file must contain columns: {required_columns}")

    # Kiểm tra trùng lặp ID
    if df["id"].duplicated().any():
        raise ValueError("Duplicate sequence IDs detected.")

    # Ghi file FASTA
    with open(fasta_file, "w") as f:
        for _, row in df.iterrows():
            f.write(f">{row['id']}\n{row['Sequence']}\n")
    
    n_seq = len(df)
    print(f"[✓] FASTA file '{fasta_file}' created successfully. N = {n_seq}")
    return n_seq  # số sequence ban đầu


# ===== STEP 2: CHECK CD-HIT INSTALLATION =====
def check_cdhit():
    try:
        result = subprocess.run(
            ["cd-hit", "-h"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        # returncode khác 0 vẫn OK vì -h chỉ in help
        print("[✓] CD-HIT is installed and ready to use.")
    except FileNotFoundError:
        raise EnvironmentError(
            "CD-HIT is not installed or not found in your system PATH."
        )


# ===== STEP 3: RUN CD-HIT =====
def run_cdhit(input_fasta, output_fasta, threshold=0.9):
    print(f"[...] Running CD-HIT on '{input_fasta}' (c = {threshold:.2f}, {threshold*100:.1f}% identity) ...")
    start_time = time.time()

    cmd = [
        "cd-hit",
        "-i", input_fasta,
        "-o", output_fasta,
        "-c", str(threshold),
        "-n", "5",
        "-d", "0",
    ]
    result = subprocess.run(
        cmd,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    print(f"[✓] CD-HIT finished. Output: '{output_fasta}'")
    print(f"[⏱] Time taken: {time.time() - start_time:.2f} seconds")

    # In rút gọn stdout/stderr nếu cần
    if result.stdout:
        print("[CD-HIT stdout] (first 500 chars):")
        print(result.stdout[:500])
    if result.stderr:
        print("[CD-HIT stderr] (first 500 chars):")
        print(result.stderr[:500])


# ===== STEP 4: FASTA (CD-HIT) -> CSV =====
def fasta_to_csv(fasta_file, output_csv):
    sequences = []
    with open(fasta_file, "r") as f:
        lines = [line.strip() for line in f if line.strip()]

    for i in range(0, len(lines), 2):
        if lines[i].startswith(">") and i + 1 < len(lines):
            seq_id = lines[i][1:].strip()
            sequence = lines[i + 1].strip()
            sequences.append({"id": seq_id, "Sequence": sequence})

    df_out = pd.DataFrame(sequences)
    df_out.to_csv(output_csv, index=False)
    n_seq = len(df_out)
    print(f"[✓] Filtered CSV saved to '{output_csv}'. N = {n_seq}")
    return n_seq  # số sequence sau CD-HIT


# ===== MAIN WORKFLOW =====
if __name__ == "__main__":
    try:
        check_cdhit()

        for csv_file, fasta_file, cdhit_output, filtered_csv in JOBS:
            print("\n======================================")
            print(f"Processing dataset: {csv_file}")
            print("======================================")

            # Số lượng trước CD-HIT
            n_before = csv_to_fasta(csv_file, fasta_file)

            # Chạy CD-HIT
            run_cdhit(fasta_file, cdhit_output, threshold=CDHIT_THRESHOLD)

            # Số lượng sau CD-HIT
            n_after = fasta_to_csv(cdhit_output, filtered_csv)

            # Thống kê thay đổi
            removed = n_before - n_after
            pct_kept = (n_after / n_before * 100) if n_before else 0.0
            pct_removed = 100.0 - pct_kept

            print(f"[STATS] {csv_file}")
            print(f"        CD-HIT threshold: c = {CDHIT_THRESHOLD:.2f} ({CDHIT_THRESHOLD*100:.1f}% identity)")
            print(f"        Sequences before CD-HIT: {n_before}")
            print(f"        Sequences after  CD-HIT: {n_after}")
            print(f"        Removed by CD-HIT:       {removed} ({pct_removed:.1f}% )")
            print(f"        Retained after CD-HIT:   {pct_kept:.1f}%")

        print("\n[✓] All datasets processed successfully.")

    except Exception as e:
        print(f"[❌] Error: {e}")
