import pandas as pd
import subprocess
import os
import tempfile
import random

# =============== CẤU HÌNH ===============
POS_FILE = "CancerPPD_Colon_cleaned_cdhit.csv"   # file 1: id, Sequence (positives)
NEG_POOL_FILE = "ACP_label_0.csv"              # file 2: Sequence, label (0/1)
NEG_LABEL_COL = "Label"                          # tên cột label trong file 2
SEQ_COL = "Sequence"

CDHIT2D_BIN = "cd-hit-2d"                        # binary cd-hit-2d trong PATH
CDHIT_IDENTITY = 0.40                            # ngưỡng 40%
CDHIT_WORDLEN = "2"                              # cho c = 0.4 dùng -n 2

OUT_CSV = "Colon_balanced_dataset_cd40.csv"      # file dataset cuối cùng
RANDOM_SEED = 42


# =============== HÀM TRỢ GIÚP ===============
def check_cdhit2d():
    try:
        subprocess.run([CDHIT2D_BIN, "-h"],
                       stdout=subprocess.PIPE,
                       stderr=subprocess.PIPE,
                       text=True)
        print(f"[✓] {CDHIT2D_BIN} available.")
    except FileNotFoundError:
        raise EnvironmentError(
            f"{CDHIT2D_BIN} is not installed or not found in PATH."
        )

def write_fasta_from_df(df, fasta_path, id_prefix):
    """Ghi df (cột Sequence) ra FASTA, prefix id để phân biệt P_ / N_."""
    with open(fasta_path, "w") as f:
        for i, seq in enumerate(df[SEQ_COL].astype(str), start=1):
            seq_u = seq.strip().upper().replace(" ", "")
            if not seq_u:
                continue
            seq_id = f"{id_prefix}{i}"
            f.write(f">{seq_id}\n{seq_u}\n")

def run_cdhit2d(db_fasta, query_fasta, out_prefix, c=0.4, n="2"):
    """Chạy cd-hit-2d để lọc query (negatives) không giống db (positives) >= c."""
    cmd = [
        CDHIT2D_BIN,
        "-i", db_fasta,     # database (positives)
        "-i2", query_fasta, # query (negatives pool)
        "-o", out_prefix,
        "-c", str(c),
        "-n", str(n),
        "-d", "0",
    ]
    print("[…] Running cd-hit-2d:", " ".join(cmd))
    res = subprocess.run(cmd, check=True,
                         stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE,
                         text=True)
    print("[✓] cd-hit-2d finished.")
    # Nếu muốn xem log:
    # print(res.stdout[:500])
    # print(res.stderr[:500])

def parse_fasta_ids_and_seqs(fasta_path):
    """Đọc FASTA, trả về list (id, seq)."""
    records = []
    if not os.path.exists(fasta_path):
        raise FileNotFoundError(f"FASTA '{fasta_path}' not found.")
    with open(fasta_path, "r") as f:
        lines = [line.strip() for line in f if line.strip()]
    for i in range(0, len(lines), 2):
        if lines[i].startswith(">") and i + 1 < len(lines):
            rid = lines[i][1:].strip()
            seq = lines[i + 1].strip()
            records.append((rid, seq))
    return records


# =============== MAIN ===============
if __name__ == "__main__":
    random.seed(RANDOM_SEED)

    # 1) Kiểm tra cd-hit-2d
    check_cdhit2d()

    # 2) Đọc positives
    pos_df = pd.read_csv(POS_FILE)
    if SEQ_COL not in pos_df.columns:
        raise ValueError(f"'{SEQ_COL}' not in {POS_FILE}")
    n_pos = len(pos_df)
    print(f"[INFO] Positives (label=1) from {POS_FILE}: N_pos = {n_pos}")
    pos_df["label"] = 1

    # 3) Đọc negatives pool (label=0)
    neg_pool = pd.read_csv(NEG_POOL_FILE)
    if SEQ_COL not in neg_pool.columns:
        raise ValueError(f"'{SEQ_COL}' not in {NEG_POOL_FILE}")
    if NEG_LABEL_COL not in neg_pool.columns:
        raise ValueError(f"'{NEG_LABEL_COL}' not in {NEG_POOL_FILE}")

    neg_pool = neg_pool[neg_pool[NEG_LABEL_COL] == 0].copy()
    neg_pool[SEQ_COL] = neg_pool[SEQ_COL].astype(str).str.upper().str.strip()
    neg_pool = neg_pool[neg_pool[SEQ_COL] != ""].drop_duplicates(subset=[SEQ_COL])

    print(f"[INFO] Negative pool label=0 (unique by seq): N_neg_pool = {len(neg_pool)}")

    if len(neg_pool) == 0:
        raise ValueError("No negative sequences (label=0) found in pool.")

    # 4) Chạy cd-hit-2d để loại negatives giống positives >= 40%
    with tempfile.TemporaryDirectory() as tmpdir:
        pos_fasta = os.path.join(tmpdir, "pos.fa")
        neg_fasta = os.path.join(tmpdir, "neg.fa")
        out_prefix = os.path.join(tmpdir, "pos_neg40")

        # Ghi FASTA
        write_fasta_from_df(pos_df, pos_fasta, id_prefix="P_")
        write_fasta_from_df(neg_pool, neg_fasta, id_prefix="N_")

        # Chạy cd-hit-2d
        run_cdhit2d(pos_fasta, neg_fasta, out_prefix,
                    c=CDHIT_IDENTITY, n=CDHIT_WORDLEN)

        # Output FASTA của cd-hit-2d (prefix; cd-hit-2d sẽ tạo file out_prefix)
        out_fasta = out_prefix

        # 5) Đọc lại FASTA output, lấy những id bắt đầu bằng "N_"
        recs = parse_fasta_ids_and_seqs(out_fasta)
        allowed_neg_seqs = set()
        for rid, seq in recs:
            if rid.startswith("N_"):
                allowed_neg_seqs.add(seq.strip().upper())

    print(f"[INFO] Negatives after cd-hit-2d filter (no >=40% identity to positives): "
          f"N = {len(allowed_neg_seqs)}")

    # 6) Lọc neg_pool theo allowed_neg_seqs
    neg_filtered = neg_pool[neg_pool[SEQ_COL].isin(allowed_neg_seqs)].copy()
    n_neg_filtered = len(neg_filtered)
    print(f"[INFO] Negatives kept after cd-hit-2d + mapping back to CSV: "
          f"N = {n_neg_filtered}")

    if n_neg_filtered < n_pos:
        raise ValueError(
            f"Not enough negatives after cd-hit-2d: need {n_pos}, "
            f"but only {n_neg_filtered} available."
        )

    # 7) Random sample negatives bằng đúng số lượng positives
    neg_sample = neg_filtered.sample(n=n_pos, random_state=RANDOM_SEED).copy()
    neg_sample["label"] = 0

    # 8) Tạo dataset cuối cùng
    #    + Giữ cột Sequence + label; nếu muốn giữ thêm id, bạn có thể chỉnh thêm.
    pos_final = pos_df[[SEQ_COL, "label"]].copy()
    neg_final = neg_sample[[SEQ_COL, "label"]].copy()

    final_df = pd.concat([pos_final, neg_final], ignore_index=True)
    final_df = final_df.sample(frac=1.0, random_state=RANDOM_SEED).reset_index(drop=True)

    final_df.to_csv(OUT_CSV, index=False)
    print(f"[✓] Final balanced dataset saved to: {OUT_CSV}")
    print(f"    Total samples: {len(final_df)} (pos={n_pos}, neg={n_pos})")
