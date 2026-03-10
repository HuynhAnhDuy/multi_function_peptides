import pandas as pd

# ---- ĐỔI TÊN FILE Ở ĐÂY ----
FILE1 = "/home/andy/andy/ACP/WGAN-GP/ACPs_generated_general_predictions_actives_3_tools_candidate.csv"   # có: sequence + các cột khác
FILE2 = "/home/andy/andy/ACP/WGAN-GP/result.csv"   # có: sequence + score
OUT   = "ACPs_generated_general_final_candidates.csv"

SEQ_COL   = "sequence"   # tên cột sequence ở cả 2 file
SCORE_COL = "score"      # tên cột score ở file 2

# (không bắt buộc) chuẩn hóa sequence cho chắc
def norm_seq(s: str) -> str:
    return str(s).upper().replace(" ", "").replace("\t", "").strip()

# ---- ĐỌC FILE ----
df1 = pd.read_csv(FILE1)
df2 = pd.read_csv(FILE2)

# Áp dụng chuẩn hóa sequence (nếu muốn nhất quán)
df1[SEQ_COL] = df1[SEQ_COL].apply(norm_seq)
df2[SEQ_COL] = df2[SEQ_COL].apply(norm_seq)

# ---- GIỮ CHỈ CỘT sequence + score TỪ FILE 2 ----
df2_small = df2[[SEQ_COL, SCORE_COL]]

# Nếu file 2 có nhiều dòng trùng sequence, bạn có thể:
# - lấy trung bình score, hoặc
# - lấy max/min,...
# Ví dụ lấy trung bình:
df2_small = df2_small.groupby(SEQ_COL, as_index=False)[SCORE_COL].mean()

# ---- JOIN: GẮN score VÀO FILE 1 ----
df_out = df1.merge(df2_small, on=SEQ_COL, how="left")

# ---- LƯU OUTPUT ----
df_out.to_csv(OUT, index=False)

print("Done, saved to:", OUT)
