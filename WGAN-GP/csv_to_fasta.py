import pandas as pd

INPUT_CSV = "ACP_candidate_selected_7_sequences_for_synthesis.csv"   # đổi tên file của bạn
OUTPUT_FASTA = "ACP_candidate_selected_7_sequences_for_synthesis.fasta"

ID_COL = "id"
SEQ_COL = "sequence"

df = pd.read_csv(INPUT_CSV)

if ID_COL not in df.columns or SEQ_COL not in df.columns:
    raise ValueError(f"CSV phải có cột '{ID_COL}' và '{SEQ_COL}'")

# clean nhẹ
df[ID_COL] = df[ID_COL].astype(str).str.strip()
df[SEQ_COL] = df[SEQ_COL].astype(str).str.upper().str.replace(" ", "", regex=False).str.strip()

# bỏ dòng rỗng / NaN
df = df[(df[ID_COL] != "") & (df[SEQ_COL] != "")]

with open(OUTPUT_FASTA, "w", encoding="utf-8") as f:
    for _, row in df.iterrows():
        f.write(f">{row[ID_COL]}\n{row[SEQ_COL]}\n")

print(f"Saved FASTA: {OUTPUT_FASTA} | n={len(df)}")
