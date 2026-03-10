import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ====== CONFIG ======
INPUT_CSV = '/home/andy/andy/ACP/WGAN-GP/ACP_full_cleaned_cdhit.csv'
OUTPUT_CSV = 'ACP_full_WGAN.csv'
HISTOGRAM_FILE = 'Histogram_ACP_full.svg'

L_MIN = 10
L_MAX_HARD = 60
EOS_TOKEN = '<EOS>'
PAD_TOKEN = '<PAD>'

# ====== 1. Load Data ======
df = pd.read_csv(INPUT_CSV)

if 'Sequence' not in df.columns:
    raise ValueError("Không tìm thấy cột 'Sequence' trong file.")

df['seq_clean'] = df['Sequence'].astype(str).str.upper().str.strip()

# ====== 2. Tính độ dài chuỗi ======
df['length'] = df['seq_clean'].str.len()
N_total = len(df)

print("=== Thống kê độ dài chuỗi ===")
print(df['length'].describe())
print(f"Tổng số chuỗi: {N_total}")

# Phân bố theo khoảng
n_lt_10  = (df['length'] < 10).sum()
n_10_15  = df['length'].between(10, 15).sum()
n_16_60  = df['length'].between(16, 60).sum()
n_gt_60  = (df['length'] > 60).sum()

print("\n=== Phân bố theo khoảng độ dài ===")
print(f"< 10 aa         : {n_lt_10:3d} sequences ({n_lt_10 / N_total * 100:5.1f}%)")
print(f"10–15 aa        : {n_10_15:3d} sequences ({n_10_15 / N_total * 100:5.1f}%)")
print(f"16–60 aa        : {n_16_60:3d} sequences ({n_16_60 / N_total * 100:5.1f}%)")
print(f"> 60 aa         : {n_gt_60:3d} sequences ({n_gt_60 / N_total * 100:5.1f}%)")

assert n_lt_10 + n_10_15 + n_16_60 + n_gt_60 == N_total

# Đặc trưng thống kê
mean_len = df['length'].mean()
median_len = df['length'].median()
q95 = df['length'].quantile(0.95)
N_q95 = (df['length'] <= q95).sum()

print("\n=== Thống kê đặc trưng ===")
print(f"Mean length    : {mean_len:.2f} aa")
print(f"Median length  : {median_len:.2f} aa")
print(f"95th percentile: {q95:.2f} aa → {N_q95} sequences ({N_q95 / N_total * 100:5.1f}%)")

# ====== 3. Vẽ histogram độ dài ======
lengths = df['length'].values
bins = np.arange(lengths.min(), lengths.max() + 2)

plt.figure(figsize=(6, 4))
plt.hist(lengths, bins=bins, color="#179C4C", edgecolor='black', linewidth=0.5, alpha=0.8)

ax = plt.gca()
ymax = ax.get_ylim()[1]
ax.axvline(10, linestyle=':', linewidth=1.5, color="#2025C1")
ax.axvline(60, linestyle=':', linewidth=1.5, color="#C22626")

ax.text(10, ymax * 0.95, "10 aa", rotation=90, va='top', ha='right', fontsize=10)
ax.text(60, ymax * 0.95, "60 aa", rotation=90, va='top', ha='right', fontsize=10)

plt.xlabel('Peptide length (aa)', fontweight='bold', fontstyle='italic', fontsize=12)
plt.ylabel('Count', fontweight='bold', fontstyle='italic', fontsize=12)

textstr = (
    f"Mean length: {mean_len:.1f} aa\n"
    f"Median length: {median_len:.1f} aa\n"
    f"< 10 aa: {n_lt_10} ({n_lt_10 / N_total * 100:4.1f}%)\n"
    f"10–15 aa: {n_10_15} ({n_10_15 / N_total * 100:4.1f}%)\n"
    f"16–60 aa: {n_16_60} ({n_16_60 / N_total * 100:4.1f}%)\n"
    f"> 60 aa: {n_gt_60} ({n_gt_60 / N_total * 100:4.1f}%)"
)

plt.gcf().text(0.65, 0.7, textstr, fontsize=10,
    bbox=dict(boxstyle='round', facecolor='none', edgecolor='black', linewidth=0.8))

plt.tight_layout()
plt.savefig(HISTOGRAM_FILE, format='svg')
print(f"[✓] Đã lưu biểu đồ histogram: {HISTOGRAM_FILE}")

# ====== 4. Độ dài phổ biến nhất ======
length_counts = df['length'].value_counts().sort_index()
most_common_len = length_counts.idxmax()
most_common_count = length_counts.max()

print("\n=== Độ dài phổ biến nhất ===")
print(f"Độ dài phổ biến nhất: {most_common_len} aa ({most_common_count} chuỗi)")

# ====== 5. Thiết lập L_MAX và lọc chuỗi cho WGAN-GP ======
L_MAX = int(min(q95, L_MAX_HARD))
print(f"\nĐề xuất L_MAX = {L_MAX}")

df_filtered = df[(df['length'] >= L_MIN) & (df['length'] <= L_MAX)].copy()
df_filtered.reset_index(drop=True, inplace=True)
print(f"Số chuỗi dùng cho WGAN-GP [{L_MIN}, {L_MAX}]: {len(df_filtered)}")

# ====== 6. Thêm EOS + PAD ======
def add_eos_and_pad(seq: str, max_len: int) -> str:
    tokens = list(seq) + [EOS_TOKEN]
    if len(tokens) > max_len:
        tokens = tokens[:max_len]
    while len(tokens) < max_len:
        tokens.append(PAD_TOKEN)
    return ' '.join(tokens)

df_filtered['seq_tokens'] = df_filtered['seq_clean'].apply(lambda s: add_eos_and_pad(s, L_MAX))
df_filtered['seq_token_list'] = df_filtered['seq_tokens'].str.split(' ')

# ====== 7. Lưu output ======
df_filtered.to_csv(OUTPUT_CSV, index=False)
print(f"[✓] Đã lưu dữ liệu huấn luyện WGAN-GP: {OUTPUT_CSV}")
print(f"L_MAX sử dụng = {L_MAX}")
