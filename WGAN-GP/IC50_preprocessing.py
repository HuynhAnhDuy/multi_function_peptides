import pandas as pd
import numpy as np

# 1. Đọc dữ liệu
df = pd.read_csv('CancerPPD 2.0_ACPs_Lung.csv', encoding='latin1')

# 2. Chuẩn hoá cột Activity: strip + sửa lỗi encoding Âµ -> µ, loại bỏ Â còn sót
df['Activity_raw'] = (
    df['Activity']
    .astype(str)
    .str.strip()
    .str.replace('Âµ', 'µ', regex=False)  # "ÂµM" -> "µM", "Âµg/ml" -> "µg/ml"
    .str.replace('Â', '', regex=False)    # xoá mọi 'Â' còn lại nếu có
)

# 3. Trích số và đơn vị từ IC50/EC50
# pattern: tìm IC50 hoặc EC50, rồi (optional =) + (optional < hoặc >) + số + đơn vị (không có khoảng trắng)
pattern = r'(?:IC50|EC50)\s*=?\s*([<>]?)\s*([\d.,]+)\s*([^\s]+)?'

extract = df['Activity_raw'].str.extract(pattern, expand=True)
extract.columns = ['IC50_sign', 'IC50_value_str', 'IC50_unit']

df[['IC50_sign', 'IC50_value_str', 'IC50_unit']] = extract

# 4. Xác định những dòng thực sự có IC50/EC50 (tức là đã bắt được số)
mask_ic50 = df['IC50_value_str'].notna()
# (siết thêm điều kiện: chuỗi bắt đầu bằng IC50/EC50)
mask_start = df['Activity_raw'].str.match(r'^\s*(IC50|EC50)', case=False, na=False)
mask_ic50 = mask_ic50 & mask_start

df['Activity_is_IC50_or_EC50'] = mask_ic50

# 5. Chuyển chuỗi số sang float (CHỈNH LẠI Ở ĐÂY – dùng to_numeric thay vì astype(errors='coerce'))
df['IC50_value'] = pd.to_numeric(
    df['IC50_value_str'].str.replace(',', '.', regex=False),
    errors='coerce'
)

# 6. Hàm chuẩn hoá về µM (micromol/L)
def to_uM(value, unit):
    if pd.isna(value):
        return np.nan
    if unit is None or pd.isna(unit):
        # nếu không có đơn vị, tạm coi là µM
        return value

    u = str(unit).strip()
    # chuẩn hóa các biến thể micro: Âµ, μ → µ
    u = u.replace('Âµ', 'µ').replace('μ', 'µ').lower()

    # Nếu là nồng độ dạng khối lượng (µg/ml, mg/ml, g/ml) → không đổi sang µM (vì thiếu MW)
    if 'g/ml' in u:
        return np.nan

    # Map các dạng molarity
    if u in ['µm', 'um', 'microm', 'micromolar', 'µmol', 'umol', 'µmol/l', 'umol/l']:
        u_std = 'um'      # micromolar
    elif u in ['nm', 'nanom', 'nanomolar']:
        u_std = 'nm'
    elif u in ['mm', 'millim', 'millimolar']:
        u_std = 'mm'
    else:
        if 'mol/l' in u:
            if 'µ' in u or 'u' in u:
                u_std = 'um'
            elif 'n' in u:
                u_std = 'nm'
            elif 'm' in u:
                u_std = 'mm'
            else:
                u_std = 'um'
        else:
            # nếu không rõ, mặc định coi là µM
            u_std = 'um'

    # Quy về µM
    if u_std == 'um':       # µM
        return value
    elif u_std == 'nm':     # nM
        return value / 1000.0
    elif u_std == 'mm':     # mM
        return value * 1000.0
    else:
        return value

df['IC50_uM'] = df.apply(lambda row: to_uM(row['IC50_value'], row['IC50_unit']), axis=1)

# 7. Ghi chú:
#   - KHÔNG phải IC50/EC50 -> "False"
#   - Là IC50/EC50 & IC50_uM < 20 µM -> "keep"
#   - Còn lại (>=20 µM hoặc NaN) -> "exclude"
df['IC50_note'] = np.where(
    ~mask_ic50,
    'False',
    np.where(df['IC50_uM'].notna() & (df['IC50_uM'] < 20), 'keep', 'exclude')
)

# 8. Lọc ra chỉ những dòng IC50/EC50 có IC50_uM < 20 µM
df_good = df[mask_ic50 & df['IC50_uM'].notna() & (df['IC50_uM'] < 20)].copy()

# 9. In kiểm tra nhanh (rất nên xem kết quả trên terminal)
print("Total rows:", len(df))
print("Rows with IC50/EC50 pattern:", mask_ic50.sum())
print("Rows with numeric IC50_uM:", df['IC50_uM'].notna().sum())
print("Rows with IC50_uM < 20 µM:", df_good.shape[0])

# 10. Lưu lại ra CSV
df.to_csv('CancerPPD_2.0_Lung_with_IC50_EC50_cleaned.csv', index=False)
df_good.to_csv('CancerPPD_2.0_Lung_IC50_EC50_lt_20uM_only.csv', index=False)
