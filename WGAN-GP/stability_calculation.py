import pandas as pd
import numpy as np
from Bio.SeqUtils.ProtParam import ProteinAnalysis

INPUT_CSV = "ACP_full_WGAN.csv"   # tập gốc
SEQ_COL = "Sequence"

MIN_LEN = 10
MAX_LEN = 15  # đổi 44 nếu bạn muốn xem toàn bộ

# --------- Hydrophobic ratio (nhất quán với bạn đang dùng) ----------
HYDROPHOBIC = set(list("AVILMFWYC"))

def clean_seq(seq: str) -> str:
    return str(seq).upper().replace(" ", "").replace("\t", "").strip()

def hydrophobic_ratio(seq: str) -> float:
    seq = clean_seq(seq)
    if len(seq) == 0:
        return np.nan
    return sum(aa in HYDROPHOBIC for aa in seq) / len(seq)

# --------- Amphipathicity: Hydrophobic moment (muH) for alpha-helix ----------
EISENBERG = {
    "A": 0.25, "C": 0.04, "D": -0.72, "E": -0.62, "F": 1.00,
    "G": 0.16, "H": -0.40, "I": 0.73, "K": -1.10, "L": 0.53,
    "M": 0.26, "N": -0.64, "P": -0.07, "Q": -0.69, "R": -1.76,
    "S": -0.26, "T": -0.18, "V": 0.54, "W": 0.37, "Y": 0.02
}

def hydrophobic_moment_helix(seq: str, angle_deg: float = 100.0) -> float:
    """
    Hydrophobic moment (muH) assuming an alpha-helix (default 100 deg/residue).
    muH = (1/n) * sqrt( (sum hi*cos(theta_i))^2 + (sum hi*sin(theta_i))^2 )
    where hi is hydrophobicity of residue i (Eisenberg scale).
    """
    seq = clean_seq(seq)
    if len(seq) == 0:
        return np.nan

    h = []
    for aa in seq:
        if aa not in EISENBERG:
            return np.nan
        h.append(EISENBERG[aa])

    h = np.array(h, dtype=float)
    theta = np.deg2rad(angle_deg) * np.arange(len(seq), dtype=float)

    x = np.sum(h * np.cos(theta))
    y = np.sum(h * np.sin(theta))
    mu = np.sqrt(x*x + y*y) / len(seq)
    return float(mu)

# --------- Boman index ----------
BOMAN_SCALE = {
    "L": 4.92, "I": 4.92, "V": 4.04, "F": 2.98, "M": 2.35, "W": 2.33,
    "A": 1.81, "C": 1.28, "G": 0.94, "Y": -0.14, "T": -2.57, "S": -3.40,
    "H": -4.66, "Q": -5.54, "K": -5.55, "N": -6.64, "E": -6.81, "D": -8.72,
    "R": -14.92
}

def boman_index(seq: str) -> float:
    """
    Boman (Potential Protein Interaction) index.
    Trả về NaN nếu có ký tự amino acid không nằm trong BOMAN_SCALE.
    """
    seq = clean_seq(seq)
    if len(seq) == 0:
        return np.nan
    vals = []
    for aa in seq:
        if aa not in BOMAN_SCALE:
            return np.nan
        vals.append(BOMAN_SCALE[aa])
    return float(-1.0 * (sum(vals) / len(vals)))

# --------- GRAVY ----------
def gravy_index(seq: str) -> float:
    """
    GRAVY (Grand Average of Hydropathy) theo Biopython/ProtParam (Kyte-Doolittle).
    Trả về NaN nếu sequence có ký tự không hợp lệ gây lỗi trong ProteinAnalysis.
    """
    seq = clean_seq(seq)
    if len(seq) == 0:
        return np.nan
    try:
        return float(ProteinAnalysis(seq).gravy())
    except Exception:
        return np.nan

# --------- Load & filter ----------
df = pd.read_csv(INPUT_CSV)
if SEQ_COL not in df.columns:
    raise ValueError(f"Input CSV must contain column '{SEQ_COL}'")

df[SEQ_COL] = df[SEQ_COL].apply(clean_seq)
df["length_check"] = df[SEQ_COL].str.len()

df = df[(df["length_check"] >= MIN_LEN) & (df["length_check"] <= MAX_LEN)].copy()

# --------- Compute features ----------
rows = []
for seq in df[SEQ_COL].tolist():
    if not seq:
        continue

    pa = ProteinAnalysis(seq)
    rows.append({
        "Sequence": seq,
        "length": len(seq),
        "molecular_weight": pa.molecular_weight(),
        "instability_index": pa.instability_index(),
        "net_charge_pH7.4": pa.charge_at_pH(7.4),
        "hydrophobic_ratio": hydrophobic_ratio(seq),
        "GRAVY": gravy_index(seq),
        "amphipathicity_muH_helix": hydrophobic_moment_helix(seq, angle_deg=100.0),
        "boman_index": boman_index(seq),
    })

feat = pd.DataFrame(rows)

# Ensure numeric (avoid accidental string dtype)
num_cols = [
    "molecular_weight",
    "instability_index",
    "net_charge_pH7.4",
    "hydrophobic_ratio",
    "GRAVY",
    "amphipathicity_muH_helix",
    "boman_index",
]
for c in num_cols:
    feat[c] = pd.to_numeric(feat[c], errors="coerce")

# Làm tròn 3 chữ số thập phân cho các cột số trong bảng feature
feat[num_cols] = feat[num_cols].round(3)

def summary_table(s: pd.Series) -> pd.Series:
    s = s.dropna()
    if len(s) == 0:
        return pd.Series({"min": np.nan, "p05": np.nan, "p25": np.nan, "median": np.nan,
                          "p75": np.nan, "p95": np.nan, "max": np.nan})
    return pd.Series({
        "min": s.min(),
        "p05": s.quantile(0.05),
        "p25": s.quantile(0.25),
        "median": s.quantile(0.50),
        "p75": s.quantile(0.75),
        "p95": s.quantile(0.95),
        "max": s.max(),
    })

summary = feat[num_cols].apply(summary_table)

# Làm tròn 3 chữ số thập phân cho bảng summary
summary = summary.round(3)

N = len(feat)
print("N =", N)
print(summary)

# ---------- Save features (3 số lẻ) ----------
feat.to_csv("ACP_full_WGAN_features_len10_15.csv", index=False)

# ---------- Save summary (3 số lẻ, kèm N) ----------
N_row = pd.DataFrame({"stat": ["N"], "value": [N]})
summary_reset = summary.reset_index().rename(columns={"index": "stat"})

summary_out = pd.concat([N_row, summary_reset], ignore_index=True)

# Làm tròn 3 số lẻ cho tất cả cột số trong summary_out (phòng khi có thêm numeric)
num_cols_summary = summary_out.select_dtypes(include=[np.number]).columns
summary_out[num_cols_summary] = summary_out[num_cols_summary].round(3)

summary_out.to_csv("ACP_full_WGAN_features_len10_15_summary.csv", index=False)
