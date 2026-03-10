import pandas as pd
import numpy as np
from modlamp.descriptors import GlobalDescriptor
from peptides import Peptide
from Bio.SeqUtils.ProtParam import ProteinAnalysis

HYDROPHOBIC_AA = set("AVILMFYW")  # amino acid kỵ nước

def norm_seq(seq: str) -> str:
    return str(seq).upper().replace(" ", "").replace("\t", "").strip()

def calc_hydrophobic_ratio(seq: str) -> float:
    seq_u = norm_seq(seq)
    if len(seq_u) == 0:
        return float("nan")
    n_hydro = sum(seq_u.count(aa) for aa in HYDROPHOBIC_AA)
    return n_hydro / len(seq_u)  # 0–1

# Amphipathicity: hydrophobic moment (muH) for alpha-helix (100°/residue)
EISENBERG = {
    "A": 0.25, "C": 0.04, "D": -0.72, "E": -0.62, "F": 1.00,
    "G": 0.16, "H": -0.40, "I": 0.73, "K": -1.10, "L": 0.53,
    "M": 0.26, "N": -0.64, "P": -0.07, "Q": -0.69, "R": -1.76,
    "S": -0.26, "T": -0.18, "V": 0.54, "W": 0.37, "Y": 0.02
}

def calc_muH_helix(seq: str, angle_deg: float = 100.0) -> float:
    seq_u = norm_seq(seq)
    if len(seq_u) == 0:
        return float("nan")
    h = []
    for aa in seq_u:
        if aa not in EISENBERG:
            return float("nan")
        h.append(EISENBERG[aa])
    h = np.array(h, dtype=float)
    theta = np.deg2rad(angle_deg) * np.arange(len(seq_u), dtype=float)
    x = np.sum(h * np.cos(theta))
    y = np.sum(h * np.sin(theta))
    return float(np.sqrt(x * x + y * y) / len(seq_u))

def calc_gravy(seq: str) -> float:
    """GRAVY theo Biopython (Kyte–Doolittle)."""
    seq_u = norm_seq(seq)
    if len(seq_u) == 0:
        return float("nan")
    try:
        pa = ProteinAnalysis(seq_u)
        return float(pa.gravy())
    except Exception:
        return float("nan")

# ---------- Đọc dữ liệu ----------
INFILE = "ACPs_generated_general_predictions_actives_3_tools.csv"
df = pd.read_csv(INFILE)

if "Sequence" not in df.columns:
    raise ValueError("Input CSV must contain column 'Sequence'")

results = []
for seq in df["Sequence"]:
    seq = norm_seq(seq)
    if not seq:
        continue

    try:
        # modlamp descriptors
        desc = GlobalDescriptor([seq])
        desc.calculate_all()

        # NOTE: indices phụ thuộc version modlamp
        boman       = float(desc.descriptor[0][3])
        instability = float(desc.descriptor[0][5])

        # peptides package
        pep    = Peptide(seq)
        mw     = float(pep.molecular_weight())
        charge = float(pep.charge(pH=7.4))

        # GRAVY dùng Biopython, không dùng pep.gravy() nữa
        gravy  = calc_gravy(seq)

        hydro_r = float(calc_hydrophobic_ratio(seq))
        muH     = float(calc_muH_helix(seq))

        results.append({
            "sequence": seq,
            "molecular_weight": mw,
            "net_charge_pH7.4": charge,
            "instability_index": instability,
            "boman_index": boman,
            "hydrophobic_ratio": hydro_r,
            "GRAVY": gravy,
            "amphipathicity_muH_helix": muH,
        })

    except Exception as e:
        print(f"Error with sequence {seq}: {e}")

df_props = pd.DataFrame(results)
print(f"✅ Number of peptides with computed properties: {len(df_props)}")
print(df_props.columns)

# ---------- Tiêu chí (ưu tiên tan nước) ----------
criteria = {
    "molecular_weight":            lambda x: x <= 2300,
    "instability_index":           lambda x: x <= 50,
    "net_charge_pH7.4":            lambda x: 2.0 <= x <= 5.5,
    "hydrophobic_ratio":           lambda x: 0.50 <= x <= 0.64,
    "GRAVY":                       lambda x: -0.4 <= x <= 1.1,
    "amphipathicity_muH_helix":    lambda x: x >= 0.33,
    "boman_index":                 lambda x: -0.6 <= x <= 2.0,
}

# ---------- Ép kiểu numeric để tránh so sánh sai ----------
for col in criteria.keys():
    if col not in df_props.columns:
        raise ValueError(f"Missing column '{col}' in df_props. Available: {list(df_props.columns)}")
    df_props[col] = pd.to_numeric(df_props[col], errors="coerce")

# ---------- Đánh giá tiêu chí (pass theo từng cột) ----------
check_cols = []
for col, rule in criteria.items():
    flag = f"pass_{col}"
    df_props[flag] = df_props[col].apply(lambda v: bool(rule(v)) if pd.notna(v) else False)
    check_cols.append(flag)

# ---------- Tổng số tiêu chí đạt ----------
df_props["n_pass"] = df_props[check_cols].sum(axis=1)

# Candidate: pass ít nhất (len(criteria) - 1) tiêu chí
df_props["candidate"] = df_props["n_pass"] >= (len(criteria) - 1)

total  = len(df_props)
n_cand = int(df_props["candidate"].sum())
pct    = (n_cand / total * 100) if total else 0.0

print(f"🎯 {n_cand}/{total} peptides ({pct:.1f} %) pass ≥ {len(criteria)-1}/{len(criteria)} criteria")

# ---------- Debug: hydrophobic_ratio > 0.64 nhưng pass_hydrophobic_ratio == True ----------
bad = df_props[(df_props["hydrophobic_ratio"] > 0.64) & (df_props["pass_hydrophobic_ratio"] == True)]
if len(bad) > 0:
    print("⚠️ Found rows where hydrophobic_ratio > 0.64 but pass_hydrophobic_ratio == True (should not happen). Showing first 10:")
    print(bad[["sequence", "hydrophobic_ratio", "pass_hydrophobic_ratio"]].head(10).to_string(index=False))
else:
    print("✅ No contradictions for hydrophobic_ratio rule.")

# ---------- Xuất kết quả ----------
OUT_ALL = "ACPs_generated_general_predictions_actives_3_tools_druglike.csv"
df_props.to_csv(OUT_ALL, index=False)

print(f"• Full report: {OUT_ALL}")
