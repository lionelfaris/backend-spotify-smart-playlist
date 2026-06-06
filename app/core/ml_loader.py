import pandas as pd
from scipy.sparse import load_npz

print("=== MEMULAI LOADING DATA OPTIMAL (V4) ===")

# 1. Memuat metadata super ringan (Hanya memuat id, nama lagu, dan artis)
df_metadata = pd.read_csv("data/track_order.csv")
df_metadata = df_metadata.reset_index(drop=True)

# 2. Memuat Sparse Matrix (.npz) hasil gabungan Audio + Genre
# Ini menggantikan file .npy dan proses FAISS yang boros RAM
combined_sparse = load_npz("data/feature_matrix_combined.npz")

print(f"Combined Sparse Matrix Shape: {combined_sparse.shape}")
print(f"Total Master Lagu Terdaftar : {len(df_metadata):,}")
print("=== SERVER SIAP (RAM KONDISI SUPER IRIT) ===")