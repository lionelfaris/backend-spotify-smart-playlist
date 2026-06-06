import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from app.core.ml_loader import df_metadata, combined_sparse

def search_tracks(q: str):
    # Pencarian string nama lagu atau musisi pada metadata
    mask_name = df_metadata["track_name"].str.contains(q, case=False, na=False)
    mask_artist = df_metadata["artists"].str.contains(q, case=False, na=False)
    
    results = df_metadata[mask_name | mask_artist].head(15)
    
    tracks = []
    for _, row in results.iterrows():
        tracks.append({
            "id": row["track_id"],
            "name": row["track_name"],
            "artist": row["artists"]
        })
    return tracks

def get_recommendations(req):
    # 1. Cari indeks posisi baris lagu berdasarkan track_id
    idx_list = df_metadata[df_metadata["track_id"] == req.track_id].index
    if len(idx_list) == 0:
        raise IndexError("Track tidak ditemukan")
        
    seed_idx = idx_list[0]
    seed_vector = combined_sparse[seed_idx]
    
    # 2. Hitung Cosine Similarity antar Sparse Matrix (Sesuai rumus Cell 9 Notebook)
    sims = cosine_similarity(seed_vector, combined_sparse)[0]
    top_indices_all = sims.argsort()[::-1]
    
    seed_song = df_metadata.iloc[seed_idx]['track_name']
    
    recommendations = []
    recommendations_count = 0
    
    # 3. Looping mencari tetangga terdekat
    for idx in top_indices_all:
        if idx == seed_idx:
            continue
            
        nama_lagu = str(df_metadata.iloc[idx]['track_name'])
        artis = str(df_metadata.iloc[idx]['artists'])
        track_id = str(df_metadata.iloc[idx]['track_id'])
        
        # Filter proteksi anti-duplikasi judul mirip dari notebook
        if seed_song.lower() in nama_lagu.lower() or nama_lagu.lower() in seed_song.lower():
            continue
            
        match_percentage = sims[idx] * 100
        recommendations_count += 1
        
        recommendations.append({
            "track_id": track_id,
            "track_name": nama_lagu,
            "artists": artis,
            "similarity_dist_percent": round(float(match_percentage), 2),
            "rank": recommendations_count
        })
        
        if recommendations_count == req.n:
            break
            
    return recommendations