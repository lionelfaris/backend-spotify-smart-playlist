import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from app.core.ml_loader import df_metadata, combined_sparse

def search_tracks(q: str):
    mask_name = df_metadata["track_name"].str.contains(q, case=False, na=False)
    mask_artist = df_metadata["artists"].str.contains(q, case=False, na=False)
    
    results = df_metadata[mask_name | mask_artist].head(15)
    
    tracks = []
    for _, row in results.iterrows():
        tracks.append({
            "id": str(row["track_id"]),
            "name": str(row["track_name"]),
            "artist": str(row["artists"]),
            "genre": str(row["genre"]) if pd.notna(row["genre"]) else "Unknown",
            "energy": float(row["energy"]),
            "tempo": float(row["tempo"]),
            "danceability": float(row["danceability"]),
            "loudness": float(row["loudness"]),
            "speechiness": float(row["speechiness"]),
            "acousticness": float(row["acousticness"]),
            "instrumentalness": float(row["instrumentalness"]),
            "liveness": float(row["liveness"]),
            "valence": float(row["valence"])
        })
    return tracks

def get_recommendations(req):
    idx_list = df_metadata[df_metadata["track_id"] == req.track_id].index
    if len(idx_list) == 0:
        raise IndexError("Track tidak ditemukan")
        
    seed_idx = idx_list[0]
    seed_vector = combined_sparse[seed_idx]
    
    sims = cosine_similarity(seed_vector, combined_sparse)[0]
    top_indices_all = sims.argsort()[::-1]
    
    seed_song = str(df_metadata.iloc[seed_idx]['track_name'])
    
    recommendations = []
    recommendations_count = 0
    
    for idx in top_indices_all:
        if idx == seed_idx:
            continue
            
        row = df_metadata.iloc[idx]
        nama_lagu = str(row['track_name'])
        
        if seed_song.lower() in nama_lagu.lower() or nama_lagu.lower() in seed_song.lower():
            continue
            
        match_percentage = sims[idx] * 100
        recommendations_count += 1
        
        recommendations.append({
            "track_id": str(row["track_id"]),
            "track_name": nama_lagu,
            "artists": str(row["artists"]),
            "genre": str(row["genre"]) if pd.notna(row["genre"]) else "Unknown",
            "energy": float(row["energy"]),
            "tempo": float(row["tempo"]),
            "danceability": float(row["danceability"]),
            "loudness": float(row["loudness"]),
            "speechiness": float(row["speechiness"]),
            "acousticness": float(row["acousticness"]),
            "instrumentalness": float(row["instrumentalness"]),
            "liveness": float(row["liveness"]),
            "valence": float(row["valence"]),
            "similarity_dist_percent": round(float(match_percentage), 2),
            "rank": recommendations_count
        })
        
        if recommendations_count == req.n:
            break
            
    return recommendations