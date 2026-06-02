from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from google import genai
import faiss
import os
from dotenv import load_dotenv
import time
import hashlib

load_dotenv()

api_key = os.environ.get("GEMINI_API_KEY")

app = FastAPI()
client = genai.Client(api_key=api_key)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:3001",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:3001",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===== IN-MEMORY CACHE =====
description_cache: dict[str, tuple[str, float]] = {}
CACHE_TTL_SECONDS = 3600

def get_cache_key(track_name, artist, energy, tempo,
                  danceability, valence, acousticness, genre):
    raw = f"{track_name}|{artist}|{energy:.2f}|{tempo:.0f}|{danceability:.2f}|{valence:.2f}|{acousticness:.3f}|{genre}"
    return hashlib.md5(raw.encode()).hexdigest()

# ===== LOAD DATA =====
df = pd.read_csv("songs.csv")
df = df.rename(columns={"id": "track_id", "name": "track_name"})
df = df.dropna(subset=["danceability","energy","loudness","speechiness",
                        "acousticness","instrumentalness","liveness",
                        "valence","tempo","key","mode","genre"])
df = df.drop_duplicates(subset=["track_id"])
df = df.reset_index(drop=True)

FEATURES_9 = [
    "danceability", "energy", "loudness", "speechiness",
    "acousticness", "instrumentalness", "valence", "tempo"
]

FEATURES_11 = [
    "danceability", "energy", "loudness", "speechiness",
    "acousticness", "instrumentalness", "liveness",
    "valence", "tempo", "key", "mode"
]

# ===== LOAD FEATURE MATRIX =====
USE_KAGGLE = False
try:
    fm_9 = np.load("feature_matrix_9.npy").astype(np.float32)
    fm_11 = np.load("feature_matrix_11.npy").astype(np.float32)
    track_order = pd.read_csv("track_order.csv")

    # Reorder df sesuai urutan track_order dari Kaggle
    df = df.set_index("track_id").loc[track_order["track_id"]].reset_index()
    df = df.reset_index(drop=True)

    if fm_9.shape[0] != len(df) or fm_11.shape[0] != len(df):
        raise ValueError(f"Shape mismatch: fm_9={fm_9.shape[0]}, fm_11={fm_11.shape[0]}, df={len(df)}")

    feature_matrix_9 = fm_9.copy()
    feature_matrix_11 = fm_11.copy()
    USE_KAGGLE = True
    print(f"Feature matrix loaded dari Kaggle PySpark LSH!")
    print(f"9 fitur : {feature_matrix_9.shape}")
    print(f"11 fitur: {feature_matrix_11.shape}")

except Exception as e:
    print(f"Fallback ke StandardScaler lokal: {e}")
    scaler_9 = StandardScaler()
    feature_matrix_9 = scaler_9.fit_transform(
        df[FEATURES_9].values
    ).astype(np.float32).copy()

    scaler_11 = StandardScaler()
    feature_matrix_11 = scaler_11.fit_transform(
        df[FEATURES_11].values
    ).astype(np.float32).copy()

# ===== BUILD FAISS INDEX =====
def build_faiss_index(matrix: np.ndarray) -> faiss.IndexIVFFlat:
    faiss.normalize_L2(matrix)
    n_clusters = min(1024, len(df) // 10)
    dim = matrix.shape[1]
    quantizer = faiss.IndexFlatIP(dim)
    index = faiss.IndexIVFFlat(quantizer, dim, n_clusters, faiss.METRIC_INNER_PRODUCT)
    index.train(matrix)
    index.add(matrix)
    index.nprobe = 50
    return index

print("Building FAISS index 9 fitur...")
index_9 = build_faiss_index(feature_matrix_9)

print("Building FAISS index 11 fitur...")
index_11 = build_faiss_index(feature_matrix_11)

# ===== GENRE INDEX =====
le = LabelEncoder()
df.loc[:, "genre_encoded"] = le.fit_transform(df["genre"].astype(str))
genre_scaler = StandardScaler()
genre_scaled = genre_scaler.fit_transform(
    df["genre_encoded"].values.reshape(-1, 1)
).astype(np.float32)

# Genre index pakai 11 fitur + genre
feature_matrix_genre = np.hstack([
    feature_matrix_11 * 0.7,
    genre_scaled * 0.3
]).astype(np.float32).copy()

print("Building FAISS index genre...")
index_genre = build_faiss_index(feature_matrix_genre)

print(f"Semua FAISS index siap!")
print(f"Dataset: {len(df):,} lagu | {df['genre'].nunique()} genre")
print(f"Mode: {'Kaggle PySpark LSH' if USE_KAGGLE else 'Lokal StandardScaler'}")

# ===== MODELS =====
class RecommendRequest(BaseModel):
    track_id: str
    n: int = 10
    use_genre: Optional[bool] = False
    feature_mode: Optional[str] = "11"  # "9" atau "11"

# ===== ENDPOINTS =====
@app.get("/")
def root():
    return {
        "status": "ok",
        "total_songs": len(df),
        "index": "FAISS IVFFlat",
        "source": "Kaggle PySpark LSH" if USE_KAGGLE else "Local StandardScaler"
    }

@app.get("/search")
def search_track(q: str):
    mask_exact = df["track_name"].str.lower() == q.lower()
    mask_name = df["track_name"].str.contains(q, case=False, na=False)
    mask_artist = df["artists"].str.contains(q, case=False, na=False)

    results_exact = df[mask_exact].sort_values("popularity", ascending=False)
    results_name = df[mask_name & ~mask_exact].sort_values("popularity", ascending=False)
    results_artist = df[mask_artist & ~mask_name].sort_values("popularity", ascending=False)

    results = pd.concat([results_exact, results_name, results_artist])
    results = results.drop_duplicates(subset=["track_name", "artists"])
    results = results.head(15)

    if results.empty:
        raise HTTPException(status_code=404, detail="Lagu tidak ditemukan")

    tracks = []
    for _, row in results.iterrows():
        tracks.append({
            "id": row["track_id"],
            "name": row["track_name"],
            "artist": row["artists"],
            "energy": round(float(row["energy"]), 2),
            "tempo": round(float(row["tempo"]), 1),
            "danceability": round(float(row["danceability"]), 2),
            "acousticness": round(float(row["acousticness"]), 3),
            "loudness": round(float(row["loudness"]), 2),
            "speechiness": round(float(row["speechiness"]), 3),
            "instrumentalness": round(float(row["instrumentalness"]), 3),
            "liveness": round(float(row["liveness"]), 3),
            "valence": round(float(row["valence"]), 2),
            "genre": row["genre"],
        })
    return {"tracks": tracks}

@app.post("/recommend")
def recommend(req: RecommendRequest):
    idx = df[df["track_id"] == req.track_id].index
    if len(idx) == 0:
        raise HTTPException(status_code=404, detail="Track tidak ditemukan")

    seed_idx = idx[0]

    # Pilih index
    if req.use_genre:
        query_vec = feature_matrix_genre[seed_idx].reshape(1, -1).copy()
        index = index_genre
    elif req.feature_mode == "9":
        query_vec = feature_matrix_9[seed_idx].reshape(1, -1).copy()
        index = index_9
    else:
        query_vec = feature_matrix_11[seed_idx].reshape(1, -1).copy()
        index = index_11

    faiss.normalize_L2(query_vec)
    scores, indices = index.search(query_vec, req.n * 3 + 1)

    recommendations = []
    seen = set()

    for score, i in zip(scores[0], indices[0]):
        if i == seed_idx or i < 0:
            continue
        row = df.iloc[i]
        key = (row["track_name"].lower(), row["artists"].lower())
        if key in seen:
            continue
        seen.add(key)

        recommendations.append({
            "track_id": row["track_id"],
            "track_name": row["track_name"],
            "artists": row["artists"],
            "genre": row["genre"],
            "energy": round(float(row["energy"]), 2),
            "tempo": round(float(row["tempo"]), 1),
            "danceability": round(float(row["danceability"]), 2),
            "loudness": round(float(row["loudness"]), 2),
            "speechiness": round(float(row["speechiness"]), 3),
            "acousticness": round(float(row["acousticness"]), 3),
            "instrumentalness": round(float(row["instrumentalness"]), 3),
            "liveness": round(float(row["liveness"]), 3),
            "valence": round(float(row["valence"]), 2),
            "similarity_dist": round(float(1 - score), 4),
        })

        if len(recommendations) >= req.n:
            break

    return {"recommendations": recommendations}

@app.get("/describe")
async def describe_track(
    track_name: str, artist: str, energy: float,
    tempo: float, danceability: float, valence: float,
    acousticness: float, genre: str
):
    cache_key = get_cache_key(
        track_name, artist, energy, tempo,
        danceability, valence, acousticness, genre
    )
    now = time.time()

    if cache_key in description_cache:
        cached_desc, cached_time = description_cache[cache_key]
        if now - cached_time < CACHE_TTL_SECONDS:
            print(f"[Cache HIT] {track_name}")
            return {"description": cached_desc, "cached": True}
        else:
            del description_cache[cache_key]

    energy_desc = "sangat energik" if energy > 0.8 else "energik" if energy > 0.6 else "sedang" if energy > 0.4 else "tenang"
    mood_desc = "ceria dan positif" if valence > 0.7 else "netral" if valence > 0.4 else "melankolis atau gelap"
    tempo_desc = "cepat" if tempo > 140 else "sedang" if tempo > 90 else "lambat"
    acoustic_desc = "akustik dan organik" if acousticness > 0.6 else "elektronik atau produksi studio"
    dance_desc = "sangat danceable" if danceability > 0.7 else "cukup groovy" if danceability > 0.5 else "tidak terlalu danceable"

    prompt = f"""Kamu adalah music reviewer Gen Z dan kurator playlist kekinian. Tulis deskripsi singkat (3-4 kalimat) untuk lagu "{track_name}" dari {artist} pakai Bahasa Indonesia yang santai, asik, dan mengalir. Boleh campur sedikit istilah gaul atau bahasa Inggris yang natural (seperti vibes, lowkey, chill, overthinking, relate).

Berdasarkan data berikut:
- Genre: {genre}
- Nuansa: {mood_desc}
- Energi: {energy_desc} ({energy:.2f})
- Tempo: {tempo_desc} ({tempo:.0f} BPM)
- Tekstur suara: {acoustic_desc}
- Grooviness: {dance_desc}

ATURAN PENTING:
1. DILARANG KERAS menggunakan format Markdown apa pun. Tulis semuanya murni sebagai teks biasa.
2. Tulisannya harus ngena, storytelling-nya dapet, tapi tidak kaku atau puitis jadul.
3. Bayangkan kamu sedang merekomendasikan lagu ini ke teman. Bahas juga kemungkinan tema liriknya berdasarkan mood.
4. Jangan pernah mulai dengan "Lagu ini" — variasikan opening-nya."""

    try:
        response = client.models.generate_content(
            model="gemini-3.5-flash",
            contents=prompt
        )
        description = response.text
        description_cache[cache_key] = (description, now)
        print(f"[API HIT] {track_name}")
        return {"description": description, "cached": False}

    except Exception as e:
        error_msg = str(e).lower()
        print(f"Error Gemini API: {e}")
        if "429" in error_msg or "resource_exhausted" in error_msg:
            raise HTTPException(status_code=429, detail="API Limit tercapai. Mohon tunggu sekitar 30 detik.")
        raise HTTPException(status_code=500, detail="Tidak dapat memuat deskripsi saat ini.")