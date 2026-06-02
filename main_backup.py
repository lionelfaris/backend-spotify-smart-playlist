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

# ===== MIDDLEWARE =====
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

# ===== IN-MEMORY DESCRIPTION CACHE =====
description_cache: dict[str, tuple[str, float]] = {}
CACHE_TTL_SECONDS = 3600

def get_cache_key(
    track_name: str, artist: str, energy: float,
    tempo: float, danceability: float, valence: float,
    acousticness: float, genre: str
) -> str:
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

AUDIO_FEATURES = ["danceability","energy","loudness","speechiness",
                  "acousticness","instrumentalness","liveness",
                  "valence","tempo","key","mode"]

# Load atau buat feature matrix
try:
    feature_matrix = np.load("feature_matrix.npy").astype(np.float32)
    if feature_matrix.shape[0] != len(df):
        raise ValueError(f"Shape mismatch: {feature_matrix.shape[0]} vs {len(df)}")
    print(f"Feature matrix loaded dari Kaggle: {feature_matrix.shape}")
except Exception as e:
    print(f"Fallback ke StandardScaler lokal: {e}")
    scaler = StandardScaler()
    feature_matrix = scaler.fit_transform(df[AUDIO_FEATURES].values).astype(np.float32).copy()

# Normalize untuk cosine similarity via FAISS inner product
faiss.normalize_L2(feature_matrix)

# Build FAISS index audio only
n_clusters = min(1024, len(df) // 10)
dim_audio = feature_matrix.shape[1]
quantizer_audio = faiss.IndexFlatIP(dim_audio)
index_audio = faiss.IndexIVFFlat(quantizer_audio, dim_audio, n_clusters, faiss.METRIC_INNER_PRODUCT)
index_audio.train(feature_matrix)
index_audio.add(feature_matrix)
index_audio.nprobe = 50

# Build FAISS index dengan genre
le = LabelEncoder()
df.loc[:, "genre_encoded"] = le.fit_transform(df["genre"].astype(str))
genre_scaler = StandardScaler()
genre_scaled = genre_scaler.fit_transform(
    df["genre_encoded"].values.reshape(-1, 1)
).astype(np.float32)

feature_matrix_with_genre = np.hstack([
    feature_matrix * 0.7,
    genre_scaled * 0.3
]).astype(np.float32)
faiss.normalize_L2(feature_matrix_with_genre)

dim_genre = feature_matrix_with_genre.shape[1]
quantizer_genre = faiss.IndexFlatIP(dim_genre)
index_genre = faiss.IndexIVFFlat(quantizer_genre, dim_genre, n_clusters, faiss.METRIC_INNER_PRODUCT)
index_genre.train(feature_matrix_with_genre)
index_genre.add(feature_matrix_with_genre)
index_genre.nprobe = 50

print(f"FAISS index siap: {index_audio.ntotal:,} lagu")
print(f"Dataset loaded: {len(df):,} lagu | {df['genre'].nunique()} genre")

# ===== MODELS =====
class RecommendRequest(BaseModel):
    track_id: str
    n: int = 10
    use_genre: Optional[bool] = False

# ===== ENDPOINTS =====
@app.get("/")
def root():
    return {"status": "ok", "total_songs": len(df), "index": "FAISS IVFFlat"}

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
    results = results.head(8)

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

    # Pilih index berdasarkan toggle genre
    if req.use_genre:
        query_vec = feature_matrix_with_genre[seed_idx].reshape(1, -1).copy()
        index = index_genre
    else:
        query_vec = feature_matrix[seed_idx].reshape(1, -1).copy()
        index = index_audio

    faiss.normalize_L2(query_vec)

    # Search N*3 kandidat untuk antisipasi duplikat
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
1. DILARANG KERAS menggunakan format Markdown apa pun (JANGAN gunakan tanda bintang * atau ** untuk kata-kata bahasa Inggris/slang). Tulis semuanya murni sebagai teks biasa (plain text).
2. Tulisannya harus ngena, storytelling-nya dapet, tapi sama sekali TIDAK BOLEH kaku atau puitis jadul.
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