from fastapi import APIRouter, HTTPException
from app.schemas.music import RecommendRequest
from app.services import recommender, gemini_service
from app.core.ml_loader import df_metadata

router = APIRouter()

@router.get("/")
def root():
    return {
        "status": "ok",
        "total_songs": len(df_metadata),
        "matrix_format": "scipy.sparse.csr_matrix (.npz)",
        "algorithm": "Cosine Similarity (Sklearn Optimized)"
    }

@router.get("/search")
def search_track(q: str):
    tracks = recommender.search_tracks(q)
    if not tracks:
        raise HTTPException(status_code=404, detail="Lagu tidak ditemukan")
    return {"tracks": tracks}

@router.post("/recommend")
def recommend(req: RecommendRequest):
    try:
        recs = recommender.get_recommendations(req)
        return {"recommendations": recs}
    except IndexError:
         raise HTTPException(status_code=404, detail="Track tidak ditemukan")

@router.get("/describe")
async def describe_track(track_name: str, artist: str, energy: float, tempo: float, danceability: float, valence: float, acousticness: float, genre: str):
    try:
        result = gemini_service.generate_track_description(track_name, artist, energy, tempo, danceability, valence, acousticness, genre)
        return result
    except Exception as e:
        error_str = str(e).lower()
        if "api limit tercapai" in error_str:
             raise HTTPException(status_code=429, detail=str(e))
        raise HTTPException(status_code=500, detail=str(e))