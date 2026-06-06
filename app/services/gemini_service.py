import os
import time
import hashlib
from google import genai
from dotenv import load_dotenv

load_dotenv()
api_key = os.environ.get("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)

description_cache: dict[str, tuple[str, float]] = {}
CACHE_TTL_SECONDS = 3600

def get_cache_key(track_name, artist, energy, tempo, danceability, valence, acousticness, genre):
    raw = f"{track_name}|{artist}|{energy:.2f}|{tempo:.0f}|{danceability:.2f}|{valence:.2f}|{acousticness:.3f}|{genre}"
    return hashlib.md5(raw.encode()).hexdigest()

def generate_track_description(track_name, artist, energy, tempo, danceability, valence, acousticness, genre):
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
            raise Exception("API Limit tercapai. Mohon tunggu sekitar 30 detik.")
        raise Exception("Tidak dapat memuat deskripsi saat ini.")