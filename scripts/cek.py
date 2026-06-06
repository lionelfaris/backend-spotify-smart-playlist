import os
from google import genai
from dotenv import load_dotenv

# Load API Key dari file .env
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

# Inisialisasi Client
client = genai.Client(api_key=api_key)

print("Daftar Model Gemini yang Bisa Dipakai:")
print("-" * 40)

# Di library versi terbaru, cukup gunakan client.models.list()
for model in client.models.list():
    print(f"Model ID : {model.name}")
    print("-" * 40)