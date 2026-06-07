# Gunakan image Python 3.12 yang ringan
FROM python:3.12-slim

# Set direktori kerja di dalam container
WORKDIR /app

# Salin file requirements dan install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Salin seluruh kode aplikasi
COPY . .

# Ekspos port internal container (Ubah ke 8001)
EXPOSE 8001

# Jalankan Uvicorn di port 8001
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001"]