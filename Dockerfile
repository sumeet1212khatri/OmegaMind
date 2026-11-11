FROM python:3.11-slim

# 1. Install C++17 compiler & OpenMP runtime for AVX2/FMA inference
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    g++ \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 2. Hugging Face Spaces runs as non-root UID 1000
RUN useradd -m -u 1000 user
WORKDIR /app
RUN chown -R user:user /app

# 3. Environment setup for HF cache, OpenMP threads & real-time logs
ENV HOME=/app \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/app/.cache/huggingface \
    HF_HUB_ENABLE_HF_TRANSFER=0 \
    OMP_NUM_THREADS=2

# 4. Install Python dependencies (explicitly includes psutil)
COPY --chown=user:user requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt psutil==6.0.0

# 5. Copy project files
COPY --chown=user:user ["inference.cpp", "app.py", "index.html", "omegamind web.html", "token-journey.html", "./"]

# 6. Fix only line 213 (tmp[5]+t[6] -> tmp[5]+tmp[6]) & compile C++17 AVX2 binary
RUN sed -i 's/tmp\[5\]+t\[6\]/tmp[5]+tmp[6]/g' inference.cpp && \
    g++ -O3 -mavx2 -mfma -fopenmp -std=c++17 inference.cpp -o inference && \
    chmod +x inference && \
    mkdir -p /app/.cache/huggingface && \
    chown -R user:user /app

# 7. Switch to HF Space user & start FastAPI on port 7860
USER user
EXPOSE 7860

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]