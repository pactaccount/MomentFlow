FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 DATA_DIR=/data HF_HOME=/data/hf
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libstdc++6 libglib2.0-0 ffmpeg && rm -rf /var/lib/apt/lists/*
COPY Prototype/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
COPY Prototype/apple_encoder.py Prototype/apple_grounder.py Prototype/search_core.py Prototype/product_service.py Prototype/media_tools.py Prototype/main.py Prototype/worker.py ./
COPY Prototype/static ./static
RUN useradd -m -u 10001 appuser && mkdir -p /data && chown -R appuser:appuser /data
USER appuser
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
