FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 DATA_DIR=/data HF_HOME=/data/hf
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libstdc++6 libglib2.0-0 ffmpeg tesseract-ocr tesseract-ocr-eng && rm -rf /var/lib/apt/lists/*
COPY Prototype/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
COPY Prototype/main.py Prototype/search_core.py Prototype/hf_space_encoder.py Prototype/product_service.py Prototype/media_tools.py Prototype/worker.py Prototype/telemetry.py Prototype/render_start.py Prototype/evaluate.py ./
COPY Prototype/static ./static
RUN useradd -m -u 10001 appuser && mkdir /data && chown appuser:appuser /data
USER appuser
CMD ["python", "render_start.py"]
