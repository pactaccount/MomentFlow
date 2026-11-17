FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 ffmpeg && rm -rf /var/lib/apt/lists/*
COPY Prototype/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
COPY Prototype/apple_encoder.py Prototype/apple_grounder.py Prototype/kaggle_deploy.py ./
COPY Prototype/static ./static
CMD ["python", "kaggle_deploy.py"]
