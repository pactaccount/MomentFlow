"""Zero-shot visual moment retrieval over sampled video frames."""
from __future__ import annotations

import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

MODEL_ID = "google/siglip2-base-patch16-224"
MODEL_REVISION = "75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2"


def normalize(v):
    v = np.asarray(v, dtype=np.float32)
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12)


def sampled_frames(path: Path, interval=1.0, max_duration=120.0):
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise ValueError("Video cannot be decoded")
        fps, count = cap.get(cv2.CAP_PROP_FPS), cap.get(cv2.CAP_PROP_FRAME_COUNT)
        if not np.isfinite(fps) or fps <= 0 or count <= 0:
            raise ValueError("Invalid video duration or frame rate")
        duration = count / fps
        if duration > max_duration:
            raise ValueError(f"Video exceeds {max_duration:g} seconds")
        for stamp in np.arange(0, duration, interval):
            cap.set(cv2.CAP_PROP_POS_MSEC, float(stamp * 1000))
            ok, frame = cap.read()
            if ok:
                height, width = frame.shape[:2]
                if max(height, width) > 512:
                    scale = 512 / max(height, width)
                    frame = cv2.resize(frame, (round(width * scale), round(height * scale)))
                yield float(stamp), Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    finally:
        cap.release()


class SearchService:
    def __init__(self, data_dir: Path, encoder=None):
        self.data_dir = data_dir
        data_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = data_dir / "index.sqlite3"
        self.model_id = MODEL_ID
        self.encoder = encoder
        self.lock = threading.RLock()
        with self.db() as db:
            db.execute("CREATE TABLE IF NOT EXISTS videos(id TEXT PRIMARY KEY, filename TEXT, path TEXT, status TEXT, error TEXT)")
            db.execute("CREATE TABLE IF NOT EXISTS frames(video_id TEXT, stamp REAL, vector BLOB, PRIMARY KEY(video_id,stamp))")

    @contextmanager
    def db(self):
        db = sqlite3.connect(self.db_path, timeout=30)
        db.row_factory = sqlite3.Row
        try:
            yield db
            db.commit()
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

    def model(self):
        with self.lock:
            if self.encoder is None:
                from hf_space_encoder import HuggingFaceSpaceEncoder
                self.encoder = HuggingFaceSpaceEncoder()
            return self.encoder

    def add_video(self, video_id, filename, path):
        with self.db() as db:
            db.execute("INSERT INTO videos VALUES(?,?,?,?,NULL)", (video_id, Path(filename).name[:200], str(path), "queued"))

    def index_video(self, video_id, path):
        try:
            with self.db() as db:
                db.execute("UPDATE videos SET status='indexing' WHERE id=?", (video_id,))
            rows, batch = [], []
            for stamp, image in sampled_frames(path):
                batch.append((stamp, image))
                if len(batch) == 8:
                    rows.extend(self.embed_batch(video_id, batch))
                    batch.clear()
            if batch:
                rows.extend(self.embed_batch(video_id, batch))
            if not rows:
                raise ValueError("No video frames decoded")
            with self.db() as db:
                db.executemany("INSERT INTO frames VALUES(?,?,?)", rows)
                db.execute("UPDATE videos SET status='ready' WHERE id=?", (video_id,))
        except Exception as exc:
            with self.db() as db:
                db.execute("UPDATE videos SET status='failed', error=? WHERE id=?", (str(exc)[:300], video_id))

    def embed_batch(self, video_id, batch):
        with self.lock:
            vectors = normalize(self.model().images([image for _, image in batch]))
        if len(vectors) != len(batch):
            raise RuntimeError("Encoder returned the wrong frame count")
        return [(video_id, stamp, vector.tobytes()) for (stamp, _), vector in zip(batch, vectors)]

    def list_videos(self):
        with self.db() as db:
            return [dict(r) for r in db.execute("SELECT id,filename,status,error FROM videos ORDER BY rowid DESC")]

    def frame_count(self):
        with self.db() as db:
            return db.execute("SELECT COUNT(*) FROM frames").fetchone()[0]

    def video_path(self, video_id):
        with self.db() as db:
            row = db.execute("SELECT path FROM videos WHERE id=? AND status='ready'", (video_id,)).fetchone()
        return Path(row[0]) if row else None

    def search(self, query, limit=8):
        with self.db() as db:
            rows = db.execute("SELECT f.video_id,f.stamp,f.vector,v.filename FROM frames f JOIN videos v ON v.id=f.video_id WHERE v.status='ready'").fetchall()
        if not rows:
            return []
        with self.lock:
            q = normalize(self.model().text(query).reshape(1, -1))[0]
        vectors = np.stack([np.frombuffer(r[2], dtype=np.float32) for r in rows])
        if vectors.shape[1] != len(q):
            raise RuntimeError("Index dimension mismatch; rebuild the index")
        scores = vectors @ q
        hits = []
        for i in np.argsort(-scores):
            r = rows[int(i)]
            if any(h["video_id"] == r[0] and abs(h["timestamp_sec"] - r[1]) < 2 for h in hits):
                continue
            hits.append({"video_id": r[0], "filename": r[3], "timestamp_sec": r[1], "score": round(float(scores[i]), 4), "video_url": f"/api/media/{r[0]}"})
            if len(hits) >= limit:
                break
        return hits
