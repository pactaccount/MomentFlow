"""Base database storage, vector normalization, and Apple MobileCLIP embedding manager."""
from __future__ import annotations

import os
import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

MODEL_ID = "apple/MobileCLIP-B"


def normalize(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32)
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12)


class SearchService:
    """Base storage and embedding service for MomentFlow using Apple MobileCLIP."""

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
        """Lazy load Apple MobileCLIP encoder."""
        with self.lock:
            if self.encoder is None:
                from apple_encoder import AppleMobileClipEncoder
                self.encoder = AppleMobileClipEncoder()
            return self.encoder

    def embed_batch(self, video_id: str, batch: list[tuple[float, Image.Image]]) -> list[tuple[str, float, bytes]]:
        """Encode a batch of (timestamp, PIL.Image) tuples into database rows."""
        with self.lock:
            vectors = normalize(self.model().images([image for _, image in batch]))
        if len(vectors) != len(batch):
            raise RuntimeError("Encoder returned the wrong frame count")
        return [(video_id, stamp, vector.tobytes()) for (stamp, _), vector in zip(batch, vectors)]

    def frame_count(self) -> int:
        with self.db() as db:
            return db.execute("SELECT COUNT(*) FROM frames").fetchone()[0]

    def list_videos(self) -> list[dict]:
        with self.db() as db:
            return [dict(r) for r in db.execute("SELECT id,filename,status,error FROM videos ORDER BY rowid DESC")]

    def video_path(self, video_id: str) -> Optional[Path]:
        with self.db() as db:
            row = db.execute("SELECT path FROM videos WHERE id=? AND status='ready'", (video_id,)).fetchone()
        return Path(row[0]) if row else None

    def search(self, query: str, limit: int = 8) -> list[dict]:
        """Base frame similarity search (baseline retrieval)."""
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
