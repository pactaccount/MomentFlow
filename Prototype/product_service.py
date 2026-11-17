"""Clean 2-model Apple stack video moment search service (MobileCLIP-B + FastVLM-0.5B)."""
from __future__ import annotations

import logging
from pathlib import Path
import re
import shutil
import time
from typing import List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

from apple_encoder import AppleMobileClipEncoder
from apple_grounder import AppleFastVLMVerifier
from media_tools import prepare_playback, sampled_frames, scene_stamps
from search_core import SearchService, normalize

logger = logging.getLogger("product_service")

PIPELINE_VERSION = "apple-2model-v1"
CALIBRATED_ABSENT_THRESHOLD = 0.20


def minmax_scale(arr: np.ndarray) -> np.ndarray:
    span = arr.max() - arr.min()
    return (arr - arr.min()) / max(1e-5, span) if span > 1e-6 else np.zeros_like(arr)


class ProductService(SearchService):
    """Unified video search service powered strictly by the 2-model Apple stack."""

    def __init__(self, data_dir: Path, encoder=None, verifier=None):
        super().__init__(data_dir, encoder=encoder)
        self.verifier = verifier
        with self.db() as db:
            db.execute("PRAGMA journal_mode=WAL")
            columns = {row[1] for row in db.execute("PRAGMA table_info(videos)")}
            for name, definition in {
                "duration_sec": "REAL", "size_bytes": "INTEGER", "sha256": "TEXT",
                "has_audio": "INTEGER", "created_at": "REAL", "indexed_version": "TEXT",
            }.items():
                if name not in columns:
                    db.execute(f"ALTER TABLE videos ADD COLUMN {name} {definition}")
            db.execute(
                "CREATE TABLE IF NOT EXISTS jobs("
                "video_id TEXT PRIMARY KEY, status TEXT NOT NULL, attempts INTEGER NOT NULL DEFAULT 0, "
                "lease_until REAL, progress TEXT, error TEXT)"
            )
            db.execute("CREATE TABLE IF NOT EXISTS feedback(id INTEGER PRIMARY KEY,video_id TEXT,query TEXT,stamp REAL,value TEXT,created_at REAL)")
            db.execute("CREATE INDEX IF NOT EXISTS videos_sha ON videos(sha256)")

    def get_verifier(self) -> AppleFastVLMVerifier:
        """Lazy load Model 2 (Apple FastVLM-0.5B) for candidate verification."""
        with self.lock:
            if self.verifier is None:
                self.verifier = AppleFastVLMVerifier()
            return self.verifier

    def playback_file(self, video_id: str) -> Path:
        return self.data_dir / "playback" / f"{video_id}.mp4"

    def thumbnail_path(self, video_id: str) -> Optional[Path]:
        path = self.data_dir / "thumbnails" / f"{video_id}.jpg"
        return path if path.is_file() else None

    def video_path(self, video_id: str) -> Optional[Path]:
        with self.db() as db:
            ready = db.execute("SELECT 1 FROM videos WHERE id=? AND status='ready'", (video_id,)).fetchone()
        path = self.playback_file(video_id)
        return path if ready and path.is_file() else None

    def add_video(self, video_id: str, filename: str, path: Path, metadata: Optional[dict] = None, sha256: Optional[str] = None):
        metadata = metadata or {}
        with self.db() as db:
            db.execute(
                "INSERT INTO videos(id,filename,path,status,error,duration_sec,size_bytes,sha256,has_audio,created_at) VALUES(?,?,?,?,?,?,?,?,?,?)",
                (video_id, Path(filename).name[:200], str(path), "queued", None,
                 metadata.get("duration_sec"), path.stat().st_size if path.exists() else None,
                 sha256, int(metadata.get("has_audio", False)), time.time()),
            )
            db.execute("INSERT INTO jobs(video_id,status,progress) VALUES(?,?,?)", (video_id, "queued", "Waiting to index"))

    def find_sha(self, sha256: str) -> Optional[str]:
        with self.db() as db:
            row = db.execute("SELECT id FROM videos WHERE sha256=? AND status!='failed' LIMIT 1", (sha256,)).fetchone()
            return row[0] if row else None

    def claim_job(self) -> Optional[Tuple[str, Path]]:
        with self.db() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(
                "SELECT j.video_id,v.path FROM jobs j JOIN videos v ON v.id=j.video_id "
                "WHERE (j.status='queued' OR (j.status='running' AND j.lease_until<?)) AND j.attempts<3 "
                "ORDER BY v.created_at LIMIT 1", (time.time(),),
            ).fetchone()
            if row:
                db.execute("UPDATE jobs SET status='running',attempts=attempts+1,lease_until=?,progress='Preparing media' WHERE video_id=?",
                           (time.time() + 120, row[0]))
                return row[0], Path(row[1])
        return None

    def set_progress(self, video_id: str, message: Optional[str]):
        with self.db() as db:
            if message is None:
                db.execute("UPDATE jobs SET lease_until=? WHERE video_id=? AND status='running'",
                           (time.time() + 120, video_id))
            else:
                db.execute("UPDATE jobs SET progress=?,lease_until=? WHERE video_id=? AND status='running'",
                           (message, time.time() + 120, video_id))

    def process_job(self, video_id: str, path: Path):
        """Index video with Model 1 (Apple MobileCLIP-B) at 100+ fps."""
        try:
            with self.db() as db:
                db.execute("UPDATE videos SET status='indexing',error=NULL WHERE id=?", (video_id,))

            self.set_progress(video_id, "Finding scene cuts")
            extras = scene_stamps(path)

            self.set_progress(video_id, "Encoding video frames with Apple MobileCLIP-B")
            frame_rows, batch = [], []
            first_image = None
            for stamp, image in sampled_frames(path, interval=1.0, extra_stamps=extras):
                if first_image is None:
                    first_image = image.copy()
                batch.append((stamp, image))
                if len(batch) >= 16:
                    frame_rows.extend(self.embed_batch(video_id, batch))
                    batch.clear()
            if batch:
                frame_rows.extend(self.embed_batch(video_id, batch))

            if first_image is None:
                raise ValueError("No video frames could be decoded")

            # Save thumbnail
            thumb_dir = self.data_dir / "thumbnails"
            thumb_dir.mkdir(exist_ok=True)
            first_image.thumbnail((480, 270))
            first_image.save(thumb_dir / f"{video_id}.jpg", quality=80)

            # Transcode web playback MP4
            self.set_progress(video_id, "Preparing browser playback")
            prepare_playback(path, self.playback_file(video_id))

            # Store in DB
            with self.db() as db:
                db.execute("DELETE FROM frames WHERE video_id=?", (video_id,))
                db.executemany("INSERT INTO frames(video_id,stamp,vector) VALUES(?,?,?)", frame_rows)
                db.execute("UPDATE videos SET status='ready',error=NULL,indexed_version=? WHERE id=?",
                           (PIPELINE_VERSION, video_id))
                db.execute("DELETE FROM jobs WHERE video_id=?", (video_id,))
            logger.info("Video %s successfully indexed with %d frames.", video_id, len(frame_rows))
        except Exception as exc:
            logger.exception("Failed to process job for %s: %s", video_id, exc)
            with self.db() as db:
                db.execute("UPDATE videos SET status='failed',error=? WHERE id=?", (str(exc)[:400], video_id))
                db.execute("UPDATE jobs SET status='failed',error=? WHERE video_id=?", (str(exc)[:400], video_id))
            raise

    def delete_video(self, video_id: str) -> bool:
        with self.db() as db:
            row = db.execute("SELECT path,status FROM videos WHERE id=?", (video_id,)).fetchone()
            if not row:
                return False
            if row[1] in {"queued", "indexing"}:
                raise ValueError("Wait until indexing finishes before deleting")
            for table in ("frames", "jobs", "feedback"):
                db.execute(f"DELETE FROM {table} WHERE video_id=?", (video_id,))
            db.execute("DELETE FROM videos WHERE id=?", (video_id,))
        Path(row[0]).unlink(missing_ok=True)
        self.playback_file(video_id).unlink(missing_ok=True)
        (self.data_dir / "thumbnails" / f"{video_id}.jpg").unlink(missing_ok=True)
        return True

    def add_feedback(self, video_id: str, query: str, stamp: float, value: str):
        if value not in {"helpful", "wrong_moment"}:
            raise ValueError("Invalid feedback")
        with self.db() as db:
            db.execute("INSERT INTO feedback(video_id,query,stamp,value,created_at) VALUES(?,?,?,?,?)",
                       (video_id, query[:300], stamp, value, time.time()))

    @staticmethod
    def _fmt_time(sec: float) -> str:
        n = max(0, int(round(float(sec))))
        return f"{n // 60}:{n % 60:02d}"

    def search(self, query: str, limit: int = 8) -> list[dict]:
        """2-stage search: Model 1 (MobileCLIP-B) fast timeline filter + Model 2 (FastVLM) verifier."""
        with self.db() as db:
            rows = db.execute(
                "SELECT f.video_id,f.stamp,f.vector,v.filename,v.path "
                "FROM frames f JOIN videos v ON v.id=f.video_id WHERE v.status='ready' "
                "ORDER BY f.video_id, f.stamp"
            ).fetchall()
        if not rows:
            return []

        # Stage 1: MobileCLIP Timeline Dot-Product
        with self.lock:
            q_vec = normalize(self.model().text(query).reshape(1, -1))[0]

        frames_by_vid: dict[str, list[tuple[float, np.ndarray, str, str]]] = {}
        for r in rows:
            frames_by_vid.setdefault(r[0], []).append((float(r[1]), np.frombuffer(r[2], dtype=np.float32), r[3], r[4]))

        candidate_peaks = []
        for vid, frame_list in frames_by_vid.items():
            stamps = [item[0] for item in frame_list]
            vectors = np.stack([item[1] for item in frame_list])
            filename = frame_list[0][2]
            orig_path = frame_list[0][3]

            scores = vectors @ q_vec
            smoothed = np.convolve(scores, np.ones(3, dtype=np.float32) / 3.0, mode="same") if len(scores) >= 3 else scores

            # Find local maxima peaks
            peak_indices = []
            for i in range(len(smoothed)):
                if smoothed[i] < CALIBRATED_ABSENT_THRESHOLD:
                    continue
                left = smoothed[i - 1] if i > 0 else -1
                right = smoothed[i + 1] if i < len(smoothed) - 1 else -1
                if smoothed[i] >= left and smoothed[i] >= right:
                    peak_indices.append(i)

            if not peak_indices:
                best_i = int(np.argmax(scores))
                if scores[best_i] >= CALIBRATED_ABSENT_THRESHOLD:
                    peak_indices.append(best_i)

            for p_idx in peak_indices:
                p_stamp = stamps[p_idx]
                p_score = float(smoothed[p_idx])
                # Expand temporal window [start, end]
                s_stamp = max(0.0, p_stamp - 5.0)
                e_stamp = p_stamp + 8.0
                candidate_peaks.append({
                    "video_id": vid,
                    "filename": filename,
                    "orig_path": orig_path,
                    "timestamp_sec": p_stamp,
                    "start_sec": s_stamp,
                    "end_sec": e_stamp,
                    "score": p_score,
                })

        if not candidate_peaks:
            return []

        # Rank candidates by Model 1 similarity score
        candidate_peaks.sort(key=lambda x: x["score"], reverse=True)
        top_candidates = candidate_peaks[:min(3, limit)]

        # Stage 2: Model 2 (Apple FastVLM) Action Verification
        verifier = self.get_verifier()
        final_hits = []

        for cand in top_candidates:
            # Extract keyframe at candidate peak
            cap = cv2.VideoCapture(cand["orig_path"])
            cap.set(cv2.CAP_PROP_POS_MSEC, cand["timestamp_sec"] * 1000.0)
            ret, frame = cap.read()
            cap.release()

            if not ret or frame is None:
                continue

            pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            verified, explanation = verifier.verify_action(pil_img, query)

            # Calculate duration and formatted interval
            start_s = round(cand["start_sec"], 1)
            end_s = round(cand["end_sec"], 1)
            duration_s = max(1, int(round(end_s - start_s)))
            interval_fmt = f"{self._fmt_time(start_s)} – {self._fmt_time(end_s)}"

            final_hits.append({
                "video_id": cand["video_id"],
                "filename": cand["filename"],
                "video_url": f"/api/media/{cand['video_id']}",
                "timestamp_sec": cand["timestamp_sec"],
                "start_sec": start_s,
                "end_sec": end_s,
                "duration_sec": duration_s,
                "interval_formatted": interval_fmt,
                "score": round(cand["score"], 4),
                "source": "Apple Stack",
                "sources": ["MobileCLIP-B", "FastVLM-0.5B"],
                "verified": verified,
                "evidence": f"Apple FastVLM verified: {explanation}" if verified else f"Detected visual moment: {explanation}",
            })

            if len(final_hits) >= limit:
                break

        return final_hits
