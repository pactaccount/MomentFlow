"""Deployable multimodal layer over the established SigLIP 2 visual baseline."""
from __future__ import annotations

import os
import re
import time
from pathlib import Path

import numpy as np

from media_tools import sampled_frames, scene_stamps
from search_core import SearchService, normalize

PIPELINE_VERSION = "multimodal-v1"


class ProductService(SearchService):
    def __init__(self, data_dir: Path, encoder=None, speech=None, ocr=None):
        super().__init__(data_dir, encoder=encoder)
        self.speech_engine = speech
        self.asr_model = None
        self.ocr_engine = ocr
        with self.db() as db:
            db.execute("PRAGMA journal_mode=WAL")
            columns = {row[1] for row in db.execute("PRAGMA table_info(videos)")}
            for name, definition in {
                "duration_sec": "REAL", "size_bytes": "INTEGER", "sha256": "TEXT",
                "has_audio": "INTEGER", "created_at": "REAL", "indexed_version": "TEXT",
            }.items():
                if name not in columns:
                    db.execute(f"ALTER TABLE videos ADD COLUMN {name} {definition}")
            db.execute("CREATE TABLE IF NOT EXISTS jobs(video_id TEXT PRIMARY KEY,status TEXT NOT NULL,attempts INTEGER NOT NULL DEFAULT 0,lease_until REAL,progress TEXT,error TEXT)")
            db.execute("CREATE VIRTUAL TABLE IF NOT EXISTS speech_fts USING fts5(video_id UNINDEXED,start UNINDEXED,end UNINDEXED,text)")
            db.execute("CREATE VIRTUAL TABLE IF NOT EXISTS screen_fts USING fts5(video_id UNINDEXED,start UNINDEXED,end UNINDEXED,text)")
            db.execute("CREATE TABLE IF NOT EXISTS feedback(id INTEGER PRIMARY KEY,video_id TEXT,query TEXT,stamp REAL,value TEXT,created_at REAL)")
            db.execute("CREATE INDEX IF NOT EXISTS videos_sha ON videos(sha256)")

    def add_video(self, video_id, filename, path, metadata=None, sha256=None):
        metadata = metadata or {}
        with self.db() as db:
            db.execute(
                "INSERT INTO videos(id,filename,path,status,error,duration_sec,size_bytes,sha256,has_audio,created_at) VALUES(?,?,?,?,?,?,?,?,?,?)",
                (video_id, Path(filename).name[:200], str(path), "queued", None,
                 metadata.get("duration_sec"), path.stat().st_size if path.exists() else None,
                 sha256, int(metadata.get("has_audio", False)), time.time()),
            )
            db.execute("INSERT INTO jobs(video_id,status,progress) VALUES(?,?,?)", (video_id, "queued", "Waiting to index"))

    def find_sha(self, sha256):
        with self.db() as db:
            row = db.execute("SELECT id FROM videos WHERE sha256=? AND status!='failed' LIMIT 1", (sha256,)).fetchone()
            return row[0] if row else None

    def claim_job(self):
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

    def set_progress(self, video_id, message):
        with self.db() as db:
            if message is None:
                db.execute("UPDATE jobs SET lease_until=? WHERE video_id=? AND status='running'",
                           (time.time() + 120, video_id))
            else:
                db.execute("UPDATE jobs SET progress=?,lease_until=? WHERE video_id=? AND status='running'",
                           (message, time.time() + 120, video_id))

    def process_job(self, video_id, path):
        try:
            with self.db() as db:
                db.execute("UPDATE videos SET status='indexing',error=NULL WHERE id=?", (video_id,))
                row = db.execute("SELECT has_audio FROM videos WHERE id=?", (video_id,)).fetchone()
            self.set_progress(video_id, "Finding scene changes")
            extras = scene_stamps(path)
            frame_rows, batch, screen_rows = [], [], []
            first_image = None
            previous_screen = None
            self.set_progress(video_id, "Reading and encoding frames")
            for stamp, image in sampled_frames(path, extra_stamps=extras):
                if first_image is None:
                    first_image = image.copy()
                batch.append((stamp, image))
                found = self.read_screen(image)
                if found and previous_screen and found == previous_screen[3] and stamp - previous_screen[2] <= 2.1:
                    previous_screen[2] = stamp + 1
                elif found:
                    previous_screen = [video_id, stamp, stamp + 1, found]
                    screen_rows.append(previous_screen)
                if len(batch) == 8:
                    frame_rows.extend(self.embed_batch(video_id, batch))
                    batch.clear()
            if batch:
                frame_rows.extend(self.embed_batch(video_id, batch))
            if not frame_rows:
                raise ValueError("No video frames decoded")
            speech_rows = []
            if row and row[0]:
                self.set_progress(video_id, "Transcribing speech")
                speech_rows = self.transcribe(video_id, path)
            if first_image is not None:
                thumb_dir = self.data_dir / "thumbnails"
                thumb_dir.mkdir(exist_ok=True)
                first_image.thumbnail((480, 270))
                first_image.save(thumb_dir / f"{video_id}.jpg", quality=80)
            # Reindex in a single transaction so search never sees partially updated content.
            with self.db() as db:
                db.execute("DELETE FROM frames WHERE video_id=?", (video_id,))
                db.execute("DELETE FROM speech_fts WHERE video_id=?", (video_id,))
                db.execute("DELETE FROM screen_fts WHERE video_id=?", (video_id,))
                db.executemany("INSERT INTO frames VALUES(?,?,?)", frame_rows)
                db.executemany("INSERT INTO speech_fts(video_id,start,end,text) VALUES(?,?,?,?)", speech_rows)
                db.executemany("INSERT INTO screen_fts(video_id,start,end,text) VALUES(?,?,?,?)", screen_rows)
                db.execute("UPDATE videos SET status='ready',indexed_version=? WHERE id=?", (PIPELINE_VERSION, video_id))
                db.execute("UPDATE jobs SET status='done',lease_until=NULL,progress='Ready',error=NULL WHERE video_id=?", (video_id,))
        except Exception as exc:
            with self.db() as db:
                attempts = db.execute("SELECT attempts FROM jobs WHERE video_id=?", (video_id,)).fetchone()[0]
                retry = not isinstance(exc, ValueError) and attempts < 3
                db.execute("UPDATE videos SET status=?,error=? WHERE id=?",
                           ("queued" if retry else "failed", str(exc)[:300], video_id))
                db.execute("UPDATE jobs SET status=?,lease_until=NULL,error=?,progress=? WHERE video_id=?",
                           ("queued" if retry else "failed", str(exc)[:300],
                            "Retrying indexing" if retry else "Failed", video_id))
            raise

    def read_screen(self, image):
        if self.ocr_engine is not None:
            return self.ocr_engine(image)
        import pytesseract
        from pytesseract import Output
        if image.width < 1100:
            image = image.resize((image.width * 2, image.height * 2))
        data = pytesseract.image_to_data(image, output_type=Output.DICT, config="--psm 11")
        words = [word.strip() for word, confidence in zip(data["text"], data["conf"])
                 if word.strip() and float(confidence) >= 50]
        return " ".join(words)[:2000]

    def transcribe(self, video_id, path):
        if self.speech_engine is not None:
            return [(video_id, start, end, text) for start, end, text in self.speech_engine(path)]
        from faster_whisper import WhisperModel
        model_name = os.getenv("ASR_MODEL", "base.en")
        if self.asr_model is None:
            self.asr_model = WhisperModel(model_name, device="cpu", compute_type="int8",
                                          cpu_threads=2, download_root=str(self.data_dir / "asr_models"))
        segments, _ = self.asr_model.transcribe(str(path), beam_size=3, vad_filter=True)
        return [(video_id, round(s.start, 2), round(s.end, 2), s.text.strip())
                for s in segments if s.text.strip()]

    def list_videos(self):
        with self.db() as db:
            return [dict(r) for r in db.execute(
                "SELECT v.id,v.filename,v.status,v.error,v.duration_sec,v.created_at,v.indexed_version,j.progress "
                "FROM videos v LEFT JOIN jobs j ON j.video_id=v.id ORDER BY v.created_at DESC")]

    def thumbnail_path(self, video_id):
        path = self.data_dir / "thumbnails" / f"{video_id}.jpg"
        return path if path.exists() else None

    def delete_video(self, video_id):
        with self.db() as db:
            row = db.execute("SELECT path,status FROM videos WHERE id=?", (video_id,)).fetchone()
            if not row:
                return False
            if row[1] in {"queued", "indexing"}:
                raise ValueError("Wait until indexing finishes before deleting")
            db.execute("DELETE FROM frames WHERE video_id=?", (video_id,))
            db.execute("DELETE FROM speech_fts WHERE video_id=?", (video_id,))
            db.execute("DELETE FROM screen_fts WHERE video_id=?", (video_id,))
            db.execute("DELETE FROM jobs WHERE video_id=?", (video_id,))
            db.execute("DELETE FROM feedback WHERE video_id=?", (video_id,))
            db.execute("DELETE FROM videos WHERE id=?", (video_id,))
        Path(row[0]).unlink(missing_ok=True)
        (self.data_dir / "thumbnails" / f"{video_id}.jpg").unlink(missing_ok=True)
        return True

    def add_feedback(self, video_id, query, stamp, value):
        if value not in {"helpful", "wrong_moment"}:
            raise ValueError("Invalid feedback")
        with self.db() as db:
            db.execute("INSERT INTO feedback(video_id,query,stamp,value,created_at) VALUES(?,?,?,?,?)",
                       (video_id, query[:300], stamp, value, time.time()))

    @staticmethod
    def _fts_terms(query):
        words = re.findall(r"\w+", query, flags=re.UNICODE)[:12]
        return " OR ".join('"' + word + '"' for word in words)

    def search(self, query, limit=8):
        with self.db() as db:
            frames = db.execute(
                "SELECT f.video_id,f.stamp,f.vector,v.filename FROM frames f JOIN videos v ON v.id=f.video_id WHERE v.status='ready'"
            ).fetchall()
            terms = self._fts_terms(query)
            text_lists = []
            for table, source in (("speech_fts", "speech"), ("screen_fts", "screen")):
                matches = db.execute(
                    f"SELECT {table}.video_id,{table}.start,{table}.end,{table}.text,v.filename FROM {table} JOIN videos v ON v.id={table}.video_id "
                    f"WHERE {table} MATCH ? AND v.status='ready' ORDER BY rank LIMIT 50", (terms,),
                ).fetchall() if terms else []
                text_lists.append((source, matches))
        if not frames and not any(matches for _, matches in text_lists):
            return []
        ranked = []
        if frames:
            with self.lock:
                q = normalize(self.model().text(query).reshape(1, -1))[0]
            vectors = np.stack([np.frombuffer(row[2], dtype=np.float32) for row in frames])
            if vectors.shape[1] != len(q):
                raise RuntimeError("Index dimension mismatch; reindex all videos")
            scores = vectors @ q
            visual, seen = [], set()
            for index in np.argsort(-scores):
                row = frames[int(index)]
                key = (row[0], int(float(row[1]) // 5))
                if key in seen:
                    continue
                seen.add(key)
                visual.append((key, {"video_id": row[0], "filename": row[3],
                                     "timestamp_sec": float(row[1]), "source": "visual",
                                     "evidence": "Visual similarity", "visual_score": float(scores[index])}))
                if len(visual) >= 50:
                    break
            ranked.append(visual)
        for source, matches in text_lists:
            candidates, seen = [], set()
            for row in matches:
                stamp = float(row[1])
                key = (row[0], int(stamp // 5))
                if key in seen:
                    continue
                seen.add(key)
                candidates.append((key, {"video_id": row[0], "filename": row[4],
                                         "timestamp_sec": stamp, "source": source,
                                         "evidence": row[3]}))
            ranked.append(candidates)
        fused = {}
        for source_list in ranked:
            for rank, (key, hit) in enumerate(source_list, 1):
                if key not in fused:
                    fused[key] = {**hit, "score": 0.0, "sources": []}
                fused[key]["score"] += 1 / (60 + rank)
                fused[key]["sources"].append(hit["source"])
                if hit["source"] != "visual" and fused[key]["source"] == "visual":
                    fused[key].update(timestamp_sec=hit["timestamp_sec"], evidence=hit["evidence"], source=hit["source"])
        hits = sorted(fused.values(), key=lambda hit: -hit["score"])
        for hit in hits:
            hit["score"] = round(hit["score"], 5)
            hit["timestamp_sec"] = round(hit["timestamp_sec"], 2)
            hit["video_url"] = f"/api/media/{hit['video_id']}"
            hit["thumbnail_url"] = f"/api/thumbnail/{hit['video_id']}"
        return hits[:limit]
