"""Media validation and timestamp-aware preparation for uploaded videos."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def probe_video(path: Path, max_duration: float = 1800, max_pixels: int = 3840 * 2160) -> dict:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_format", "-show_streams", "-of", "json", str(path)],
        capture_output=True, text=True, timeout=30, check=False,
    )
    if result.returncode:
        raise ValueError("Video could not be decoded")
    metadata = json.loads(result.stdout)
    video = next((stream for stream in metadata.get("streams", []) if stream.get("codec_type") == "video"), None)
    if not video:
        raise ValueError("No video stream found")
    duration = float(metadata.get("format", {}).get("duration") or video.get("duration") or 0)
    width, height = int(video.get("width") or 0), int(video.get("height") or 0)
    if duration <= 0 or duration > max_duration:
        raise ValueError(f"Video must be between 0 and {max_duration / 60:g} minutes")
    if width <= 0 or height <= 0 or width * height > max_pixels:
        raise ValueError("Video resolution is unsupported")
    return {
        "duration_sec": duration,
        "width": width,
        "height": height,
        "has_audio": any(s.get("codec_type") == "audio" for s in metadata.get("streams", [])),
    }


def scene_stamps(path: Path, max_stamps: int = 300) -> list[float]:
    """Add scene boundaries without allowing one video to create an unbounded index."""
    from scenedetect import ContentDetector, detect

    scenes = detect(str(path), ContentDetector(threshold=27.0), show_progress=False)
    return [start.get_seconds() for start, _ in scenes[1:max_stamps + 1]]


def sampled_frames(path: Path, interval: float = 1.0, max_duration: float = 1800.0,
                   extra_stamps: list[float] | None = None):
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise ValueError("Video cannot be decoded")
        fps, count = cap.get(cv2.CAP_PROP_FPS), cap.get(cv2.CAP_PROP_FRAME_COUNT)
        if not np.isfinite(fps) or fps <= 0 or count <= 0:
            raise ValueError("Invalid video duration or frame rate")
        duration = count / fps
        if duration > max_duration + 1:
            raise ValueError("Video exceeds duration limit")
        regular = np.arange(0, duration, interval).tolist()
        stamps = sorted(set(round(float(s), 2) for s in regular + (extra_stamps or []) if 0 <= s < duration))
        seen_actual = set()
        for stamp in stamps:
            cap.set(cv2.CAP_PROP_POS_MSEC, stamp * 1000)
            ok, frame = cap.read()
            if not ok:
                continue
            actual = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
            if not np.isfinite(actual) or actual < 0:
                actual = stamp
            actual = round(float(actual), 2)
            if actual in seen_actual:
                continue
            seen_actual.add(actual)
            height, width = frame.shape[:2]
            if max(height, width) > 768:
                scale = 768 / max(height, width)
                frame = cv2.resize(frame, (round(width * scale), round(height * scale)))
            yield actual, Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    finally:
        cap.release()
