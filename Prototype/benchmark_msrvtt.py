"""Evaluate zero-shot video retrieval on a frozen 50-video MSR-VTT subset.

Download with:
  hf download hari31416/search-anything-benchmark --repo-type dataset \
    --include 'search-benchmark/manifest.json' \
    --include 'search-benchmark/media/*.mp4' \
    --local-dir runtime/benchmark/search_anything
"""
import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path

import numpy as np

from search_core import SearchService, normalize


def evaluate(queries, service, ranker=None):
    details = []
    for item in queries:
        started = time.perf_counter()
        moments = service.search(item["text"], limit=1000) if ranker is None else None
        elapsed = time.perf_counter() - started
        # One vote per video, scored by its best frame. No label is used in ranking.
        if ranker is None:
            seen = set()
            videos = []
            for moment in moments:
                if moment["video_id"] not in seen:
                    seen.add(moment["video_id"])
                    videos.append(moment["video_id"])
        else:
            videos = ranker(item["text"])
            elapsed = time.perf_counter() - started
        rank = next((i + 1 for i, video_id in enumerate(videos) if video_id in item["relevant"]), None)
        details.append({"query_id": item["id"], "query": item["text"], "rank": rank,
                        "top_video": videos[0] if videos else None,
                        "latency_sec": round(elapsed, 3)})
    n = len(details)
    return {"queries": n,
            "recall_at_1": sum(d["rank"] == 1 for d in details) / n,
            "recall_at_5": sum(d["rank"] is not None and d["rank"] <= 5 for d in details) / n,
            "recall_at_10": sum(d["rank"] is not None and d["rank"] <= 10 for d in details) / n,
            "mrr_at_10": sum(1 / d["rank"] for d in details if d["rank"] is not None and d["rank"] <= 10) / n,
            "median_query_latency_sec": statistics.median(d["latency_sec"] for d in details),
            "details": details}


def first_frame_ranker(service):
    """Ablation: compare the product with a single thumbnail per video."""
    with service.db() as db:
        rows = db.execute("SELECT video_id,stamp,vector FROM frames ORDER BY video_id,stamp").fetchall()
    first = {}
    for row in rows:
        first.setdefault(row[0], np.frombuffer(row[2], dtype=np.float32))
    ids = list(first)
    vectors = np.stack([first[video_id] for video_id in ids])

    def rank(query):
        vector = normalize(service.model().text(query).reshape(1, -1))[0]
        return [ids[i] for i in np.argsort(-(vectors @ vector))]

    return rank


def run(dataset_dir: Path, data_dir: Path):
    source = dataset_dir / "search-benchmark"
    manifest = json.loads((source / "manifest.json").read_text())
    assets = [x for x in manifest["assets"] if x["task"] == "video"]
    queries = [x for x in manifest["queries"] if x["task"] == "video"]
    if len(assets) != 50 or len(queries) != 50:
        raise ValueError("Expected the frozen 50-video / 50-query subset")
    service = SearchService(data_dir)
    present = {v["id"]: v for v in service.list_videos()}
    indexing = []
    for asset in assets:
        path = source / asset["path"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != asset["sha256"]:
            raise ValueError(f"Checksum mismatch: {path}")
        if asset["id"] in present and present[asset["id"]]["status"] == "ready":
            continue
        if asset["id"] not in present:
            service.add_video(asset["id"], path.name, path)
        started = time.perf_counter()
        service.index_video(asset["id"], path)
        indexing.append(round(time.perf_counter() - started, 2))
        status = next(v for v in service.list_videos() if v["id"] == asset["id"])
        if status["status"] != "ready":
            raise RuntimeError(f"Indexing failed for {path}: {status['error']}")
    thumbnail = first_frame_ranker(service)
    return {"dataset": "hari31416/search-anything-benchmark / MSR-VTT subset",
            "model": service.model_id, "candidate_videos": len(assets),
            "indexed_frames": service.frame_count(),
            "indexing_seconds_this_run": round(sum(indexing), 2),
            "first_frame_baseline_development": evaluate([q for q in queries if q["split"] == "dev"], service, thumbnail),
            "first_frame_baseline_held_out": evaluate([q for q in queries if q["split"] == "test"], service, thumbnail),
            "development": evaluate([q for q in queries if q["split"] == "dev"], service),
            "held_out": evaluate([q for q in queries if q["split"] == "test"], service)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, default=Path(__file__).resolve().parent / "runtime/benchmark/search_anything")
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parent / "runtime/msrvtt_index_final")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "evaluation_results/msrvtt_results.json")
    args = parser.parse_args()
    report = run(args.dataset_dir, args.data_dir)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps({k: ({x: y for x, y in v.items() if x != "details"} if isinstance(v, dict) else v)
                      for k, v in report.items()}, indent=2))
