"""Reproducible zero-shot check on the public Ball Sports Key-Action dataset.

Download with:
  hf download Chris-davis-thor/Ball_sports_movements --repo-type dataset \
    --local-dir runtime/benchmark/ball_sports
"""
import argparse
import json
import statistics
import time
from pathlib import Path

from search_core import SearchService


def run(dataset_dir: Path, data_dir: Path):
    metadata = [json.loads(line) for line in (dataset_dir / "data/metadata.jsonl").read_text().splitlines() if line]
    service = SearchService(data_dir)
    present = {v["id"]: v for v in service.list_videos()}
    indexing = []
    for item in metadata:
        path = dataset_dir / "data" / item["file_name"]
        if not path.is_file():
            raise FileNotFoundError(path)
        if item["id"] in present and present[item["id"]]["status"] == "ready":
            continue
        if item["id"] not in present:
            service.add_video(item["id"], path.name, path)
        start = time.perf_counter()
        service.index_video(item["id"], path)
        indexing.append({"video": item["id"], "seconds": round(time.perf_counter() - start, 2)})
        result = next(v for v in service.list_videos() if v["id"] == item["id"])
        if result["status"] != "ready":
            raise RuntimeError(f"Indexing failed for {path}: {result['error']}")

    details = []
    for item in metadata:
        for segment in item["segments"]:
            # Published segment descriptions, with no training or query tuning.
            query = segment["description"]
            start = time.perf_counter()
            hits = service.search(query, limit=20)
            elapsed = time.perf_counter() - start
            video_rank = next((i + 1 for i, h in enumerate(hits) if h["video_id"] == item["id"]), None)
            moment_rank = next((i + 1 for i, h in enumerate(hits)
                                if h["video_id"] == item["id"]
                                and segment["start"] <= h["timestamp_sec"] <= segment["end"]), None)
            details.append({"query": query, "video": item["id"],
                            "segment_sec": [segment["start"], segment["end"]],
                            "video_rank": video_rank, "moment_rank": moment_rank,
                            "latency_sec": round(elapsed, 3), "top_hit": hits[0] if hits else None})

    n = len(details)
    return {
        "dataset": "Chris-davis-thor/Ball_sports_movements",
        "model": service.model_id,
        "videos": len(metadata), "segments": n,
        "indexed_frames": service.frame_count(),
        "video_recall_at_1": sum(d["video_rank"] == 1 for d in details) / n,
        "video_recall_at_5": sum(d["video_rank"] is not None and d["video_rank"] <= 5 for d in details) / n,
        "segment_recall_at_1": sum(d["moment_rank"] == 1 for d in details) / n,
        "segment_recall_at_5": sum(d["moment_rank"] is not None and d["moment_rank"] <= 5 for d in details) / n,
        "median_query_latency_sec": statistics.median(d["latency_sec"] for d in details),
        "indexing": indexing, "details": details,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, default=Path(__file__).resolve().parent / "runtime/benchmark/ball_sports")
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parent / "runtime")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "evaluation_results/ball_sports_results.json")
    args = parser.parse_args()
    report = run(args.dataset_dir, args.data_dir)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "details"}, indent=2))
