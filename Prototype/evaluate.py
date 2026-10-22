"""Evaluate the deployed multimodal ranking path on hand-labeled queries."""
import argparse
import json
import statistics
import time
from pathlib import Path

from product_service import PIPELINE_VERSION, ProductService


def evaluate(service, cases, tolerance_sec=2.0):
    details = []
    for case in cases:
        started = time.perf_counter()
        hits = service.search(case["query"], limit=20)
        elapsed = time.perf_counter() - started
        def relevant(hit):
            return any(
                (hit["video_id"] == label["video_id"] if "video_id" in label
                 else hit["filename"] == label["filename"])
                and abs(hit["timestamp_sec"] - label["timestamp_sec"]) <= tolerance_sec
                for label in case["matches"]
            )
        rank = next((i + 1 for i, hit in enumerate(hits) if relevant(hit)), None)
        details.append({"query": case["query"], "modality": case.get("modality", "unspecified"),
                        "rank": rank, "latency_sec": round(elapsed, 3),
                        "top_hit": hits[0] if hits else None})
    n = len(details)
    by_modality = {}
    for modality in sorted({d["modality"] for d in details}):
        group = [d for d in details if d["modality"] == modality]
        by_modality[modality] = {
            "num_queries": len(group),
            "moment_recall_at_1": sum(d["rank"] == 1 for d in group) / len(group),
            "moment_recall_at_5": sum(d["rank"] is not None and d["rank"] <= 5 for d in group) / len(group),
        }
    return {
        "pipeline": PIPELINE_VERSION,
        "visual_model": service.model_id,
        "num_queries": n,
        "timestamp_tolerance_sec": tolerance_sec,
        "moment_recall_at_1": sum(d["rank"] == 1 for d in details) / n,
        "moment_recall_at_5": sum(d["rank"] is not None and d["rank"] <= 5 for d in details) / n,
        "median_query_latency_sec": statistics.median(d["latency_sec"] for d in details),
        "by_modality": by_modality,
        "details": details,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path, help="JSON list of {query,matches:[{filename,timestamp_sec}]}")
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parent / "runtime")
    parser.add_argument("--output", type=Path, default=Path("evaluation_results.json"))
    args = parser.parse_args()
    cases = json.loads(args.manifest.read_text())
    if not cases or any(
        not c.get("query") or not c.get("matches") or
        any("timestamp_sec" not in label or not ("video_id" in label or "filename" in label)
            for label in c["matches"])
        for c in cases
    ):
        parser.error("Provide queries and matching video IDs or filenames with timestamps")
    report = evaluate(ProductService(args.data_dir), cases)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "details"}, indent=2))
