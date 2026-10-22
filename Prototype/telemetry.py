"""Private Prometheus metrics with bounded labels; no queries or video IDs."""
from __future__ import annotations

import os
import threading

from prometheus_client import Counter, Gauge, Histogram, start_http_server

HTTP_REQUESTS = Counter("momentflow_http_requests_total", "API requests", ["route", "method", "status"])
HTTP_SECONDS = Histogram("momentflow_http_request_seconds", "API latency", ["route", "method"])
SEARCH_RESULTS = Histogram("momentflow_search_results", "Search result count", buckets=(0, 1, 2, 5, 10, 20))
JOBS = Counter("momentflow_index_jobs_total", "Indexing attempts", ["outcome"])
JOB_SECONDS = Histogram("momentflow_index_job_seconds", "Indexing time in seconds", buckets=(10, 30, 60, 120, 300, 600, 1800, 3600))
QUEUE_DEPTH = Gauge("momentflow_queue_depth", "Queued indexing jobs")

_lock = threading.Lock()
_started = False


def start_metrics():
    global _started
    port = int(os.getenv("METRICS_PORT", "0"))
    if not port:
        return
    with _lock:
        if not _started:
            start_http_server(port, addr="0.0.0.0")
            _started = True
