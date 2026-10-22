"""Single durable indexing worker; run as a separate container command."""
import os
import threading
import time
from pathlib import Path

from product_service import ProductService
from telemetry import JOBS, JOB_SECONDS, QUEUE_DEPTH, start_metrics


def main():
    start_metrics()
    service = ProductService(Path(os.getenv("DATA_DIR", Path(__file__).resolve().parent / "runtime")))
    while True:
        with service.db() as db:
            QUEUE_DEPTH.set(db.execute("SELECT COUNT(*) FROM jobs WHERE status='queued'").fetchone()[0])
        job = service.claim_job()
        if job is None:
            time.sleep(2)
            continue
        video_id, path = job
        stop = threading.Event()

        def heartbeat():
            while not stop.wait(30):
                service.set_progress(video_id, None)

        thread = threading.Thread(target=heartbeat, daemon=True)
        thread.start()
        started = time.perf_counter()
        try:
            service.process_job(video_id, path)
            JOBS.labels("success").inc()
        except Exception as exc:
            JOBS.labels("failure").inc()
            print(f"Indexing failed for {video_id}: {exc}", flush=True)
        finally:
            JOB_SECONDS.observe(time.perf_counter() - started)
            stop.set()
            thread.join(timeout=2)


if __name__ == "__main__":
    main()
