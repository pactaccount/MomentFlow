"""Single durable indexing worker; run beside the local API."""
import os
import threading
import time
from pathlib import Path

from product_service import ProductService


def main():
    service = ProductService(Path(os.getenv("DATA_DIR", Path(__file__).resolve().parent / "runtime")))
    while True:
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
        try:
            service.process_job(video_id, path)
        except Exception as exc:
            print(f"Indexing failed for {video_id}: {exc}", flush=True)
        finally:
            stop.set()
            thread.join(timeout=2)


if __name__ == "__main__":
    main()
