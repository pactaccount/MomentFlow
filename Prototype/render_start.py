"""Run the API and index worker in one Render Free web service."""
import os
import threading
import traceback

import uvicorn


def main():
    for name in ("HF_SPACE_ID", "HF_TOKEN"):
        if not os.getenv(name):
            raise RuntimeError(f"Set {name} in the Render service Environment page")

    # Importing the app initializes SQLite before the worker opens its connection.
    from main import app
    from worker import main as run_worker

    def worker_entry():
        try:
            run_worker()
        except BaseException:
            traceback.print_exc()
            # A dead worker must make the service unhealthy instead of leaving
            # uploads stuck in the queue behind an apparently healthy API.
            os._exit(1)

    threading.Thread(target=worker_entry, name="index-worker", daemon=True).start()
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "10000")), workers=1)


if __name__ == "__main__":
    main()
