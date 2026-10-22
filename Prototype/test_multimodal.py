import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from product_service import ProductService


class FakeEncoder:
    def images(self, images):
        return np.array([[1., 0.] if image.getpixel((0, 0))[0] > 100 else [0., 1.] for image in images])

    def text(self, query):
        return np.array([1., 0.])


class HostedFlowTest(unittest.TestCase):
    def test_transient_failure_retries_then_stops(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "demo.mp4"
            path.write_bytes(b"fixture")
            service = ProductService(Path(temp), encoder=FakeEncoder())
            service.add_video("demo", "demo.mp4", path)
            with patch("product_service.scene_stamps", side_effect=OSError("temporary decoder error")):
                for attempt in range(3):
                    self.assertEqual(service.claim_job()[0], "demo")
                    with self.assertRaises(OSError):
                        service.process_job("demo", path)
                    self.assertEqual(service.list_videos()[0]["status"],
                                     "queued" if attempt < 2 else "failed")
            self.assertIsNone(service.claim_job())

    def test_job_recovery_and_multimodal_evidence(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "demo.mp4"
            path.write_bytes(b"fixture")
            service = ProductService(Path(temp), encoder=FakeEncoder(),
                                     speech=lambda _: [(8.0, 11.0, "click billing settings")],
                                     ocr=lambda image: "Billing Settings" if image.getpixel((0, 0))[0] > 100 else "")
            service.add_video("demo", "demo.mp4", path, {"duration_sec": 12, "has_audio": True}, "sha-test")
            self.assertEqual(service.claim_job()[0], "demo")
            with service.db() as db:
                db.execute("UPDATE jobs SET lease_until=? WHERE video_id='demo'", (time.time() - 1,))
            self.assertEqual(service.claim_job()[0], "demo")
            frames = [(0.0, Image.new("RGB", (10, 10), "blue")),
                      (8.0, Image.new("RGB", (10, 10), "red"))]
            with patch("product_service.scene_stamps", return_value=[]), patch("product_service.sampled_frames", return_value=iter(frames)):
                service.process_job("demo", path)
            self.assertEqual(service.list_videos()[0]["status"], "ready")
            hits = service.search("billing settings")
            self.assertEqual(hits[0]["video_id"], "demo")
            self.assertIn("speech", hits[0]["sources"])
            self.assertIn("screen", hits[0]["sources"])
            self.assertEqual(hits[0]["timestamp_sec"], 8.0)
            self.assertTrue(service.thumbnail_path("demo").exists())
            self.assertEqual(service.find_sha("sha-test"), "demo")
            self.assertTrue(service.delete_video("demo"))
            self.assertEqual(service.search("billing"), [])
