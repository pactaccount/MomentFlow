import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from search_core import SearchService
from evaluate import evaluate


class FakeEncoder:
    def images(self, images):
        return np.array([[1., 0.] if image.getpixel((0, 0))[0] > 100 else [0., 1.] for image in images])
    def text(self, query):
        return np.array([1., 0.])


class ProductFlowTest(unittest.TestCase):
    def test_index_search_and_moment_evaluation(self):
        with tempfile.TemporaryDirectory() as temp:
            service = SearchService(Path(temp), encoder=FakeEncoder())
            service.add_video("sample", "demo.mp4", Path(temp) / "demo.mp4")
            frames = [(0.0, Image.new("RGB", (4, 4), "blue")), (3.0, Image.new("RGB", (4, 4), "red"))]
            with patch("search_core.sampled_frames", return_value=iter(frames)):
                service.index_video("sample", Path(temp) / "demo.mp4")
            self.assertEqual(service.list_videos()[0]["status"], "ready")
            self.assertEqual(service.search("red object")[0]["timestamp_sec"], 3.0)
            report = evaluate(service, [{"query": "red object", "matches": [{"filename": "demo.mp4", "timestamp_sec": 3.0}]}])
            self.assertEqual(report["moment_recall_at_1"], 1.0)


if __name__ == "__main__":
    unittest.main()
