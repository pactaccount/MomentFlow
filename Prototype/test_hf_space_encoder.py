import json
import unittest

import numpy as np
from PIL import Image

from hf_space_encoder import HuggingFaceSpaceEncoder
from search_core import MODEL_ID, MODEL_REVISION


class FakeSpaceClient:
    def __init__(self):
        self.calls = []

    def predict(self, *args, api_name):
        self.calls.append(api_name)
        if api_name == "/health":
            return json.dumps({"model": MODEL_ID, "revision": MODEL_REVISION})
        if api_name == "/embed_text":
            return json.dumps({"vectors": [[3.0, 4.0]]})
        if api_name == "/embed_images":
            return json.dumps({"vectors": [[3.0, 4.0] for _ in json.loads(args[0])]})
        raise AssertionError(api_name)


class HuggingFaceSpaceEncoderTest(unittest.TestCase):
    def test_text_and_image_contract(self):
        client = FakeSpaceClient()
        encoder = HuggingFaceSpaceEncoder("owner/space", "test-token", client=client)
        self.assertTrue(np.allclose(encoder.text("red ball"), [0.6, 0.8]))
        self.assertTrue(np.allclose(encoder.images([Image.new("RGB", (8, 8), "red")]), [[0.6, 0.8]]))
        self.assertEqual(client.calls, ["/health", "/embed_text", "/embed_images"])

    def test_rejects_wrong_revision(self):
        class WrongRevision(FakeSpaceClient):
            def predict(self, *args, api_name):
                return json.dumps({"model": MODEL_ID, "revision": "wrong"})

        encoder = HuggingFaceSpaceEncoder("owner/space", "test-token", client=WrongRevision())
        with self.assertRaises(RuntimeError):
            encoder.text("red ball")


if __name__ == "__main__":
    unittest.main()
