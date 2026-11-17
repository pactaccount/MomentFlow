import unittest
import tempfile
from pathlib import Path
import numpy as np
from fastapi.testclient import TestClient

from render_gateway import app as render_app
from search_core import SearchService, normalize


class TestMomentFlowCore(unittest.TestCase):
    def test_vector_normalization(self):
        v = np.array([3.0, 4.0], dtype=np.float32)
        norm_v = normalize(v)
        self.assertAlmostEqual(float(np.linalg.norm(norm_v)), 1.0, places=5)

    def test_search_service_db_initialization(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            service = SearchService(Path(tmpdir))
            with service.db() as db:
                tables = [r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
                self.assertIn("videos", tables)
                self.assertIn("frames", tables)

    def test_render_gateway_routes(self):
        client = TestClient(render_app)
        res = client.get("/")
        self.assertEqual(res.status_code, 200)

        res_health = client.get("/api/health")
        self.assertEqual(res_health.status_code, 200)
        self.assertEqual(res_health.json()["service"], "MomentFlow-Render-Gateway")

        res_set = client.post("/api/set_backend", json={"backend_url": "https://test-tunnel.trycloudflare.com"})
        self.assertEqual(res_set.status_code, 200)
        self.assertEqual(res_set.json()["kaggle_backend_url"], "https://test-tunnel.trycloudflare.com")


if __name__ == "__main__":
    unittest.main()
