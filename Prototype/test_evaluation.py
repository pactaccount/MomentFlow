import unittest

from evaluate import evaluate


class SearchStub:
    model_id = "test-encoder"

    def search(self, query, limit=20):
        if query == "spoken phrase":
            return [{"video_id": "other", "filename": "other.mp4", "timestamp_sec": 1},
                    {"video_id": "right", "filename": "right.mp4", "timestamp_sec": 8}]
        return [{"video_id": "right", "filename": "right.mp4", "timestamp_sec": 3}]


class EvaluationTest(unittest.TestCase):
    def test_video_and_timestamp_ranks_are_reported_by_modality(self):
        cases = [
            {"query": "visible scene", "modality": "visual", "matches": [{"video_id": "right", "timestamp_sec": 3}]},
            {"query": "spoken phrase", "modality": "speech", "matches": [{"video_id": "right", "timestamp_sec": 8}]},
            {"query": "wrong moment", "modality": "screen", "matches": [{"video_id": "right", "timestamp_sec": 30}]},
        ]
        report = evaluate(SearchStub(), cases)
        self.assertEqual([row["rank"] for row in report["details"]], [1, 2, None])
        self.assertEqual(report["moment_recall_at_1"], 1 / 3)
        self.assertEqual(report["moment_recall_at_5"], 2 / 3)
        self.assertEqual(report["by_modality"]["speech"]["moment_recall_at_5"], 1)
        self.assertEqual(report["by_modality"]["screen"]["moment_recall_at_5"], 0)
