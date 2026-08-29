import tempfile
import unittest
from pathlib import Path

from ane_transformers.bnnsgraph import BNNSGraphAccelerator


class TestBNNSGraphAccelerator(unittest.TestCase):

    def test_backend_name(self):
        self.assertEqual(
            BNNSGraphAccelerator.backend_name,
            "apple-bnnsgraph",
        )

    def test_missing_model_rejected(self):
        with self.assertRaises(FileNotFoundError):
            BNNSGraphAccelerator("/definitely/not/a/model.mlpackage")

    def test_model_artifact_accepted(self):
        with tempfile.TemporaryDirectory() as tmp:
            model_path = Path(tmp) / "test.mlpackage"
            model_path.mkdir()

            accelerator = BNNSGraphAccelerator(model_path)

            self.assertEqual(accelerator.path, model_path)
            self.assertEqual(
                accelerator.describe()["status"],
                "artifact-ready",
            )


if __name__ == "__main__":
    unittest.main()
