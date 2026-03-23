from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

if "cv2" not in sys.modules:
    sys.modules["cv2"] = ModuleType("cv2")

if "torch" not in sys.modules:
    fake_torch = ModuleType("torch")
    fake_torch.cuda = ModuleType("torch.cuda")
    fake_torch.cuda.is_available = lambda: False
    sys.modules["torch"] = fake_torch
    sys.modules["torch.cuda"] = fake_torch.cuda

if "pycocotools" not in sys.modules:
    fake_pycocotools = ModuleType("pycocotools")
    fake_mask = ModuleType("pycocotools.mask")
    fake_pycocotools.mask = fake_mask
    sys.modules["pycocotools"] = fake_pycocotools
    sys.modules["pycocotools.mask"] = fake_mask

if "numpy" not in sys.modules:
    sys.modules["numpy"] = ModuleType("numpy")

from sam3_video_poc.config import Settings
from sam3_video_poc import pipeline
from sam3_video_poc.targeting import normalize_click_point, pick_target_obj_id


class PipelineUtilsTest(unittest.TestCase):
    def test_normalize_click_point(self) -> None:
        self.assertEqual(normalize_click_point(50, 25, 200, 100), [[0.25, 0.25]])

    def test_pick_target_obj_id_prefers_highest_score(self) -> None:
        outputs = {
            "out_obj_ids": [4, 9, 2],
            "out_probs": [0.6, 0.95, 0.5],
        }
        obj_id, score = pick_target_obj_id(outputs)
        self.assertEqual(obj_id, 9)
        self.assertAlmostEqual(score or 0.0, 0.95, places=6)

    def test_pick_target_obj_id_empty(self) -> None:
        self.assertEqual(pick_target_obj_id({}), (None, None))


class PredictorInitTest(unittest.TestCase):
    def setUp(self) -> None:
        pipeline._predictor_cache.clear()

    def tearDown(self) -> None:
        pipeline._predictor_cache.clear()

    def test_get_predictor_omits_load_from_hf_kwarg(self) -> None:
        calls: dict[str, object] = {}
        fake_builder = ModuleType("sam3.model_builder")

        def build_sam3_video_predictor(**kwargs):
            calls.update(kwargs)
            return object()

        fake_builder.build_sam3_video_predictor = build_sam3_video_predictor
        settings = Settings(
            runs_dir=Path("/tmp/sam3-video-tests"),
            sam3_device="cuda",
            sam3_checkpoint_path=None,
            sam3_load_from_hf=True,
            sam3_compile=False,
            sam3_apply_temporal_disambiguation=True,
        )

        with patch("torch.cuda.is_available", return_value=True):
            with patch.dict(sys.modules, {"sam3.model_builder": fake_builder}, clear=False):
                predictor = pipeline._get_predictor(settings)

        self.assertIsNotNone(predictor)
        self.assertNotIn("load_from_HF", calls)
        self.assertIsNone(calls["checkpoint_path"])
        self.assertTrue(calls["apply_temporal_disambiguation"])
        self.assertFalse(calls["compile"])

    def test_get_predictor_requires_checkpoint_when_hf_disabled(self) -> None:
        settings = Settings(
            runs_dir=Path("/tmp/sam3-video-tests"),
            sam3_device="cuda",
            sam3_checkpoint_path=None,
            sam3_load_from_hf=False,
            sam3_compile=False,
            sam3_apply_temporal_disambiguation=True,
        )

        with patch("torch.cuda.is_available", return_value=True):
            with self.assertRaisesRegex(
                RuntimeError,
                "SAM3_CHECKPOINT_PATH required when SAM3_LOAD_FROM_HF=0",
            ):
                pipeline._get_predictor(settings)


if __name__ == "__main__":
    unittest.main()
