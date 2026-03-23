from __future__ import annotations

import sys
import tempfile
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

    def test_get_predictor_does_not_reuse_predictor_instance(self) -> None:
        builds: list[object] = []
        fake_builder = ModuleType("sam3.model_builder")

        def build_sam3_video_predictor(**kwargs):
            predictor = {"build_num": len(builds) + 1, "kwargs": kwargs}
            builds.append(predictor)
            return predictor

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
                first = pipeline._get_predictor(settings)
                second = pipeline._get_predictor(settings)

        self.assertEqual(len(builds), 2)
        self.assertIsNot(first, second)


class ClickSessionPrepTest(unittest.TestCase):
    def test_prime_click_prompt_session_seeds_empty_cache_for_all_frames(self) -> None:
        predictor = ModuleType("fake_predictor")
        predictor._ALL_INFERENCE_STATES = {
            "session-1": {
                "state": {
                    "cached_frame_outputs": {2: {"keep": True}},
                }
            }
        }

        pipeline._prime_click_prompt_session(predictor, "session-1", 4)

        cached = predictor._ALL_INFERENCE_STATES["session-1"]["state"]["cached_frame_outputs"]
        self.assertEqual(sorted(cached.keys()), [0, 1, 2, 3])
        self.assertEqual(cached[2], {"keep": True})
        self.assertEqual(cached[0], {})

    def test_mark_session_frame_has_outputs_sets_flag(self) -> None:
        predictor = ModuleType("fake_predictor")
        predictor._ALL_INFERENCE_STATES = {
            "session-1": {
                "state": {
                    "previous_stages_out": [None, None, None],
                }
            }
        }

        pipeline._mark_session_frame_has_outputs(predictor, "session-1", 1)

        previous = predictor._ALL_INFERENCE_STATES["session-1"]["state"]["previous_stages_out"]
        self.assertIsNone(previous[0])
        self.assertEqual(previous[1], "_THIS_FRAME_HAS_OUTPUTS_")


class RunTrackFlowTest(unittest.TestCase):
    def test_run_track_uses_predictor_instance(self) -> None:
        settings = Settings(
            runs_dir=Path("/tmp/sam3-video-tests"),
            sam3_device="cuda",
            sam3_checkpoint_path=None,
            sam3_load_from_hf=True,
            sam3_compile=False,
            sam3_apply_temporal_disambiguation=True,
        )

        class FakePredictor:
            def __init__(self) -> None:
                self.request_types: list[str] = []
                self.closed_session_id: str | None = None

            def handle_request(self, request):
                self.request_types.append(request["type"])
                if request["type"] == "start_session":
                    return {"session_id": "session-1"}
                if request["type"] == "add_prompt":
                    return {"frame_index": 0, "outputs": {"out_obj_ids": [], "out_binary_masks": []}}
                if request["type"] == "close_session":
                    self.closed_session_id = request["session_id"]
                    return {"is_success": True}
                raise AssertionError(f"unexpected request: {request}")

            def handle_stream_request(self, request):
                self.request_types.append(request["type"])
                if False:
                    yield request
                return

        class FakeWriter:
            def write(self, frame) -> None:
                _ = frame

            def release(self) -> None:
                return None

        predictor = FakePredictor()

        def fake_extract_frame0(video_path: Path, frame0_path: Path) -> tuple[int, int]:
            _ = video_path
            frame0_path.parent.mkdir(parents=True, exist_ok=True)
            frame0_path.write_bytes(b"frame0")
            return (640, 480)

        def fake_decode_frames(video_path: Path, frames_dir: Path) -> tuple[int, int, float, int]:
            _ = video_path
            frames_dir.mkdir(parents=True, exist_ok=True)
            (frames_dir / "000000.jpg").write_bytes(b"frame")
            return (640, 480, 30.0, 1)

        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / "run"
            video_path = Path(tmpdir) / "input.mp4"
            video_path.write_bytes(b"video")

            with patch.object(pipeline, "extract_frame0", side_effect=fake_extract_frame0):
                with patch.object(pipeline, "decode_frames", side_effect=fake_decode_frames):
                    with patch.object(pipeline, "_get_predictor", return_value=predictor):
                        with patch.object(pipeline, "_predictor_gpu_name", return_value="fake-gpu"):
                            with patch.object(pipeline, "_mask_for_obj", return_value=None):
                                with patch.object(pipeline, "_overlay_frame", side_effect=lambda frame, mask: frame):
                                    with patch.object(pipeline.cv2, "VideoWriter", return_value=FakeWriter(), create=True):
                                        with patch.object(pipeline.cv2, "VideoWriter_fourcc", return_value=0, create=True):
                                            with patch.object(pipeline.cv2, "IMREAD_COLOR", 1, create=True):
                                                with patch.object(pipeline.cv2, "imread", return_value=object(), create=True):
                                                    pipeline.run_sam3_video_track(
                                                        settings=settings,
                                                        video_path=video_path,
                                                        source_filename="input.mp4",
                                                        run_dir=run_dir,
                                                        prompt=pipeline.PromptSpec(mode="click", click_xy=(10, 10)),
                                                        progress_cb=lambda progress, message: None,
                                                    )

        self.assertEqual(predictor.request_types[0], "start_session")
        self.assertIn("add_prompt", predictor.request_types)
        self.assertIn("propagate_in_video", predictor.request_types)
        self.assertEqual(predictor.closed_session_id, "session-1")


if __name__ == "__main__":
    unittest.main()
