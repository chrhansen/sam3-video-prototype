from __future__ import annotations

import gc
import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Callable, Literal

import cv2
import numpy as np
import torch
from pycocotools import mask as mask_utils

from .config import Settings
from .targeting import normalize_click_point, resolve_text_target_obj_id


ProgressCb = Callable[[float, str], None]
PromptMode = Literal["click", "text"]
logger = logging.getLogger("uvicorn.error")


@dataclass(frozen=True)
class PromptSpec:
    mode: PromptMode
    click_xy: tuple[int, int] | None = None
    text: str | None = None


@dataclass(frozen=True)
class VideoInfo:
    source_filename: str
    width: int
    height: int
    fps: float
    num_frames: int


@dataclass(frozen=True)
class RunArtifacts:
    frame0_path: Path
    overlay_path: Path
    masks_path: Path


_predictor_cache: dict[tuple[str, bool, bool], object] = {}
_predictor_lock = Lock()


def _require_cuda(settings: Settings) -> None:
    if settings.sam3_device != "cuda":
        raise RuntimeError(
            f"SAM3 Video official predictor only supports CUDA here; got SAM3_DEVICE={settings.sam3_device!r}"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required for SAM3 Video, but torch.cuda.is_available() is false")


def _resolve_predictor_checkpoint_path(settings: Settings) -> str | None:
    if settings.sam3_checkpoint_path:
        return str(settings.sam3_checkpoint_path)
    if settings.sam3_load_from_hf:
        return None
    raise RuntimeError("SAM3_CHECKPOINT_PATH required when SAM3_LOAD_FROM_HF=0")


def _get_predictor(settings: Settings):
    _require_cuda(settings)
    checkpoint_path = _resolve_predictor_checkpoint_path(settings)
    with _predictor_lock:
        try:
            from sam3.model_builder import build_sam3_video_predictor
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                f"SAM3 import failed: {type(exc).__name__}: {exc}"
            ) from exc

        predictor = build_sam3_video_predictor(
            checkpoint_path=checkpoint_path,
            apply_temporal_disambiguation=settings.sam3_apply_temporal_disambiguation,
            compile=settings.sam3_compile,
        )
        return predictor


def extract_frame0(video_path: Path, frame0_path: Path) -> tuple[int, int]:
    cap = cv2.VideoCapture(str(video_path))
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise RuntimeError("Failed to decode first frame")
    frame0_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(frame0_path), frame)
    height, width = frame.shape[:2]
    return width, height


def decode_frames(video_path: Path, frames_dir: Path) -> tuple[int, int, float, int]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    if fps <= 0:
        fps = 30.0

    frames_dir.mkdir(parents=True, exist_ok=True)
    idx = 0
    width = 0
    height = 0

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        if idx == 0:
            height, width = frame.shape[:2]
        cv2.imwrite(str(frames_dir / f"{idx:06d}.jpg"), frame)
        idx += 1

    cap.release()

    if idx == 0:
        raise RuntimeError("No frames decoded from video")

    return width, height, fps, idx


def _mask_for_obj(outputs: dict[str, object], target_obj_id: int) -> np.ndarray | None:
    obj_ids = np.asarray(outputs.get("out_obj_ids", []), dtype=np.int64)
    masks = np.asarray(outputs.get("out_binary_masks", []))
    if obj_ids.size == 0 or masks.size == 0:
        return None

    matches = np.where(obj_ids == target_obj_id)[0]
    if matches.size == 0:
        return None

    mask = masks[int(matches[0])]
    if mask.ndim > 2:
        mask = mask.squeeze()
    mask = (mask > 0).astype(np.uint8)
    return mask if mask.any() else None


def _bbox_xyxy(mask: np.ndarray) -> list[int] | None:
    ys, xs = np.where(mask > 0)
    if xs.size == 0 or ys.size == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


def _encode_coco_rle(mask: np.ndarray) -> dict[str, object]:
    rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    counts = rle["counts"]
    if isinstance(counts, bytes):
        counts = counts.decode("utf-8")
    return {
        "encoding": "coco_rle",
        "size_hw": [int(rle["size"][0]), int(rle["size"][1])],
        "counts": counts,
    }


def _overlay_frame(frame_bgr: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
    if mask is None:
        return frame_bgr

    out = frame_bgr.copy()
    tint = np.array([20, 220, 20], dtype=np.uint8)
    masked = mask > 0
    out[masked] = (0.55 * out[masked] + 0.45 * tint).astype(np.uint8)

    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        cv2.drawContours(out, contours, -1, (0, 0, 0), 1, lineType=cv2.LINE_AA)
    return out


def _predictor_gpu_name() -> str:
    if not torch.cuda.is_available():
        return "none"
    return torch.cuda.get_device_name(torch.cuda.current_device())


def _prime_click_prompt_session(predictor: object, session_id: str, num_frames: int) -> None:
    sessions = getattr(predictor, "_ALL_INFERENCE_STATES", None)
    if not isinstance(sessions, dict):
        return
    session = sessions.get(session_id)
    if not isinstance(session, dict):
        return
    inference_state = session.get("state")
    if not isinstance(inference_state, dict):
        return
    cached_outputs = inference_state.setdefault("cached_frame_outputs", {})
    if not isinstance(cached_outputs, dict):
        return
    for frame_idx in range(num_frames):
        cached_outputs.setdefault(frame_idx, {})


def _mark_session_frame_has_outputs(predictor: object, session_id: str, frame_idx: int) -> None:
    sessions = getattr(predictor, "_ALL_INFERENCE_STATES", None)
    if not isinstance(sessions, dict):
        return
    session = sessions.get(session_id)
    if not isinstance(session, dict):
        return
    inference_state = session.get("state")
    if not isinstance(inference_state, dict):
        return
    previous_stages_out = inference_state.get("previous_stages_out")
    if not isinstance(previous_stages_out, list):
        return
    if 0 <= frame_idx < len(previous_stages_out):
        previous_stages_out[frame_idx] = "_THIS_FRAME_HAS_OUTPUTS_"


def _release_predictor_resources() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_sam3_video_track(
    *,
    settings: Settings,
    video_path: Path,
    source_filename: str,
    run_dir: Path,
    prompt: PromptSpec,
    progress_cb: ProgressCb,
) -> tuple[VideoInfo, RunArtifacts]:
    total_start = time.perf_counter()
    run_dir.mkdir(parents=True, exist_ok=True)
    frame0_path = run_dir / "frame0.jpg"
    masks_path = run_dir / "masks.json"
    overlay_path = run_dir / "overlay.mp4"
    frames_dir = run_dir / "frames_orig"

    job_ref = run_dir.name
    logger.info(
        "[sam3-video][%s] start source=%s prompt_mode=%s load_from_hf=%s checkpoint=%s",
        job_ref,
        source_filename,
        prompt.mode,
        settings.sam3_load_from_hf,
        settings.sam3_checkpoint_path or "<hf>",
    )

    stage_start = time.perf_counter()
    progress_cb(0.03, "extracting frame0")
    extract_frame0(video_path, frame0_path)
    logger.info(
        "[sam3-video][%s] frame0 extracted in %.2fs",
        job_ref,
        time.perf_counter() - stage_start,
    )

    stage_start = time.perf_counter()
    progress_cb(0.08, "decoding frames")
    width, height, fps, num_frames = decode_frames(video_path, frames_dir)
    logger.info(
        "[sam3-video][%s] decoded frames=%d size=%dx%d fps=%.2f in %.2fs",
        job_ref,
        num_frames,
        width,
        height,
        fps,
        time.perf_counter() - stage_start,
    )

    video_info = VideoInfo(
        source_filename=source_filename,
        width=width,
        height=height,
        fps=fps,
        num_frames=num_frames,
    )

    session_id: str | None = None
    predictor = None
    outputs_per_frame: dict[int, dict[str, object]] = {}

    try:
        stage_start = time.perf_counter()
        progress_cb(0.16, "loading sam3 video")
        predictor = _get_predictor(settings)
        logger.info(
            "[sam3-video][%s] predictor ready in %.2fs gpu=%s",
            job_ref,
            time.perf_counter() - stage_start,
            _predictor_gpu_name(),
        )

        stage_start = time.perf_counter()
        progress_cb(0.24, "starting session")
        session_response = predictor.handle_request(
            {
                "type": "start_session",
                "resource_path": str(video_path),
            }
        )
        session_id = str(session_response["session_id"])
        logger.info(
            "[sam3-video][%s] session=%s started in %.2fs",
            job_ref,
            session_id,
            time.perf_counter() - stage_start,
        )
        if prompt.mode == "click":
            _prime_click_prompt_session(predictor, session_id, num_frames)

        stage_start = time.perf_counter()
        progress_cb(0.30, "adding prompt")
        if prompt.mode == "click":
            if prompt.click_xy is None:
                raise RuntimeError("click prompt missing click_xy")
            click_x, click_y = prompt.click_xy
            prompt_response = predictor.handle_request(
                {
                    "type": "add_prompt",
                    "session_id": session_id,
                    "frame_index": 0,
                    "points": normalize_click_point(click_x, click_y, width, height),
                    "point_labels": [1],
                    "obj_id": 1,
                }
            )
        else:
            if not prompt.text:
                raise RuntimeError("text prompt missing text")
            prompt_response = predictor.handle_request(
                {
                    "type": "add_prompt",
                    "session_id": session_id,
                    "frame_index": 0,
                    "text": prompt.text,
                }
            )

        prompt_frame_idx = int(prompt_response["frame_index"])
        prompt_outputs = prompt_response["outputs"]
        outputs_per_frame[prompt_frame_idx] = prompt_outputs
        _mark_session_frame_has_outputs(predictor, session_id, prompt_frame_idx)
        logger.info(
            "[sam3-video][%s] prompt added frame=%d in %.2fs",
            job_ref,
            prompt_frame_idx,
            time.perf_counter() - stage_start,
        )

        stage_start = time.perf_counter()
        progress_cb(0.36, "tracking video")
        yielded_frames = 0
        for response in predictor.handle_stream_request(
            {
                "type": "propagate_in_video",
                "session_id": session_id,
                "propagation_direction": "forward",
                "start_frame_index": prompt_frame_idx,
            }
        ):
            frame_idx = int(response["frame_index"])
            outputs_per_frame[frame_idx] = response["outputs"]
            yielded_frames += 1
            progress = 0.36 + 0.36 * ((frame_idx + 1) / max(1, num_frames))
            progress_cb(min(progress, 0.72), f"tracking frame {frame_idx + 1}/{num_frames}")
            if yielded_frames == 1 or yielded_frames % 50 == 0:
                elapsed = time.perf_counter() - stage_start
                fps_track = yielded_frames / max(elapsed, 1e-6)
                logger.info(
                    "[sam3-video][%s] propagate frame=%d/%d yielded=%d speed_fps=%.3f",
                    job_ref,
                    frame_idx + 1,
                    num_frames,
                    yielded_frames,
                    fps_track,
                )

        if prompt.mode == "click":
            target_obj_id = 1
            resolved_frame_idx = 0
            selected_score = 1.0
        else:
            target_obj_id, resolved_frame_idx, selected_score = resolve_text_target_obj_id(
                prompt_outputs,
                outputs_per_frame,
            )

        logger.info(
            "[sam3-video][%s] selected target_obj_id=%d resolved_frame=%d score=%s",
            job_ref,
            target_obj_id,
            resolved_frame_idx,
            "n/a" if selected_score is None else f"{selected_score:.4f}",
        )

        stage_start = time.perf_counter()
        progress_cb(0.78, "building outputs")
        frame_paths = sorted(frames_dir.glob("*.jpg"), key=lambda path: int(path.stem))
        writer = cv2.VideoWriter(
            str(overlay_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )

        frames_json: list[dict[str, object]] = []
        for idx, frame_path in enumerate(frame_paths):
            frame = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
            if frame is None:
                raise RuntimeError(f"Failed to read frame: {frame_path}")

            outputs = outputs_per_frame.get(idx, {})
            mask = _mask_for_obj(outputs, target_obj_id)
            bbox = _bbox_xyxy(mask) if mask is not None else None
            present = mask is not None and bbox is not None

            frames_json.append(
                {
                    "frame_idx": idx,
                    "present": present,
                    "bbox_xyxy": bbox,
                    "bbox_score": selected_score if present else None,
                    "mask": _encode_coco_rle(mask) if present and mask is not None else None,
                    "track": None,
                }
            )
            writer.write(_overlay_frame(frame, mask))

        writer.release()

        output = {
            "schema_version": "masktrack.v1",
            "video": {
                "source_filename": source_filename,
                "width": width,
                "height": height,
                "fps": fps,
                "num_frames": num_frames,
            },
            "processing": {
                "method": "sam3_video",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "notes": (
                    "SAM3 Video prototype; single-target output. "
                    "Text mode keeps the top-scoring detected object."
                ),
            },
            "target": {
                "target_id": "target-0",
                "resolved_obj_id": target_obj_id,
                "resolved_frame_idx": resolved_frame_idx,
                "init": {
                    "type": prompt.mode,
                    "frame_idx": 0,
                    "click_xy": list(prompt.click_xy) if prompt.click_xy is not None else None,
                    "text_prompt": prompt.text if prompt.mode == "text" else None,
                },
            },
            "frames": frames_json,
        }
        masks_path.write_text(json.dumps(output, separators=(",", ":")), encoding="utf-8")

        present_count = sum(1 for frame in frames_json if frame["present"])
        logger.info(
            "[sam3-video][%s] rendered overlay+json in %.2fs present_frames=%d/%d total=%.2fs",
            job_ref,
            time.perf_counter() - stage_start,
            present_count,
            num_frames,
            time.perf_counter() - total_start,
        )
    finally:
        if session_id:
            try:
                predictor.handle_request({"type": "close_session", "session_id": session_id})
            except Exception as exc:  # noqa: BLE001
                logger.warning("[sam3-video][%s] close_session failed: %s", job_ref, exc)
        predictor = None
        _release_predictor_resources()

    return video_info, RunArtifacts(
        frame0_path=frame0_path,
        overlay_path=overlay_path,
        masks_path=masks_path,
    )
