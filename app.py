from __future__ import annotations

import logging
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import cv2
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from pydantic import BaseModel, Field, model_validator

from sam3_video_poc.config import load_settings
from sam3_video_poc.pipeline import PromptSpec, extract_frame0, run_sam3_video_track
from sam3_video_poc.ui import INDEX_HTML


@dataclass
class JobState:
    job_id: str
    run_dir: Path
    input_video_path: Path
    source_filename: str
    width: int
    height: int
    fps: float
    num_frames: int
    state: str
    progress: float
    message: str
    error: str | None


class PromptPayload(BaseModel):
    mode: Literal["click", "text"]
    x: int | None = Field(default=None)
    y: int | None = Field(default=None)
    text: str | None = Field(default=None)

    @model_validator(mode="after")
    def validate_shape(self) -> "PromptPayload":
        if self.mode == "click":
            if self.x is None or self.y is None:
                raise ValueError("click mode requires x and y")
            if self.text not in {None, ""}:
                raise ValueError("click mode does not accept text")
        else:
            if not (self.text or "").strip():
                raise ValueError("text mode requires text")
            if self.x is not None or self.y is not None:
                raise ValueError("text mode does not accept x/y")
        return self


app = FastAPI(title="SAM3 Video Skier Tracker POC")
settings = load_settings()
settings.runs_dir.mkdir(parents=True, exist_ok=True)
logger = logging.getLogger("uvicorn.error")

_jobs: dict[str, JobState] = {}
_jobs_lock = threading.Lock()


def _video_meta(path: Path) -> tuple[int, int, float, int]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"Could not open video: {path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    if fps <= 0:
        fps = 30.0
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    return width, height, fps, num_frames


def _update_job(job_id: str, **kwargs) -> None:
    with _jobs_lock:
        job = _jobs.get(job_id)
        if job is None:
            return
        for key, value in kwargs.items():
            setattr(job, key, value)


def _get_job(job_id: str) -> JobState:
    with _jobs_lock:
        job = _jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="job not found")
        return job


def _job_payload(job: JobState) -> dict:
    payload = {
        "job_id": job.job_id,
        "state": job.state,
        "progress": round(job.progress, 4),
        "message": job.message,
    }
    if job.error:
        payload["error"] = job.error
    if job.state == "done":
        payload["results"] = {
            "overlay_url": f"/files/{job.job_id}/overlay.mp4",
            "masks_url": f"/files/{job.job_id}/masks.json",
            "frame0_url": f"/files/{job.job_id}/frame0.jpg",
        }
    return payload


def _run_job(job_id: str, prompt: PromptSpec) -> None:
    job = _get_job(job_id)
    logger.info(
        "[sam3-video][%s] job start mode=%s source=%s",
        job_id,
        prompt.mode,
        job.source_filename,
    )

    def progress_cb(progress: float, message: str) -> None:
        _update_job(job_id, progress=max(0.0, min(1.0, progress)), message=message)

    try:
        video_info, _ = run_sam3_video_track(
            settings=settings,
            video_path=job.input_video_path,
            source_filename=job.source_filename,
            run_dir=job.run_dir,
            prompt=prompt,
            progress_cb=progress_cb,
        )
        _update_job(
            job_id,
            width=video_info.width,
            height=video_info.height,
            fps=video_info.fps,
            num_frames=video_info.num_frames,
            state="done",
            progress=1.0,
            message="complete",
            error=None,
        )
        logger.info("[sam3-video][%s] job complete", job_id)
    except Exception as exc:  # noqa: BLE001
        _update_job(
            job_id,
            state="failed",
            message="processing failed",
            error=str(exc),
        )
        logger.exception("[sam3-video][%s] job failed: %s", job_id, exc)


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    return HTMLResponse(content=INDEX_HTML)


@app.get("/readyz")
def readyz() -> JSONResponse:
    return JSONResponse({"ok": True})


@app.get("/healthz")
def healthz() -> JSONResponse:
    return JSONResponse(
        {
            "ok": True,
            "cuda_available": torch.cuda.is_available(),
            "sam3_device": settings.sam3_device,
            "sam3_load_from_hf": settings.sam3_load_from_hf,
            "sam3_checkpoint_path": str(settings.sam3_checkpoint_path) if settings.sam3_checkpoint_path else None,
        }
    )


@app.post("/upload")
async def upload(video: UploadFile = File(...)) -> JSONResponse:
    job_id = uuid.uuid4().hex[:12]
    run_dir = settings.runs_dir / job_id
    run_dir.mkdir(parents=True, exist_ok=True)

    filename = video.filename or "input.mp4"
    suffix = Path(filename).suffix or ".mp4"
    input_video_path = run_dir / f"input{suffix}"

    with input_video_path.open("wb") as out:
        while chunk := await video.read(1024 * 1024):
            out.write(chunk)

    frame0_path = run_dir / "frame0.jpg"
    try:
        width, height, fps, num_frames = _video_meta(input_video_path)
        extract_frame0(input_video_path, frame0_path)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"video decode failed: {exc}") from exc

    job = JobState(
        job_id=job_id,
        run_dir=run_dir,
        input_video_path=input_video_path,
        source_filename=filename,
        width=width,
        height=height,
        fps=fps,
        num_frames=num_frames,
        state="ready",
        progress=0.0,
        message="uploaded",
        error=None,
    )

    with _jobs_lock:
        _jobs[job_id] = job

    return JSONResponse(
        {
            "job_id": job_id,
            "frame0_url": f"/files/{job_id}/frame0.jpg",
            "width": width,
            "height": height,
        }
    )


@app.post("/prompt/{job_id}")
def prompt(job_id: str, payload: PromptPayload) -> JSONResponse:
    job = _get_job(job_id)

    if job.state == "processing":
        raise HTTPException(status_code=409, detail="job already processing")
    if job.state == "done":
        return JSONResponse({"ok": True, "already_done": True})

    if payload.mode == "click":
        x = max(0, min(int(payload.x or 0), job.width - 1))
        y = max(0, min(int(payload.y or 0), job.height - 1))
        prompt_spec = PromptSpec(mode="click", click_xy=(x, y))
    else:
        prompt_spec = PromptSpec(mode="text", text=(payload.text or "").strip())

    _update_job(job_id, state="processing", progress=0.01, message="starting", error=None)
    worker = threading.Thread(target=_run_job, args=(job_id, prompt_spec), daemon=True)
    worker.start()
    return JSONResponse({"ok": True})


@app.get("/status/{job_id}")
def status(job_id: str) -> JSONResponse:
    return JSONResponse(_job_payload(_get_job(job_id)))


@app.get("/files/{job_id}/{name}")
def files(job_id: str, name: str):
    if name not in {"overlay.mp4", "masks.json", "frame0.jpg"}:
        raise HTTPException(status_code=404, detail="file not found")

    job = _get_job(job_id)
    path = (job.run_dir / name).resolve()
    if not path.exists() or path.parent != job.run_dir.resolve():
        raise HTTPException(status_code=404, detail="file not found")

    media_type = {
        "overlay.mp4": "video/mp4",
        "masks.json": "application/json",
        "frame0.jpg": "image/jpeg",
    }[name]
    return FileResponse(path=path, media_type=media_type, filename=name)
