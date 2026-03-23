from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    runs_dir: Path
    sam3_device: str
    sam3_checkpoint_path: Path | None
    sam3_load_from_hf: bool
    sam3_compile: bool
    sam3_apply_temporal_disambiguation: bool


def load_settings() -> Settings:
    root = Path(__file__).resolve().parent.parent
    checkpoint_raw = os.getenv("SAM3_CHECKPOINT_PATH", "").strip()
    checkpoint_path = Path(checkpoint_raw).expanduser().resolve() if checkpoint_raw else None
    return Settings(
        runs_dir=Path(os.getenv("RUNS_DIR", root / "runs")).expanduser().resolve(),
        sam3_device=os.getenv("SAM3_DEVICE", "cuda").strip().lower() or "cuda",
        sam3_checkpoint_path=checkpoint_path,
        sam3_load_from_hf=_env_bool("SAM3_LOAD_FROM_HF", True),
        sam3_compile=_env_bool("SAM3_COMPILE", False),
        sam3_apply_temporal_disambiguation=_env_bool(
            "SAM3_APPLY_TEMPORAL_DISAMBIGUATION", True
        ),
    )
