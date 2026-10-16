"""Runtime settings, overridable through WT_* environment variables."""

import os
from dataclasses import dataclass


def _get(name, default):
    value = os.environ.get(name)
    return value if value not in (None, "") else default


def _get_int(name, default):
    try:
        return int(os.environ.get(name))
    except (TypeError, ValueError):
        return default


def _get_bool(name, default):
    value = os.environ.get(name)
    return default if value is None else value.lower() in ("1", "true", "yes", "on")


@dataclass(frozen=True)
class Settings:
    model_size: str = _get("WT_MODEL", "base")  # tiny..large-v3
    device: str = _get("WT_DEVICE", "auto")  # auto | cpu | cuda
    compute_type: str = _get("WT_COMPUTE_TYPE", "int8")  # int8 on cpu, float16 on gpu
    language: str | None = os.environ.get("WT_LANGUAGE") or None
    beam_size: int = _get_int("WT_BEAM_SIZE", 5)
    vad_filter: bool = _get_bool("WT_VAD", False)
    sample_rate: int = _get_int("WT_SAMPLE_RATE", 16000)
    chunk_seconds: int = _get_int("WT_CHUNK_SECONDS", 600)
    chunk_overlap_seconds: int = _get_int("WT_CHUNK_OVERLAP", 3)
    model_cache: str = _get(
        "WT_MODEL_CACHE",
        os.path.join(os.path.expanduser("~"), ".cache", "whisper-transcribe"),
    )


DEFAULT_SETTINGS = Settings()
