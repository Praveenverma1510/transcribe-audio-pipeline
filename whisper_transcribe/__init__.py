"""whisper-transcribe: a small Whisper transcription pipeline."""

from .config import DEFAULT_SETTINGS, Settings
from .engine import EngineResult, WhisperEngine
from .models import Segment, Transcript
from .pipeline import TranscriptionPipeline

__all__ = [
    "TranscriptionPipeline",
    "WhisperEngine",
    "EngineResult",
    "Settings",
    "DEFAULT_SETTINGS",
    "Segment",
    "Transcript",
]
