"""Adapter around faster-whisper. The only module that imports it."""

from dataclasses import dataclass

from .audio import load_wav_array
from .config import Settings
from .models import Segment


@dataclass
class EngineResult:
    segments: list[Segment]
    language: str | None = None
    language_probability: float | None = None


class WhisperEngine:
    def __init__(self, settings=None):
        self.settings = settings or Settings()
        self._model = None  # built on first use, it's expensive

    def _load(self):
        if self._model is None:
            from faster_whisper import WhisperModel

            self._model = WhisperModel(
                self.settings.model_size,
                device=self.settings.device,
                compute_type=self.settings.compute_type,
                download_root=self.settings.model_cache,
            )
        return self._model

    def transcribe(
        self, wav_path, *, language=None, beam_size=None, vad_filter=None, offset=0.0
    ):
        """Transcribe a 16k mono WAV. `offset` shifts times onto the full timeline."""
        audio = load_wav_array(wav_path, self.settings.sample_rate)
        stream, info = self._load().transcribe(
            audio,
            language=language if language is not None else self.settings.language,
            beam_size=beam_size if beam_size is not None else self.settings.beam_size,
            vad_filter=vad_filter
            if vad_filter is not None
            else self.settings.vad_filter,
        )

        segments = []
        for s in stream:  # generator, decodes as we iterate
            text = s.text.strip()
            if text:
                segments.append(Segment(s.start + offset, s.end + offset, text))

        return EngineResult(
            segments, info.language, getattr(info, "language_probability", None)
        )
