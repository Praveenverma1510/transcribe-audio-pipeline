"""End-to-end: a file path in, a Transcript out."""

import logging
import tempfile
from pathlib import Path

from .audio import prepare, investigate
from .chunking import split
from .config import Settings
from .engine import WhisperEngine
from .models import Segment, Transcript

log = logging.getLogger(__name__)


class TranscriptionPipeline:
    def __init__(self, settings=None, engine=None):
        self.settings = settings or Settings()
        self.engine = engine or WhisperEngine(self.settings)

    def transcribe(self, path):
        source = Path(path)
        info = investigate(source)

        # intermediate files live in one temp dir that cleans itself up
        with tempfile.TemporaryDirectory(prefix="wt-work-") as tmp:
            workdir = Path(tmp)
            wav, _, converted = prepare(source, self.settings.sample_rate, workdir)
            log.info(
                "input=%s duration=%.2fs codec=%s converted=%s",
                source.name,
                info.duration,
                info.codec,
                converted,
            )

            if info.duration > self.settings.chunk_seconds:
                segments, language = self._transcribe_long(wav, workdir)
                mode = "chunked"
            else:
                result = self.engine.transcribe(wav, offset=0.0)
                segments, language = result.segments, result.language
                mode = "single"

        text = " ".join(s.text for s in segments).strip()
        log.info("segments=%d chars=%d language=%s", len(segments), len(text), language)
        return Transcript(
            text=text,
            segments=segments,
            language=language,
            duration=info.duration,
            source=str(source),
            meta={
                "mode": mode,
                "model": self.settings.model_size,
                "device": self.settings.device,
                "vad": self.settings.vad_filter,
            },
        )

    def _transcribe_long(self, wav, workdir):
        """The chunking overlap used to keep the sementic context of the audio, but it can also produce duplicate segments. We filter those out."""
        chunks = split(
            wav,
            chunk_seconds=self.settings.chunk_seconds,
            overlap_seconds=self.settings.chunk_overlap_seconds,
            workdir=workdir / "chunks",
        )

        segments = []
        language = None
        running_end = 0.0

        for chunk in chunks:
            result = self.engine.transcribe(chunk.path, offset=chunk.start)
            language = language or result.language
            for seg in result.segments:
                if seg.end <= running_end:
                    continue  # already covered by the previous chunk
                if seg.start < running_end:
                    seg = Segment(running_end, seg.end, seg.text)
                segments.append(seg)
                running_end = seg.end

        return segments, language
