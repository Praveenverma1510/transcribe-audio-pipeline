"""Cut long audio into overlapping windows so memory stays bounded."""

from dataclasses import dataclass
from pathlib import Path

from .audio import FFMPEG, AudioError, _run, investigate


@dataclass
class Chunk:
    path: Path
    start: float
    end: float


def plan(duration, chunk_seconds, overlap_seconds):
    """Return [(start, end), ...] windows covering `duration`."""
    if duration <= 0:
        return [(0.0, 0.0)]
    if duration <= chunk_seconds:
        return [(0.0, duration)]

    step = max(1, chunk_seconds - overlap_seconds)
    spans = []
    start = 0.0
    while start < duration:
        end = min(start + chunk_seconds, duration)
        spans.append((start, end))
        if end >= duration:
            break
        start += step
    return spans


def split(path, chunk_seconds, overlap_seconds, workdir):
    """Write one WAV per window and remember where each sits in the original."""
    info = investigate(path)
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    chunks = []
    for i, (start, end) in enumerate(
        plan(info.duration, chunk_seconds, overlap_seconds)
    ):
        dst = workdir / f"chunk_{i:04d}.wav"
        _run(
            [
                FFMPEG,
                "-y",
                "-ss",
                f"{start:.3f}",  # seek before -i is much faster
                "-i",
                str(path),
                "-t",
                f"{end - start:.3f}",
                "-ac",
                "1",
                "-ar",
                "16000",
                "-acodec",
                "pcm_s16le",
                str(dst),
            ]
        )
        chunks.append(Chunk(dst, start, end))

    if not chunks:
        raise AudioError(f"produced no chunks for {path}")
    return chunks
