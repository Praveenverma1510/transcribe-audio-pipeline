"""Probe and normalize audio to 16 kHz mono PCM WAV with ffmpeg."""

import json
import shutil
import subprocess
import tempfile
import wave
from dataclasses import dataclass
from pathlib import Path


class AudioError(RuntimeError):
    pass


FFMPEG = shutil.which("ffmpeg") or "ffmpeg"
FFPROBE = shutil.which("ffprobe") or "ffprobe"


@dataclass
class AudioInfo:
    path: Path
    duration: float
    sample_rate: int
    channels: int
    codec: str

    @property
    def whisper_ready(self):
        return self.sample_rate == 16000 and self.channels == 1


def _run(cmd):
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise AudioError(f"{' '.join(cmd)} failed:\n{proc.stderr.strip()}")
    return proc.stdout


def investigate(path):
    """Duration , sample rate , channels , codec we get from the first audio stream."""
    path = Path(path)
    if not path.exists():
        raise AudioError(f"audio file not found: {path}")

    out = _run(
        [
            FFPROBE,
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=sample_rate,channels,codec_name:format=duration",
            "-of",
            "json",
            str(path),
        ]
    )
    data = json.loads(out or "{}")
    streams = data.get("streams") or []
    if not streams:
        raise AudioError(f"no audio stream in {path}")

    s = streams[0]
    return AudioInfo(
        path=path,
        duration=float(data.get("format", {}).get("duration") or 0.0),
        sample_rate=int(s.get("sample_rate") or 0),
        channels=int(s.get("channels") or 0),
        codec=s.get("codec_name") or "unknown",
    )


def to_whisper_wav(src, dst, sample_rate=16000):
    """Transcode anything ffmpeg can read into mono 16-bit PCM WAV."""
    src, dst = Path(src), Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    _run(
        [
            FFMPEG,
            "-y",
            "-i",
            str(src),
            "-vn",  # drop any video stream
            "-ac",
            "1",  # mono
            "-ar",
            str(sample_rate),  # 16 kHz
            "-acodec",
            "pcm_s16le",
            str(dst),
        ]
    )
    return dst


def prepare(path, sample_rate=16000, workdir=None):
    """Return (wav_path, info, converted). Reuses the file if it's already fine."""
    info = investigate(path)
    src = Path(path)
    if (
        src.suffix.lower() == ".wav"
        and info.sample_rate == sample_rate
        and info.channels == 1
    ):
        return src, info, False

    workdir = Path(workdir) if workdir else Path(tempfile.mkdtemp(prefix="wt-audio-"))
    dst = workdir / f"{src.stem}.{sample_rate // 1000}k.wav"
    return to_whisper_wav(src, dst, sample_rate), info, True


def load_wav_array(path, sample_rate=16000):
    """Load a 16-bit PCM WAV into a float32 array scaled to [-1, 1].

    faster-whisper takes an array directly and skips its own PyAV decoder, so we
    decode here instead of depending on the av build.
    """
    import numpy as np

    with wave.open(str(path), "rb") as wf:
        if wf.getframerate() != sample_rate:
            raise AudioError(f"expected {sample_rate} Hz, got {wf.getframerate()}")
        if wf.getnchannels() != 1:
            raise AudioError("expected mono")
        if wf.getsampwidth() != 2:
            raise AudioError("expected 16-bit PCM")
        frames = wf.readframes(wf.getnframes())

    return np.frombuffer(frames, dtype="<i2").astype("float32") / 32768.0
