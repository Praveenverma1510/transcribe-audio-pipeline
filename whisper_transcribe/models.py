"""Result types the pipeline works with."""

from dataclasses import dataclass, field
from typing import Any


def format_timestamp(seconds):
    """Seconds -> SRT timestamp (HH:MM:SS,mmm)."""
    seconds = max(0.0, float(seconds))
    hours = int(seconds // 3600)
    minutes = int(seconds % 3600 // 60)
    secs = int(seconds % 60)
    millis = int(round((seconds - int(seconds)) * 1000))
    if millis == 1000:  # rounding can carry into the next second
        secs += 1
        millis = 0
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


@dataclass
class Segment:
    start: float
    end: float
    text: str

    @property
    def duration(self):
        return max(0.0, self.end - self.start)

    def to_dict(self):
        return {
            "start": round(self.start, 3),
            "end": round(self.end, 3),
            "text": self.text.strip(),
        }


@dataclass
class Transcript:
    text: str
    segments: list[Segment] = field(default_factory=list)
    language: str | None = None
    duration: float = 0.0
    source: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self):
        return {
            "text": self.text.strip(),
            "language": self.language,
            "duration": round(self.duration, 3),
            "source": self.source,
            "segments": [s.to_dict() for s in self.segments],
            "meta": self.meta,
        }

    def to_srt(self):
        out = []
        for i, seg in enumerate(self.segments, 1):
            out.append(str(i))
            out.append(f"{format_timestamp(seg.start)} --> {format_timestamp(seg.end)}")
            out.append(seg.text.strip() or " ")
            out.append("")
        return "\n".join(out).strip() + "\n"

    def to_txt(self):
        return "\n".join(s.text.strip() for s in self.segments if s.text.strip())
