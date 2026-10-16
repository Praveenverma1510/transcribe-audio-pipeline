"""CLI entry point: python -m whisper_transcribe.cli <file> name."""

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

from .config import DEFAULT_SETTINGS
from .pipeline import TranscriptionPipeline


def build_parser():
    p = argparse.ArgumentParser(
        prog="whisper-transcribe",
        description="Transcribe audio (WAV/MP3/M4A/FLAC/...) to text with timestamps.",
    )
    p.add_argument("input", help="audio (or video) file to transcribe")
    p.add_argument("-o", "--output", help="output path, or '-' for stdout (default)")
    p.add_argument("-f", "--format", choices=["json", "srt", "txt"], default="json")
    p.add_argument("--model", help="tiny | base | small | medium | large-v3")
    p.add_argument("--device", help="cpu | cuda | auto")
    p.add_argument("--language", help="force a language code, e.g. en (default: auto)")
    p.add_argument("--beam-size", type=int)
    p.add_argument("--chunk-seconds", type=int, help="split files longer than this")
    p.add_argument("--no-vad", action="store_true", help="disable the silence filter")
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def render(transcript, fmt):
    if fmt == "srt":
        return transcript.to_srt()
    if fmt == "txt":
        return transcript.to_txt()
    return json.dumps(transcript.to_dict(), ensure_ascii=False, indent=2)


def main(argv=None):
    args = build_parser().parse_args(argv)

    # start from the env defaults, override only what the user passed
    settings = replace(
        DEFAULT_SETTINGS,
        model_size=args.model or DEFAULT_SETTINGS.model_size,
        device=args.device or DEFAULT_SETTINGS.device,
        language=args.language
        if args.language is not None
        else DEFAULT_SETTINGS.language,
        beam_size=args.beam_size or DEFAULT_SETTINGS.beam_size,
        chunk_seconds=args.chunk_seconds or DEFAULT_SETTINGS.chunk_seconds,
        vad_filter=not args.no_vad,
    )

    try:
        transcript = TranscriptionPipeline(settings).transcribe(args.input)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    output = render(transcript, args.format)
    if args.output and args.output != "-":
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(output, encoding="utf-8")
        print(f"wrote {out} ({len(transcript.segments)} segments)", file=sys.stderr)
    else:
        print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
