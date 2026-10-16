# whisper-transcribe

Give it an audio file, get back the text with a timestamp for every segment.

I used faster-whisper for this. It's free, MIT licensed, and it runs Whisper
models locally on your own machine, so there's no account, no API key and no
per-minute billing. ffmpeg does the audio decoding.

Walking through it:

```
audio file -> [audio.py] normalize to 16 kHz mono WAV -> [chunking.py] split if long
                                                            |
                                  [engine.py] whisper decode <--+   (offset added per chunk)
                                                            |
                                      [pipeline.py] stitch back into one timeline
                                                            |
                                           Transcript {text, language, segments[]}
                                                            |
                                      [cli.py]  JSON / SRT / TXT
```

## Install

```
pip install -r requirements.txt
```

You also need ffmpeg somewhere on your PATH.

## Running it

```
# json with timestamps (default)
python -m whisper_transcribe.cli samples\harvard.wav

# save the json instead of printing it
python -m whisper_transcribe.cli samples\harvard.wav -o transcript.json

# srt subtitles
python -m whisper_transcribe.cli samples\harvard.mp3 -f srt -o out.srt
```

The output looks like this:

```json
{
  "text": "The stale smell of old beer lingers. It takes heat to bring out the odor. A cold dip restores health and zest. A salt pickle tastes fine with ham. Tacos al pastor are my favorite. A zestful food is the hot cross bun.",
  "language": "en",
  "duration": 18.356,
  "source": "samples\\harvard.wav",
  "segments": [
    {
      "start": 0.91,
      "end": 4.51,
      "text": "The stale smell of old beer lingers."
    },
    {
      "start": 4.51,
      "end": 7.03,
      "text": "It takes heat to bring out the odor."
    },
    {
      "start": 7.03,
      "end": 9.95,
      "text": "A cold dip restores health and zest."
    },
    {
      "start": 9.95,
      "end": 12.67,
      "text": "A salt pickle tastes fine with ham."
    },
    {
      "start": 12.67,
      "end": 15.11,
      "text": "Tacos al pastor are my favorite."
    },
    {
      "start": 15.11,
      "end": 17.59,
      "text": "A zestful food is the hot cross bun."
    }
  ],
  "meta": {
    "mode": "single",
    "model": "base",
    "device": "auto",
    "vad": true
  }
}
```

## The files

| File | What it does |
|------|--------------|
| `audio.py`    | Takes any input file and normalizes it to 16 kHz mono PCM WAV with ffmpeg. |
| `chunking.py` | Decides where to cut long audio and writes the chunks. |
| `engine.py`   | The only file that imports faster-whisper. |
| `pipeline.py` | Runs the whole thing: normalize, chunk if needed, transcribe, stitch. |
| `models.py`   | `Segment` and `Transcript`, plus the JSON/SRT/TXT rendering. |
| `config.py`   | Settings like model size, language and chunk length. |
| `cli.py`      | Command line entry point. |

## Some notes on the decisions I made

**Why faster-whisper.** It's free, it runs offline once the model is downloaded,
and it uses CTranslate2 so I don't have to pull in PyTorch just to run inference
on a CPU. The model files come from Hugging Face on first use and get cached.

**Formats are ffmpeg's problem, not mine.** Every input, whether it's WAV, MP3,
M4A, FLAC, OGG or the audio track of a video, goes through one function that
converts it to 16 kHz mono 16-bit WAV, which is the format Whisper wants.
Everything after that point just deals with a WAV, so I never wrote per-format
code. The one bit of cleverness here is a check up front: if the file is already
16 kHz mono WAV, I skip the conversion and use it as-is. ffprobe runs before that
to read the duration, sample rate and channel count, and it also catches a broken
file before the model gets loaded.

**I read the WAV myself instead of letting faster-whisper do it.** Its normal path
decodes audio through PyAV. I had a broken av install on Windows and lost time to
it, so I decode the PCM WAV with Python's own wave module and pass a float32
numpy array to the model. It's a few lines and it removes a binary I don't want
to depend on. The only downside is that the engine expects WAV, but the pipeline
already guarantees that.

**Long files get split so I don't have to hold them in memory.** Anything longer
than chunk_seconds (10 minutes by default) is cut into roughly 6 minute windows
with 3 seconds of overlap. Chunks are cheap to retry on their own, and memory
doesn't grow with file length. Each chunk is transcribed with a start offset so
its timestamps line up with the original file, and then pipeline.py joins them
with a running cursor, dropping anything the previous chunk already covered.
Timestamps come out ordered and non-overlapping.

I'll be honest about the weak spot: the overlap de-duplication is a heuristic
based on time, not text, so you can occasionally see a repeated word right at a
seam. Cutting at a detected silence, or using word level timestamps, is the
proper fix. I've left it as a known limitation rather than pretending it's done.

**The engine is swappable.** pipeline.py talks to `Segment` and `Transcript`,
which are my own classes, and takes an engine as a constructor argument. So the
only file that knows faster-whisper exists is engine.py. I could drop in a mock
for tests, or a cloud API, or a different model, without touching anything else.
engine.py also consumes faster-whisper's segments as a generator, so segments
arrive as they're decoded rather than all at the end.

## If this went further

The current thing is a CLI. Next step would be an HTTP API in front of it. Since
transcription is slow, the upload shouldn't wait for it: POST /transcribe saves
the file, creates a job and returns a job id, and a small pool of workers picks
jobs off a queue and does the transcription. Cap the workers (Whisper is heavy on
CPU and memory, so ten at once just makes everything slower) and have each worker
load the model once and reuse it. The client polls GET /jobs/{id} until it's done.

For storage I'd put the audio in object storage like S3 and the transcripts in
Postgres, one row per job with the segments as JSONB. For retries, only retry the
failures that could actually succeed (timeouts, out of memory) a couple of times,
and fail fast on corrupt or unsupported files. A heartbeat on running jobs so a
dead worker doesn't leave a job stuck forever.

## The two questions from the brief

**How do you handle different audio formats?**
I don't handle them separately. ffmpeg converts anything it can read into 16 kHz
mono 16-bit WAV in one place, and that's the only format the rest of the code and
the model ever see. That also means a new format needs no new code.

**How do you deal with long audio files?**
I split them. Files over the threshold are cut into overlapping windows, each
window is transcribed with its time offset applied, and the windows are stitched
back together with a running cursor so the final timestamps are ordered and don't
overlap. Memory stays flat and one bad chunk doesn't take down the whole file.
