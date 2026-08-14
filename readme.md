# infer-meeting-summarize

Self-hosted pipeline that turns a meeting recording into a Markdown summary with
speaker attribution. Audio never leaves the machine; everything runs locally on
GPU through vLLM.

## What it does

| Stage | Model | Answers |
|---|---|---|
| Transcription | `openai/whisper-large-v3` (vLLM) | *what* was said, and when |
| Diarization | `pyannote/speaker-diarization-3.1` | *who* speaks in each time range |
| Summarization | `Qwen/Qwen2.5-72B-Instruct-AWQ` (vLLM) | what it all *means* |

Whisper and pyannote cut the audio at different points, so each transcript
segment is assigned the speaker occupying most of its range; consecutive
segments from the same speaker are merged. The summary covers executive summary,
topics, decisions, action items and open questions.

Transcripts that fit in the summarizer's context are summarized in one pass.
Longer ones fall back to segment (topic boundaries) → extract (structured notes
per section, run concurrently) → reduce (fold notes down until they fit).

## Architecture

Three containers in `docker-compose.yaml`:

| Service | Port | Profile | Role |
|---|---|---|---|
| `vllm-asr` | 8001 | `transcribe` | Whisper behind an OpenAI-compatible API |
| `pipeline` | 8002 | `transcribe` | FastAPI app: ffmpeg, diarization, orchestration |
| `vllm-summarize` | 8003 | `summarize` | Qwen 72B (AWQ) behind an OpenAI-compatible API |

Profiles let you bring up only what you need; the two vLLM services split the
same GPU budget via `--gpu-memory-utilization` (0.15 ASR, 0.50 summarizer).
Both Dockerfiles build on `vllm/vllm-openai:latest` because that image already
ships a working PyTorch — `pipeline/Dockerfile` pins it through a constraints
file so pip cannot swap in generic wheels that lose GPU support.

## Setup

Requirements: Docker with the NVIDIA container runtime, a GPU with room for both
models (~3 GB Whisper, ~42 GB for the AWQ 72B plus KV cache), and a Hugging Face
token that has **accepted the terms** of `pyannote/speaker-diarization-3.1` (it
is gated; the pipeline fails at startup without it).

```bash
cp .env.example .env   # HF_TOKEN, plus HF_CACHE_DIR as an absolute path
                       # (compose does not expand ~ or $HOME in .env)

docker compose --profile transcribe --profile summarize up --build
docker compose --profile transcribe up --build   # transcription only
```

All services run with `HF_HUB_OFFLINE=1` and use only the local cache. To add a
model: remove that variable, start the service once to download, put it back.

## API

The pipeline listens on `http://localhost:8002`.

- **`GET /health`** — reports the configured ASR URL, diarization model and
  summarizer URL.
- **`POST /transcribe`** — transcript with speakers, no summarization (the
  `summarize` profile is not needed). Fields: `file` (any format ffmpeg
  decodes), and optionally `language`, `num_speakers`, or
  `min_speakers`/`max_speakers`. Returns `speakers`, `segments` (start, end,
  speaker, text) and a formatted `transcript`; a segment's speaker is `null`
  when pyannote found nobody speaking there.
- **`POST /summarize`** — summarizes a transcript you already have: `text`, plus
  optional `instructions` prepended to the prompt.
- **`POST /transcribe/summarize`** — the whole flow. Every field of
  `/transcribe`, plus `instructions` and `all` (default `false`; when true also
  returns speakers, segments and transcript).

```bash
curl -s -F "file=@meeting.m4a" -F "language=es" \
  http://localhost:8002/transcribe | jq -r .transcript

curl -s -F "file=@meeting.m4a" \
  -F "instructions=Focus on the commitments made by the infra team." \
  http://localhost:8002/transcribe/summarize | jq -r .summary
```

## Tuning

Set on the `pipeline` service; defaults are sized for a 32k-token summarizer
context.

| Variable | Default | Meaning |
|---|---|---|
| `SUMMARIZER_CONTEXT` | `32768` | **Must match** `--max-model-len` on `vllm-summarize`; decides single-pass vs. hierarchical |
| `SUMMARY_MAX_TOKENS` | `4096` | Output budget for the final summary |
| `NOTES_MAX_TOKENS` | `1536` | Output budget per section's notes |
| `SECTION_BUDGET_TOKENS` | `6000` | Max size of a leaf section |
| `SEGMENTATION_WINDOW_TOKENS` | `12000` | Window for the boundary-finding pass |
| `OUTLINE_MAX_TOKENS` | `600` | Ceiling on the outline included in each leaf call |
| `MAP_CONCURRENCY` | `4` | Concurrent extraction calls (batching is what makes a 72B worth running) |
| `SUMMARY_LANGUAGE` | *(empty)* | Empty follows the transcript's language |
| `ASR_TIMEOUT` / `SUMMARIZER_TIMEOUT` | `3600` / `1800` | Seconds |

Also raise `VLLM_MAX_AUDIO_CLIP_FILESIZE_MB` and
`VLLM_MAX_AUDIO_DECODE_DURATION_S` on `vllm-asr` together: the defaults
(25 MB / 600 s) target short clips, but a meeting arrives as one 16 kHz mono WAV
at roughly 115 MB per hour.

## Notes

- Audio is normalized to 16 kHz mono WAV first — what Whisper and pyannote use
  internally, and the vLLM image's libsndfile has no MP3 support.
- Speaker labels are anonymous (`SPEAKER_00`, …) unless a real name comes up.
- Transcription and diarization run in series; they compete for the same GPU.
- The diarization model is loaded once at startup, not per request.
