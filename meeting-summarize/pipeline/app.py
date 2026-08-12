"""Transcription with speaker attribution.

Combines two models that solve different halves of the problem:
  - Whisper (served by vLLM) says WHAT was said and in which time range.
  - pyannote says WHO speaks in each range, but not what they say.

The two are merged by temporal overlap: each Whisper segment is assigned the
speaker who occupies the most time within that range.
"""

import logging
import os
import subprocess
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pyannote.audio import Pipeline

ASR_URL = os.environ.get("ASR_URL", "http://vllm-asr:8000/v1/audio/transcriptions")
ASR_MODEL = os.environ.get("ASR_MODEL", "openai/whisper-large-v3")
DIARIZATION_MODEL = os.environ.get("DIARIZATION_MODEL", "pyannote/speaker-diarization-3.1")
HF_TOKEN = os.environ.get("HF_TOKEN")

SUMMARIZER_URL = os.environ.get("SUMMARIZER_URL", "http://vllm-summarize:8000/v1/chat/completions")
SUMMARIZER_MODEL = os.environ.get("SUMMARIZER_MODEL", "Qwen/Qwen2.5-72B-Instruct-AWQ")

# A long meeting can take several minutes: the ASR request is not streamed.
ASR_TIMEOUT = float(os.environ.get("ASR_TIMEOUT", "3600"))
SUMMARIZER_TIMEOUT = float(os.environ.get("SUMMARIZER_TIMEOUT", "1800"))

SYSTEM_PROMPT = """You summarize meeting transcripts.

The transcript is labeled by speaker (SPEAKER_00, SPEAKER_01, ...) with timestamps.
Speaker labels are anonymous: use them as-is unless the transcript reveals a name.

Produce a summary in Markdown format, Dont omit any important information and do not invent details.
"""

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("pipeline")

state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the model here rather than per request: it costs ~10s we don't want to pay every time.
    if not HF_TOKEN:
        raise RuntimeError(
            "HF_TOKEN is missing. pyannote/speaker-diarization-3.1 is a gated model: "
            "its terms must be accepted on huggingface.co with the same account as the token."
        )

    logger.info("Loading %s...", DIARIZATION_MODEL)
    # pyannote 3.x uses use_auth_token; 4.x renamed it to token.
    try:
        pipeline = Pipeline.from_pretrained(DIARIZATION_MODEL, token=HF_TOKEN)
    except TypeError:
        pipeline = Pipeline.from_pretrained(DIARIZATION_MODEL, use_auth_token=HF_TOKEN)
    if pipeline is None:
        raise RuntimeError(
            f"{DIARIZATION_MODEL} returned None. This almost always means the token "
            "has not accepted the model's terms on the Hub."
        )

    if torch.cuda.is_available():
        pipeline.to(torch.device("cuda"))
        logger.info("Diarization running on GPU.")
    else:
        logger.warning("No CUDA available: diarization will run on CPU and be slow.")

    state["diarization"] = pipeline
    yield
    state.clear()


app = FastAPI(title="meeting-summarize pipeline", lifespan=lifespan)


def to_wav(src: Path, dst: Path) -> None:
    """Normalize to 16 kHz mono WAV.

    Two reasons: it is what Whisper and pyannote use internally, and the vLLM
    image ships a libsndfile without MP3 support, so sending it an MP3 fails.
    """
    subprocess.run(
        ["ffmpeg", "-nostdin", "-loglevel", "error", "-y",
         "-i", str(src), "-ar", "16000", "-ac", "1", str(dst)],
        check=True,
    )


async def transcribe(wav: Path, language: str | None) -> list[dict]:
    """Return Whisper's segments with their timestamps."""
    data = {
        "model": ASR_MODEL,
        "response_format": "verbose_json",
        # The bracketed alias is what vLLM expects for this list.
        "timestamp_granularities[]": "segment",
    }
    if language:
        data["language"] = language

    async with httpx.AsyncClient(timeout=ASR_TIMEOUT) as client:
        with wav.open("rb") as fh:
            resp = await client.post(ASR_URL, data=data, files={"file": (wav.name, fh, "audio/wav")})

    if resp.status_code != 200:
        raise HTTPException(status_code=502, detail=f"ASR returned {resp.status_code}: {resp.text}")

    body = resp.json()
    segments = body.get("segments")
    if not segments:
        raise HTTPException(
            status_code=502,
            detail="The ASR returned no timestamped segments; without them speakers "
                   "cannot be attributed. Check that it supports timestamp_granularities.",
        )
    return segments


def diarize(wav: Path, num_speakers, min_speakers, max_speakers) -> list[dict]:
    """Return the speech turns as labeled time ranges."""
    kwargs = {}
    if num_speakers:
        kwargs["num_speakers"] = num_speakers
    else:
        if min_speakers:
            kwargs["min_speakers"] = min_speakers
        if max_speakers:
            kwargs["max_speakers"] = max_speakers

    result = state["diarization"](str(wav), **kwargs)
    annotation = _as_annotation(result)
    return [
        {"start": turn.start, "end": turn.end, "speaker": speaker}
        for turn, _, speaker in annotation.itertracks(yield_label=True)
    ]


def _as_annotation(result):
    """Extract the Annotation from the pipeline's result.

    pyannote 3.x returns the Annotation directly; 4.x wraps it in a
    DiarizeOutput alongside other fields.
    """
    if hasattr(result, "itertracks"):
        return result

    for attr in ("speaker_diarization", "diarization", "annotation"):
        candidate = getattr(result, attr, None)
        if candidate is not None and hasattr(candidate, "itertracks"):
            return candidate

    fields = [a for a in dir(result) if not a.startswith("_")]
    raise RuntimeError(
        f"No Annotation found in {type(result).__name__}. Fields: {fields}"
    )


def assign_speakers(segments: list[dict], turns: list[dict]) -> list[dict]:
    """Assign each ASR segment the speaker with the largest temporal overlap.

    Whisper's cuts and pyannote's cuts do not line up, so comparing start times
    is not enough: we measure how much of each turn falls inside the segment and
    keep whichever contributes the most.
    """
    result = []
    for seg in segments:
        start, end = seg["start"], seg["end"]

        overlap_by_speaker: dict[str, float] = {}
        for turn in turns:
            overlap = min(end, turn["end"]) - max(start, turn["start"])
            if overlap > 0:
                overlap_by_speaker[turn["speaker"]] = (
                    overlap_by_speaker.get(turn["speaker"], 0.0) + overlap
                )

        speaker = max(overlap_by_speaker, key=overlap_by_speaker.get) if overlap_by_speaker else None
        result.append({
            "start": start,
            "end": end,
            "speaker": speaker,        # None = nobody was speaking per pyannote (music, noise)
            "text": seg["text"].strip(),
        })
    return result


def merge_consecutive(segments: list[dict]) -> list[dict]:
    """Join back-to-back segments from the same speaker into a single block."""
    merged: list[dict] = []
    for seg in segments:
        if merged and merged[-1]["speaker"] == seg["speaker"]:
            merged[-1]["end"] = seg["end"]
            merged[-1]["text"] += " " + seg["text"]
        else:
            merged.append(dict(seg))
    return merged


def format_transcript(segments: list[dict]) -> str:
    def stamp(seconds: float) -> str:
        m, s = divmod(int(seconds), 60)
        h, m = divmod(m, 60)
        return f"{h:02d}:{m:02d}:{s:02d}"

    return "\n".join(
        f"[{stamp(s['start'])}] {s['speaker'] or 'UNKNOWN'}: {s['text']}" for s in segments
    )


async def summarize_text(transcript: str, instructions: str | None) -> str:
    """Send the transcript to the summarization model."""
    user_content = transcript
    if instructions:
        user_content = f"{instructions}\n\n---\n\n{transcript}"

    payload = {
        "model": SUMMARIZER_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        "temperature": 0.2,
    }

    async with httpx.AsyncClient(timeout=SUMMARIZER_TIMEOUT) as client:
        try:
            resp = await client.post(SUMMARIZER_URL, json=payload)
        except httpx.ConnectError:
            raise HTTPException(
                status_code=503,
                detail=f"Could not reach the summarization model at {SUMMARIZER_URL}. "
                       "Is the 'summarize' profile up?",
            )

    if resp.status_code != 200:
        raise HTTPException(status_code=502, detail=f"Summarizer returned {resp.status_code}: {resp.text}")

    return resp.json()["choices"][0]["message"]["content"]


async def run_transcription(
    file: UploadFile,
    language: str | None,
    num_speakers: int | None,
    min_speakers: int | None,
    max_speakers: int | None,
) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        src = tmpdir / (file.filename or "audio")
        src.write_bytes(await file.read())

        wav = tmpdir / "audio.wav"
        try:
            to_wav(src, wav)
        except subprocess.CalledProcessError as exc:
            raise HTTPException(status_code=400, detail=f"ffmpeg could not decode the file: {exc}")

        # Diarization is synchronous and blocking; the ASR is an HTTP call.
        # They run in series for simplicity: both compete for the same GPU anyway.
        logger.info("Transcribing...")
        segments = await transcribe(wav, language)
        logger.info("Diarizing...")
        turns = diarize(wav, num_speakers, min_speakers, max_speakers)

    labeled = merge_consecutive(assign_speakers(segments, turns))
    speakers = sorted({s["speaker"] for s in labeled if s["speaker"]})

    return {
        "speakers": speakers,
        "segments": labeled,
        "transcript": format_transcript(labeled),
    }


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "asr_url": ASR_URL,
        "diarization_model": DIARIZATION_MODEL,
        "summarizer_url": SUMMARIZER_URL,
    }


@app.post("/transcribe")
async def transcribe_endpoint(
    file: UploadFile = File(...),
    language: str | None = Form(None),
    num_speakers: int | None = Form(None),
    min_speakers: int | None = Form(None),
    max_speakers: int | None = Form(None),
):
    return await run_transcription(file, language, num_speakers, min_speakers, max_speakers)


@app.post("/summarize")
async def summarize_endpoint(
    text: str = Form(...),
    instructions: str | None = Form(None),
):
    return {"summary": await summarize_text(text, instructions)}


@app.post("/transcribe/summarize")
async def transcribe_and_summarize_endpoint(
    file: UploadFile = File(...),
    language: str | None = Form(None),
    num_speakers: int | None = Form(None),
    min_speakers: int | None = Form(None),
    max_speakers: int | None = Form(None),
    instructions: str | None = Form(None),
    # The transcript is usually noise for the caller: by default only the summary
    # comes back. Pass all=true to also get the speakers, segments and transcript.
    want_all: bool = Form(False, alias="all"),
):
    result = await run_transcription(file, language, num_speakers, min_speakers, max_speakers)
    summary = await summarize_text(result["transcript"], instructions)

    if not want_all:
        return {"summary": summary}

    result["summary"] = summary
    return result
