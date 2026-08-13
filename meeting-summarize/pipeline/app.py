"""Transcription with speaker attribution.

Combines two models that solve different halves of the problem:
  - Whisper (served by vLLM) says WHAT was said and in which time range.
  - pyannote says WHO speaks in each range, but not what they say.

The two are merged by temporal overlap: each Whisper segment is assigned the
speaker who occupies the most time within that range.
"""

import asyncio
import json
import logging
import os
import re
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
# vLLM exposes the tokenizer next to the OpenAI routes, outside /v1.
TOKENIZE_URL = SUMMARIZER_URL.replace("/v1/chat/completions", "/tokenize")

# A long meeting can take several minutes: the ASR request is not streamed.
ASR_TIMEOUT = float(os.environ.get("ASR_TIMEOUT", "3600"))
SUMMARIZER_TIMEOUT = float(os.environ.get("SUMMARIZER_TIMEOUT", "1800"))

# Must match --max-model-len on the vllm-summarize service.
SUMMARIZER_CONTEXT = int(os.environ.get("SUMMARIZER_CONTEXT", "32768"))
# Output budgets. Without an explicit max_tokens vLLM gives the reply whatever is
# left of the context, which silently shrinks to nothing on a long transcript.
SUMMARY_MAX_TOKENS = int(os.environ.get("SUMMARY_MAX_TOKENS", "4096"))
NOTES_MAX_TOKENS = int(os.environ.get("NOTES_MAX_TOKENS", "1536"))
# Leaf sections stay well under the context so the model attends over all of it.
SECTION_BUDGET_TOKENS = int(os.environ.get("SECTION_BUDGET_TOKENS", "6000"))
# The segmentation pass only emits boundaries, so its window can be much larger.
SEGMENTATION_WINDOW_TOKENS = int(os.environ.get("SEGMENTATION_WINDOW_TOKENS", "12000"))
# vLLM batches concurrent requests, and batching is what makes a memory-bound 72B
# worth running: the weights are read once for the whole batch.
MAP_CONCURRENCY = int(os.environ.get("MAP_CONCURRENCY", "4"))

# Empty means "follow the transcript". Set it to force the output language.
SUMMARY_LANGUAGE = os.environ.get("SUMMARY_LANGUAGE", "").strip()
_LANGUAGE_RULE = (
    f"Write your reply in {SUMMARY_LANGUAGE}."
    if SUMMARY_LANGUAGE
    else "Write your reply in the same language the transcript is in."
)

SYSTEM_PROMPT = f"""You summarize meeting transcripts.

You receive either the transcript itself or structured notes taken over it section
by section, in order. Speaker labels (SPEAKER_00, SPEAKER_01, ...) are anonymous:
use them as-is unless a real name is revealed.

{_LANGUAGE_RULE} Produce Markdown. Cover every topic your input mentions and do not
invent details that are not in it.

Structure it as:
- **Executive summary** - a short paragraph.
- **Topics discussed** - one subsection per topic, with what was said and by whom.
- **Decisions** - what was actually decided, not what was merely discussed.
- **Action items** - owner, task and deadline where stated; say so when one is missing.
- **Open questions** - anything left unresolved.

Drop a section only if the notes genuinely have nothing for it.
"""

SEGMENTER_PROMPT = f"""You split meeting transcripts into topical sections.

You receive numbered blocks of a transcript: `[N] [timestamp] SPEAKER: text`.
Identify where the conversation moves on to a new topic.

Reply with ONLY a JSON array, no prose and no code fence:
[{{"block": 12, "title": "short topic title"}}, ...]

Rules:
- `block` is the number of the block where the new topic STARTS.
- Return the boundaries in ascending order.
- Only real topic shifts. An excerpt this size usually has a handful, not dozens.
- {_LANGUAGE_RULE}
- If the whole excerpt is a single topic, reply with [].
"""

EXTRACTOR_PROMPT = f"""You take structured notes on one section of a meeting transcript.

Do NOT write a summary or a narrative: extract the facts, so that someone can write
the summary later from your notes alone. Keep the detail - this is the step where
information gets lost if you compress it.

{_LANGUAGE_RULE} Reply in Markdown bullets, covering whatever is present: points made
(and by which speaker), decisions taken, action items with their owner, figures, dates,
names, tools, and open questions. Keep the `[hh:mm:ss]` timestamp on anything
important. Do not invent anything.
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


async def chat(system: str, user: str, max_tokens: int) -> str:
    """One call to the summarization model."""
    payload = {
        "model": SUMMARIZER_MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": 0.2,
        "max_tokens": max_tokens,
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


async def chars_per_token(text: str) -> float:
    """Calibrate the chars->tokens ratio on this transcript.

    Packing decisions are made on character counts, which is cheap but depends on
    the language. One call to the real tokenizer pins the ratio; if the endpoint
    is not there we fall back to a figure conservative enough for Spanish.
    """
    sample = text[:20000]
    if not sample:
        return 3.0
    try:
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(
                TOKENIZE_URL, json={"model": SUMMARIZER_MODEL, "prompt": sample}
            )
        if resp.status_code == 200:
            body = resp.json()
            count = body.get("count") or len(body.get("tokens") or [])
            if count:
                return len(sample) / count
    except httpx.HTTPError:
        pass
    logger.warning("Could not reach %s; estimating tokens by character count.", TOKENIZE_URL)
    return 3.0


def pack_ranges(lines: list[str], budget_chars: float, start: int = 0, stop: int | None = None):
    """Group consecutive lines into [from, to) ranges under the budget.

    Lines are never split: a transcript line is one speaker turn, and cutting
    somebody mid-sentence is what makes naive chunking produce bad sections.
    """
    stop = len(lines) if stop is None else stop
    ranges, begin, size = [], start, 0
    for i in range(start, stop):
        length = len(lines[i]) + 1
        if size and size + length > budget_chars:
            ranges.append((begin, i))
            begin, size = i, 0
        size += length
    if begin < stop:
        ranges.append((begin, stop))
    return ranges


def parse_boundaries(raw: str) -> dict[int, str]:
    """Pull {block: title} out of the segmenter's reply.

    The model is asked for bare JSON but sometimes wraps it in prose or a fence,
    so we take the outermost array rather than trusting the whole string.
    """
    match = re.search(r"\[.*\]", raw, re.DOTALL)
    if not match:
        return {}
    try:
        items = json.loads(match.group(0))
    except json.JSONDecodeError:
        logger.warning("Segmenter returned invalid JSON, ignoring this window.")
        return {}

    found = {}
    for item in items:
        if isinstance(item, dict) and isinstance(item.get("block"), int):
            found[item["block"]] = str(item.get("title") or "Untitled section")
    return found


async def find_sections(lines: list[str], ratio: float) -> list[tuple[int, int, str]]:
    """Split the transcript into (from, to, title) topical sections.

    Two passes, because topic boundaries and size limits solve different problems:
    the model marks where topics change, then any section still too large for a
    leaf call is subdivided by size. A single topic can run for forty minutes.
    """
    window_chars = SEGMENTATION_WINDOW_TOKENS * ratio
    boundaries: dict[int, str] = {}

    for start, stop in pack_ranges(lines, window_chars):
        numbered = "\n".join(f"[{i}] {lines[i]}" for i in range(start, stop))
        raw = await chat(SEGMENTER_PROMPT, numbered, max_tokens=1024)
        for block, title in parse_boundaries(raw).items():
            # Ignore a boundary on the window's own first block: it is where we cut,
            # not a topic change the model found.
            if start < block < stop:
                boundaries[block] = title

    cuts = sorted(boundaries)
    logger.info("Segmentation found %d topic boundaries.", len(cuts))

    topics, previous, title = [], 0, "Opening"
    for cut in cuts:
        topics.append((previous, cut, title))
        previous, title = cut, boundaries[cut]
    topics.append((previous, len(lines), title))

    section_chars = SECTION_BUDGET_TOKENS * ratio
    sections = []
    for start, stop, title in topics:
        if start >= stop:
            continue
        parts = pack_ranges(lines, section_chars, start, stop)
        for n, (a, b) in enumerate(parts):
            sections.append((a, b, title if len(parts) == 1 else f"{title} ({n + 1}/{len(parts)})"))
    return sections


async def summarize_text(transcript: str, instructions: str | None) -> str:
    """Summarize a transcript, going hierarchical when it does not fit in one pass.

    A transcript that fits is summarized directly: the model seeing the whole
    meeting at once connects a decision at minute 10 with its follow-up at minute
    50, which a chunked pass cannot. Beyond that it is segmented by topic, each
    section is turned into notes, and the notes are reduced into the summary.
    """
    transcript = transcript.strip()
    if not transcript:
        raise HTTPException(status_code=400, detail="The transcript is empty.")

    ratio = await chars_per_token(transcript)
    approx_tokens = len(transcript) / ratio
    # Leave room for the reply and the prompts themselves.
    single_pass_budget = SUMMARIZER_CONTEXT - SUMMARY_MAX_TOKENS - 1000

    if approx_tokens <= single_pass_budget:
        logger.info("Transcript is ~%d tokens: summarizing in one pass.", approx_tokens)
        return await reduce_summary(transcript, instructions, source="transcript")

    lines = [line for line in transcript.splitlines() if line.strip()]
    sections = await find_sections(lines, ratio)
    logger.info("Transcript is ~%d tokens: %d sections.", approx_tokens, len(sections))

    semaphore = asyncio.Semaphore(MAP_CONCURRENCY)

    async def notes_for(start: int, stop: int, title: str) -> str:
        body = "\n".join(lines[start:stop])
        async with semaphore:
            notes = await chat(EXTRACTOR_PROMPT, f"# {title}\n\n{body}", NOTES_MAX_TOKENS)
        return f"## {title}\n\n{notes.strip()}"

    notes = await asyncio.gather(*(notes_for(*section) for section in sections))
    combined = "\n\n".join(notes)

    # The notes are a fraction of the transcript, but a very long meeting can still
    # overflow the reduce. Fold them down a level at a time until they fit.
    while len(combined) / ratio > single_pass_budget:
        blocks = combined.split("\n\n")
        groups = pack_ranges(blocks, SECTION_BUDGET_TOKENS * ratio)
        if len(groups) <= 1:
            # Cannot split further; the reduce would 400 on context length anyway.
            logger.warning("Notes do not fit and cannot be folded further, truncating.")
            combined = combined[: int(single_pass_budget * ratio)]
            break
        logger.info("Notes still too long, folding %d blocks into %d.", len(blocks), len(groups))
        folded = await asyncio.gather(*(
            chat(EXTRACTOR_PROMPT, "\n\n".join(blocks[a:b]), NOTES_MAX_TOKENS)
            for a, b in groups
        ))
        combined = "\n\n".join(folded)

    return await reduce_summary(combined, instructions, source="notes")


async def reduce_summary(body: str, instructions: str | None, source: str) -> str:
    header = (
        "Here is the full meeting transcript."
        if source == "transcript"
        else "Here are the notes taken over the meeting, section by section, in order."
    )
    user_content = f"{header}\n\n{body}"
    if instructions:
        user_content = f"{instructions}\n\n---\n\n{user_content}"
    return await chat(SYSTEM_PROMPT, user_content, SUMMARY_MAX_TOKENS)


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
