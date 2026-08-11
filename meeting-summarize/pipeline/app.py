"""Transcripción con atribución de hablante.

Combina dos modelos que resuelven mitades distintas del problema:
  - Whisper (servido por vLLM) dice QUÉ se dijo y en qué rango de tiempo.
  - pyannote dice QUIÉN habla en cada rango, pero no qué dice.

El cruce se hace por solapamiento temporal: a cada segmento de Whisper se le
asigna el hablante que más tiempo ocupa dentro de ese rango.
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

# Una reunión larga puede tardar varios minutos: la request al ASR no es streaming.
ASR_TIMEOUT = float(os.environ.get("ASR_TIMEOUT", "3600"))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("pipeline")

state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Cargar el modelo acá y no por request: son ~10s de carga que no queremos pagar cada vez.
    if not HF_TOKEN:
        raise RuntimeError(
            "Falta HF_TOKEN. pyannote/speaker-diarization-3.1 es un modelo gated: "
            "hay que aceptar las condiciones en huggingface.co con la misma cuenta del token."
        )

    logger.info("Cargando %s...", DIARIZATION_MODEL)
    # pyannote 3.x usa use_auth_token; 4.x lo renombró a token.
    try:
        pipeline = Pipeline.from_pretrained(DIARIZATION_MODEL, token=HF_TOKEN)
    except TypeError:
        pipeline = Pipeline.from_pretrained(DIARIZATION_MODEL, use_auth_token=HF_TOKEN)
    if pipeline is None:
        raise RuntimeError(
            f"{DIARIZATION_MODEL} devolvió None. Casi siempre significa que el token "
            "no tiene aceptadas las condiciones del modelo en el Hub."
        )

    if torch.cuda.is_available():
        pipeline.to(torch.device("cuda"))
        logger.info("Diarización en GPU.")
    else:
        logger.warning("Sin CUDA: la diarización va por CPU y va a ser lenta.")

    state["diarization"] = pipeline
    yield
    state.clear()


app = FastAPI(title="meeting-summarize pipeline", lifespan=lifespan)


def to_wav(src: Path, dst: Path) -> None:
    """Normaliza a WAV 16 kHz mono.

    Dos razones: es lo que Whisper y pyannote usan internamente, y la imagen de
    vLLM trae un libsndfile sin soporte MP3, con lo que mandarle un MP3 falla.
    """
    subprocess.run(
        ["ffmpeg", "-nostdin", "-loglevel", "error", "-y",
         "-i", str(src), "-ar", "16000", "-ac", "1", str(dst)],
        check=True,
    )


async def transcribe(wav: Path, language: str | None) -> list[dict]:
    """Devuelve los segmentos de Whisper con sus timestamps."""
    data = {
        "model": ASR_MODEL,
        "response_format": "verbose_json",
        # El alias con corchetes es el que espera vLLM para esta lista.
        "timestamp_granularities[]": "segment",
    }
    if language:
        data["language"] = language

    async with httpx.AsyncClient(timeout=ASR_TIMEOUT) as client:
        with wav.open("rb") as fh:
            resp = await client.post(ASR_URL, data=data, files={"file": (wav.name, fh, "audio/wav")})

    if resp.status_code != 200:
        raise HTTPException(status_code=502, detail=f"ASR respondió {resp.status_code}: {resp.text}")

    body = resp.json()
    segments = body.get("segments")
    if not segments:
        raise HTTPException(
            status_code=502,
            detail="El ASR no devolvió segmentos con timestamps; sin ellos no se puede "
                   "atribuir hablante. Verificá que soporte timestamp_granularities.",
        )
    return segments


def diarize(wav: Path, num_speakers, min_speakers, max_speakers) -> list[dict]:
    """Devuelve los turnos de habla como rangos etiquetados."""
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
    """Extrae el Annotation del resultado del pipeline.

    pyannote 3.x devuelve el Annotation directo; 4.x lo envuelve en un
    DiarizeOutput junto con otros campos.
    """
    if hasattr(result, "itertracks"):
        return result

    for attr in ("speaker_diarization", "diarization", "annotation"):
        candidate = getattr(result, attr, None)
        if candidate is not None and hasattr(candidate, "itertracks"):
            return candidate

    fields = [a for a in dir(result) if not a.startswith("_")]
    raise RuntimeError(
        f"No se encontró el Annotation en {type(result).__name__}. Campos: {fields}"
    )


def assign_speakers(segments: list[dict], turns: list[dict]) -> list[dict]:
    """Asigna a cada segmento del ASR el hablante con mayor solapamiento temporal.

    Los cortes de Whisper y los de pyannote no coinciden, así que no alcanza con
    comparar los inicios: hay que medir cuánto tiempo de cada turno cae dentro
    del segmento y quedarse con el que más aporta.
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
            "speaker": speaker,        # None = nadie hablaba según pyannote (música, ruido)
            "text": seg["text"].strip(),
        })
    return result


def merge_consecutive(segments: list[dict]) -> list[dict]:
    """Junta segmentos seguidos del mismo hablante en un solo bloque."""
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


@app.get("/health")
async def health():
    return {"status": "ok", "asr_url": ASR_URL, "diarization_model": DIARIZATION_MODEL}


@app.post("/transcribe")
async def transcribe_endpoint(
    file: UploadFile = File(...),
    language: str | None = Form(None),
    num_speakers: int | None = Form(None),
    min_speakers: int | None = Form(None),
    max_speakers: int | None = Form(None),
):
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        src = tmpdir / (file.filename or "audio")
        src.write_bytes(await file.read())

        wav = tmpdir / "audio.wav"
        try:
            to_wav(src, wav)
        except subprocess.CalledProcessError as exc:
            raise HTTPException(status_code=400, detail=f"ffmpeg no pudo decodificar el archivo: {exc}")

        # La diarización es sincrónica y bloquea; el ASR es una llamada HTTP.
        # Se corren en serie por simplicidad: ambos compiten por la misma GPU igual.
        logger.info("Transcribiendo...")
        segments = await transcribe(wav, language)
        logger.info("Diarizando...")
        turns = diarize(wav, num_speakers, min_speakers, max_speakers)

    labeled = merge_consecutive(assign_speakers(segments, turns))
    speakers = sorted({s["speaker"] for s in labeled if s["speaker"]})

    return {
        "speakers": speakers,
        "segments": labeled,
        "transcript": format_transcript(labeled),
    }
