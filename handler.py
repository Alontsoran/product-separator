"""RunPod serverless worker. One song in, the same stems the studio makes.

The card stays off until a job arrives. This file only runs while a job is
in hand. The weights are downloaded once, into the image, from the release
of this repository.
"""
import os
import subprocess
import tempfile
import urllib.request
from pathlib import Path

import numpy as np
import runpod

import separate_full

MODEL_DIR = os.environ.get("AUDIO_SEPARATOR_MODEL_DIR", "/models")
WHISPER_DIR = os.environ.get("WHISPER_MODEL", "/models/whisper-he")
BUCKET = os.environ.get("SUPABASE_BUCKET", "product-songs")
_lyrics_model = None


def fetch(url, dest):
    request = urllib.request.Request(url, headers={"User-Agent": "product-separator"})
    with urllib.request.urlopen(request, timeout=180) as response, open(dest, "wb") as out:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            out.write(chunk)


def put(path, key):
    base = os.environ["SUPABASE_URL"].rstrip("/")
    key_name = os.environ["SUPABASE_SERVICE_ROLE_KEY"]
    body = Path(path).read_bytes()
    request = urllib.request.Request(
        f"{base}/storage/v1/object/{BUCKET}/{key}",
        data=body,
        method="POST",
        headers={
            "Authorization": f"Bearer {key_name}",
            "Content-Type": "audio/flac",
            "x-upsert": "true",
        },
    )
    with urllib.request.urlopen(request, timeout=180) as response:
        response.read()
    return f"{base}/storage/v1/object/public/{BUCKET}/{key}"


def stems_of(result, out_dir):
    found = []
    for stem in result.get("stems") or []:
        found.append((stem["name"], out_dir / stem["file"]))
    drums = result.get("drums") or {}
    for layer in drums.get("layers") or []:
        found.append((f"drums-{layer['piece']}", out_dir / layer["file"]))
    return [(name, path) for name, path in found if path.is_file()]


def separate(audio_url, job_id):
    work = Path(tempfile.mkdtemp(prefix="sep-"))
    source = work / "song"
    out = work / "out"
    out.mkdir()
    fetch(audio_url, source)
    result = separate_full.full(str(source), out, MODEL_DIR)
    saved = []
    for name, path in stems_of(result, out):
        key = f"separations/{job_id}/{path.name}"
        saved.append({"name": name, "url": put(path, key), "bytes": path.stat().st_size})
    return {
        "duration": result.get("duration"),
        "folded": result.get("folded") or [],
        "stems": saved,
    }


def lyrics_model():
    # Loaded on the first lyrics job, on the card, not on the studio machine.
    global _lyrics_model
    if _lyrics_model is None:
        from faster_whisper import WhisperModel
        _lyrics_model = WhisperModel(WHISPER_DIR, device="cuda", compute_type="float16")
    return _lyrics_model


def load_audio(path):
    proc = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-ac", "1", "-ar", "16000", "-f", "s16le", "-"],
        check=True,
        capture_output=True,
    )
    return np.frombuffer(proc.stdout, dtype=np.int16).astype(np.float32) / 32768.0


def transcribe(audio_url):
    work = Path(tempfile.mkdtemp(prefix="lyr-"))
    source = work / "song"
    fetch(audio_url, source)
    segments, _info = lyrics_model().transcribe(
        load_audio(source),
        language="he",
        vad_filter=False,
        condition_on_previous_text=False,
        beam_size=5,
    )
    lines = []
    for segment in segments:
        text = segment.text.strip()
        if text:
            lines.append({"text": text, "start": round(segment.start, 2), "end": round(segment.end, 2)})
    return {"model": "ivrit-ai/whisper-large-v3-turbo", "lines": lines}


def handler(job):
    data = job.get("input") or {}
    audio_url = data.get("audio_url") or ""
    if not audio_url.startswith("https://"):
        return {"error": "audio_url must be an https address"}
    if not os.environ.get("SUPABASE_URL") or not os.environ.get("SUPABASE_SERVICE_ROLE_KEY"):
        return {"error": "the endpoint is missing SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY"}
    task = data.get("task") or "separate"
    if task == "lyrics":
        return transcribe(audio_url)
    return separate(audio_url, job.get("id") or "job")


runpod.serverless.start({"handler": handler})
