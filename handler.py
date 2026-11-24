"""RunPod serverless worker. One song in, the same stems the studio makes.

The card stays off until a job arrives. This file only runs while a job is
in hand. The weights are downloaded once, into the image, from the release
of this repository.
"""
import json
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


# Every instrument the song split into gets its own notes. Drums keep the
# hits found from their pieces. The others are heard note by note.
PITCHED = {
    "Vocals": ("vocal", {"minimum_frequency": 75, "maximum_frequency": 1400, "melodia_trick": True, "onset_threshold": 0.5, "frame_threshold": 0.3, "minimum_note_length": 90}),
    "Bass": ("bass", {"minimum_frequency": 28, "maximum_frequency": 400, "melodia_trick": True, "onset_threshold": 0.45, "frame_threshold": 0.3, "minimum_note_length": 80}),
    "Guitar": ("guitar", {"minimum_frequency": 70, "maximum_frequency": 1600, "melodia_trick": False, "onset_threshold": 0.5, "frame_threshold": 0.3, "minimum_note_length": 80}),
    "Piano": ("keys", {"minimum_frequency": 27, "maximum_frequency": 4200, "melodia_trick": False, "onset_threshold": 0.5, "frame_threshold": 0.3, "minimum_note_length": 70}),
    "Keyboard": ("keys", {"minimum_frequency": 27, "maximum_frequency": 4200, "melodia_trick": False, "onset_threshold": 0.5, "frame_threshold": 0.3, "minimum_note_length": 70}),
    "Other": ("poly", {"minimum_frequency": 30, "maximum_frequency": 4200, "melodia_trick": False, "onset_threshold": 0.55, "frame_threshold": 0.35, "minimum_note_length": 80}),
}


def hear(stems):
    try:
        from basic_pitch import FilenameSuffix, build_icassp_2022_model_path
        from basic_pitch.inference import Model, predict
    except Exception as error:
        return [], str(error)
    model = Model(build_icassp_2022_model_path(FilenameSuffix.onnx))
    tracks = []
    for name, path in stems:
        spec = PITCHED.get(name)
        if spec is None or not path.is_file():
            continue
        profile, limits = spec
        try:
            _midi, _roll, events = predict(str(path), model, **limits)
        except Exception as error:
            tracks.append({"name": name, "profile": profile, "notes": [], "error": str(error)})
            continue
        notes = []
        for start, end, pitch, amplitude, _bends in events:
            if end <= start:
                continue
            notes.append({
                "start": round(float(start), 3),
                "end": round(float(end), 3),
                "pitch": int(pitch),
                "velocity": int(max(1, min(127, round(float(amplitude) * 127)))),
                "kind": "note",
            })
        if len(notes) > 1500:
            notes = sorted(notes, key=lambda note: note["velocity"], reverse=True)[:1500]
        notes.sort(key=lambda note: (note["start"], note["pitch"]))
        tracks.append({"name": name, "profile": profile, "notes": notes})
    return tracks, ""


PIECES = {
    "kick": ("Kick", 36),
    "snare": ("Snare", 38),
    "toms": ("Toms", 45),
    "hh": ("Hi-Hat", 42),
    "ride": ("Ride", 51),
    "crash": ("Crash", 49),
}


def hit_tracks(layers):
    tracks = []
    kit = []
    for layer in layers:
        known = PIECES.get(layer["piece"])
        if known is None:
            continue
        name, pitch = known
        pitch = layer.get("pitch") or pitch
        notes = []
        for hit in layer.get("hits") or []:
            start = hit.get("start")
            end = hit.get("end")
            if not isinstance(start, (int, float)) or not isinstance(end, (int, float)):
                continue
            notes.append({
                "start": start,
                "end": max(end, start + 0.04),
                "pitch": pitch,
                "velocity": hit.get("velocity") or 80,
                "kind": "drum",
            })
        if not notes:
            continue
        kit.extend(notes)
        tracks.append({"id": f"piece:{name}", "name": name, "profile": f"drum:{pitch}", "trimmed": 0, "notes": notes})
    kit.sort(key=lambda note: (note["start"], note["pitch"]))
    if kit:
        tracks.insert(0, {"id": "kit:Drums", "name": "Drums", "profile": "drum", "trimmed": 0, "notes": kit})
    return tracks


def store_notes(clip_id, tracks):
    if not clip_id or len(clip_id) < 32 or "/" in clip_id:
        return
    base = os.environ["SUPABASE_URL"].rstrip("/")
    key_name = os.environ["SUPABASE_SERVICE_ROLE_KEY"]
    body = json.dumps({"tracks": tracks}).encode()
    request = urllib.request.Request(
        f"{base}/storage/v1/object/{BUCKET}/notes/{clip_id}.json",
        data=body,
        method="POST",
        headers={
            "Authorization": f"Bearer {key_name}",
            "Content-Type": "application/json",
            "x-upsert": "true",
        },
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        response.read()


def separate(audio_url, job_id, clip_id=""):
    work = Path(tempfile.mkdtemp(prefix="sep-"))
    source = work / "song"
    out = work / "out"
    out.mkdir()
    fetch(audio_url, source)
    result = separate_full.full(str(source), out, MODEL_DIR)
    saved = []
    paths = []
    for name, path in stems_of(result, out):
        key = f"separations/{job_id}/{path.name}"
        saved.append({"name": name, "url": put(path, key), "bytes": path.stat().st_size})
        paths.append((name, path))
    drums = result.get("drums") or {}
    hits = [
        {"piece": layer["piece"], "pitch": layer["pitch"], "hits": layer.get("hits") or []}
        for layer in drums.get("layers") or []
        if layer.get("hits")
    ]
    notes, notes_error = hear(paths)
    # The boxes are part of the separation. A song without them is not done.
    print(json.dumps({"boxes": sum(len(track.get("notes") or []) for track in notes) + sum(len(layer.get("hits") or []) for layer in hits), "clip": bool(clip_id)}), flush=True)
    try:
        store_notes(clip_id, hit_tracks(hits) + notes)
    except Exception as error:
        notes_error = notes_error or str(error)
    payload = {
        "duration": result.get("duration"),
        "folded": result.get("folded") or [],
        "stems": saved,
        "hits": hits,
        "notes": notes,
    }
    if notes_error:
        payload["notes_error"] = notes_error
    return payload


def lyrics_model():
    # Loaded on the first lyrics job, on the card, not on the studio machine.
    # The Hebrew model is pulled here when the image does not already have it.
    global _lyrics_model
    if _lyrics_model is None:
        from pathlib import Path
        from faster_whisper import WhisperModel
        if not Path(WHISPER_DIR).is_dir() or not any(Path(WHISPER_DIR).iterdir()):
            from huggingface_hub import snapshot_download
            snapshot_download("ivrit-ai/whisper-large-v3-ct2-20250513", local_dir=WHISPER_DIR)
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
        temperature=0.0,
        hallucination_silence_threshold=2.0,
    )
    lines = []
    for segment in segments:
        text = segment.text.strip()
        if text:
            lines.append({"text": text, "start": round(segment.start, 2), "end": round(segment.end, 2)})
    return {"model": "ivrit-ai/whisper-large-v3-ct2-20250513", "lines": lines}


def handler(job):
    data = job.get("input") or {}
    audio_url = data.get("audio_url") or ""
    if not audio_url.startswith("https://"):
        return {"error": "audio_url must be an https address"}
    task = data.get("task") or "separate"
    if task == "lyrics":
        return transcribe(audio_url)
    if not os.environ.get("SUPABASE_URL") or not os.environ.get("SUPABASE_SERVICE_ROLE_KEY"):
        return {"error": "the endpoint is missing SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY"}
    return separate(audio_url, job.get("id") or "job", data.get("clip_id") or "")


runpod.serverless.start({"handler": handler})
