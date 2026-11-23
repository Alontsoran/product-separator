"""Full separation of one song on this machine. No Suno credits, nothing expires.

  separate_full.py full  <input> <out_dir> <model_dir>
  separate_full.py drums <input> <out_dir> <model_dir>
  separate_full.py rehits <out_dir>

full:  song -> Vocals, Drums, Bass, Guitar, Piano, Other, and the drums again
       into kick, snare, toms, hi-hat, ride and crash. An instrument the song
       does not have is not written as an empty stem: when there is no piano
       the keys of the song are in what is left, and that stem is "Keyboard".
drums: only the second half, for a drums stem that already exists.
rehits: reads the hits again from drum pieces that were already separated.

Every file is written lossless at the same rate and length, and the parts are
built so that they add up to what they were cut from: the stems to the song,
the drum pieces (plus what is left over) to the drums.
"""
import json
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path

# Full songs hang inside torch.istft on Apple MPS; the reconstruction runs on the CPU.
os.environ["AUDIO_SEPARATOR_FORCE_CPU_COMPLEX"] = "1"

import types

import librosa
import numpy as np
import soundfile as sf

try:
    import diffq  # noqa: F401
except ModuleNotFoundError:
    # Demucs imports diffq only to open quantized checkpoints. The models used
    # here are not quantized, and diffq does not build on this machine, so a
    # stand-in that refuses to be used is enough to let Demucs load.
    def _missing(*_args, **_kwargs):
        raise RuntimeError("diffq is not installed; quantized Demucs checkpoints are not supported")

    _stub = types.ModuleType("diffq")
    _stub.DiffQuantizer = _missing
    _stub.UniformQuantizer = _missing
    _stub.restore_quantized_state = _missing
    sys.modules["diffq"] = _stub

from audio_separator.separator import Separator

RATE = 44100
# One model for the whole song: vocals, drums, bass, guitar, piano, other.
SONG_MODEL = "BS-Roformer-SW.ckpt"
# A stem this far under the song is an instrument the song does not have:
# what the model wrote there is bleed, and it goes back to the rest.
ABSENT_DB = -26
DRUM_MODEL = "MDX23C-DrumSep-aufr33-jarredou.ckpt"
# Piece -> General MIDI drum number, and how long a hit is drawn.
PIECES = {"kick": (36, 0.18), "snare": (38, 0.14), "toms": (45, 0.16), "hh": (42, 0.07), "ride": (51, 0.12), "crash": (49, 0.3)}
# The shortest time between two real hits of a piece. An onset that comes
# sooner is the same hit: its body after its attack.
GAPS = {"kick": 0.09, "snare": 0.06, "toms": 0.08, "hh": 0.05, "ride": 0.06, "crash": 0.12}


def say(**fields):
    print(json.dumps(fields, ensure_ascii=False), flush=True)


def load(path):
    audio, _ = librosa.load(str(path), sr=RATE, mono=False)
    if audio.ndim == 1:
        audio = np.stack([audio, audio])
    return np.ascontiguousarray(audio[:2], dtype=np.float32)


def save(path, audio):
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), np.clip(audio, -1, 1).T, RATE, format="FLAC", subtype="PCM_16")


def fit(audio, length):
    if audio.shape[1] >= length:
        return audio[:, :length]
    return np.pad(audio, ((0, 0), (0, length - audio.shape[1])))


def rms(audio):
    return float(np.sqrt(np.mean(np.square(audio, dtype=np.float64)))) if audio.size else 0.0


def db(value):
    return round(20 * float(np.log10(max(value, 1e-9))), 1)


class Runner:
    def __init__(self, model_dir):
        self.work = Path(tempfile.mkdtemp(prefix="suno-studio-sep-"))
        self.separator = Separator(output_dir=str(self.work), output_format="WAV", model_file_dir=model_dir)
        self.count = 0

    def run(self, model, audio):
        """Separates `audio` with `model`; returns {stem name in lower case: audio}."""
        self.count += 1
        source = self.work / f"in{self.count}.wav"
        sf.write(str(source), audio.T, RATE, subtype="PCM_24")
        self.separator.load_model(model_filename=model)
        out = {}
        for item in self.separator.separate(str(source)):
            path = Path(item)
            if not path.is_absolute():
                path = self.work / path
            found = re.search(r"_\(([^)]+)\)_", path.name)
            if found:
                out[found.group(1).strip().lower()] = fit(load(path), audio.shape[1])
            path.unlink(missing_ok=True)
        source.unlink(missing_ok=True)
        return out

    def close(self):
        shutil.rmtree(self.work, ignore_errors=True)


def level(mono, start, stop):
    part = mono[max(0, start):max(0, stop)]
    return float(np.sqrt(np.mean(np.square(part, dtype=np.float64)))) if part.size else 0.0


def find_hits(audio, gap, ring=0.25):
    """Onsets of one drum piece, read from its own separated audio.

    Returns the hits as samples with their strength, and the level of the
    piece's loud hits, which the weak ones are measured against.
    """
    mono = audio.mean(axis=0)
    hop = 256
    envelope = librosa.onset.onset_strength(y=mono, sr=RATE, hop_length=hop)
    frames = librosa.onset.onset_detect(onset_envelope=envelope, sr=RATE, hop_length=hop, backtrack=True, units="frames")
    reach = int(0.06 * RATE)
    found = []
    for frame in frames:
        at = int(frame) * hop
        if at >= mono.size:
            continue
        after = level(mono, at, at + int(0.03 * RATE))
        before = level(mono, at - int(0.03 * RATE), at - int(0.004 * RATE))
        found.append({
            "at": at,
            "peak": float(np.max(np.abs(mono[at:at + reach]))),
            "loud": after,
            "rise": after / (before + 1e-9),
        })
    if not found:
        return [], 1.0
    peaks = [item["peak"] for item in found]
    # The level of the piece's real hits. When most onsets are bleed, the
    # 95th percentile is bleed too, so the loudest hit sets a floor for it.
    top = max(float(np.percentile(peaks, 95)), 0.35 * max(peaks)) or 1.0
    hits = []
    for item in found:
        # What is far below the piece's own level is bleed from another piece.
        if item["peak"] < 0.07 * top:
            continue
        if hits:
            since = (item["at"] - hits[-1]["at"]) / RATE
            if since < gap:
                # Two onsets inside one hit: the attack and its own body. The
                # first is the hit, unless it was only a faint lead-in.
                if item["peak"] > 2.5 * hits[-1]["peak"]:
                    hits[-1] = item
                continue
            # Close to the last hit and no louder than its tail: still ringing.
            if since < ring and item["rise"] < 1.25:
                continue
        hits.append(item)
    return hits, top


def drop_bleed(pieces):
    """Removes hits that are another piece heard through this one.

    A weak onset that falls exactly on a hit of a piece that is much louder at
    that moment is that piece's bleed, not a hit of its own.
    """
    span = int(0.05 * RATE)
    near = int(0.025 * RATE)
    marks = {name: np.array([hit["at"] for hit in piece["hits"]], dtype=np.int64) for name, piece in pieces.items()}
    for name, piece in pieces.items():
        kept = []
        for hit in piece["hits"]:
            masked = False
            if hit["peak"] < 0.2 * piece["top"]:
                for other, their in pieces.items():
                    if other == name or marks[other].size == 0:
                        continue
                    index = int(np.searchsorted(marks[other], hit["at"]))
                    close = min(
                        abs(int(marks[other][index - 1]) - hit["at"]) if index > 0 else near + 1,
                        abs(int(marks[other][index]) - hit["at"]) if index < marks[other].size else near + 1,
                    )
                    if close <= near and level(their["mono"], hit["at"], hit["at"] + span) >= 4 * level(piece["mono"], hit["at"], hit["at"] + span):
                        masked = True
                        break
            if not masked:
                kept.append(hit)
        piece["kept"] = kept
    for piece in pieces.values():
        piece["hits"] = piece.pop("kept")


def fold_shadows(heard, level_all):
    """Gives a faint, steady cymbal line back to the hi-hat.

    The drum model often writes part of the hi-hat's ticking into the ride or
    the crash output. A real ride or crash that quiet is a few clear hits; a
    steady line of small equal ones is the hi-hat, and muting the hi-hat has
    to take it away too.
    """
    folded = []
    if "hh" not in heard:
        return folded
    for piece in ("ride", "crash"):
        audio = heard.get(piece)
        if audio is None or rms(audio) / level_all >= 10 ** (-26 / 20):
            continue
        hits, _top = find_hits(audio, GAPS[piece], ring=0)
        if len(hits) < 40:
            continue
        energy = sorted((hit["loud"] ** 2 for hit in hits), reverse=True)
        if sum(energy[:max(1, len(energy) // 10)]) / (sum(energy) or 1.0) < 0.6:
            heard["hh"] = heard["hh"] + heard.pop(piece)
            folded.append(piece)
    return folded


def read_hits(parts):
    """{piece: audio} -> {piece: [{start, end, velocity}]} for the pieces that are really played."""
    pieces = {}
    for piece, audio in parts.items():
        hits, top = find_hits(audio, GAPS[piece])
        pieces[piece] = {"mono": audio.mean(axis=0), "hits": hits, "top": top}
    drop_bleed(pieces)
    out = {}
    for piece, data in pieces.items():
        hold = PIECES[piece][1]
        out[piece] = [{
            "start": round(hit["at"] / RATE, 3),
            "end": round(hit["at"] / RATE + hold, 3),
            "velocity": int(np.clip(round(127 * hit["peak"] / data["top"]), 1, 127)),
        } for hit in data["hits"]]
    return out


def split_drums(runner, drums, out_dir, base):
    say(step="drums", note="kick, snare, toms, hi-hat, ride, crash")
    parts = runner.run(DRUM_MODEL, drums)
    level_all = rms(drums) or 1e-9
    heard = {}
    for piece in PIECES:
        audio = parts.get(piece)
        # A piece the song does not have comes back as faint bleed. It stays in
        # the leftover instead of becoming an empty row.
        if audio is None or rms(audio) / level_all < 10 ** (-34 / 20):
            continue
        heard[piece] = audio
    fold_shadows(heard, level_all)
    hits = read_hits(heard)
    layers = []
    total = np.zeros_like(drums)
    for piece, audio in heard.items():
        if not hits[piece]:
            continue
        file = f"drums/{piece}.flac"
        save(out_dir / file, audio)
        total += audio
        layers.append({"piece": piece, "pitch": PIECES[piece][0], "file": file, "level_db": db(rms(audio) / level_all), "hits": hits[piece]})
    return {
        "base": base,
        "layers": layers,
        "leftover_db": db(rms(drums - total) / level_all),
    }


def rehits(out_dir):
    """Reads the hits again from drum pieces that were already separated."""
    path = out_dir / "result.json"
    result = json.loads(path.read_text(encoding="utf-8"))
    drums = result["drums"]
    base = load(out_dir / drums["base"])
    level_all = rms(base) or 1e-9
    heard = {layer["piece"]: load(out_dir / layer["file"]) for layer in drums["layers"]}
    if fold_shadows(heard, level_all):
        save(out_dir / "drums" / "hh.flac", heard["hh"])
    hits = read_hits(heard)
    layers = []
    for layer in drums["layers"]:
        piece = layer["piece"]
        if piece not in heard or not hits[piece]:
            continue
        layers.append({**layer, "level_db": db(rms(heard[piece]) / level_all), "hits": hits[piece]})
    drums["layers"] = layers
    return result


def full(source, out_dir, model_dir):
    runner = Runner(model_dir)
    try:
        say(step="load")
        mix = load(source)
        peak = float(np.max(np.abs(mix))) or 1.0
        scale = min(1.0, 0.8 / peak)
        mix = mix * scale
        length = mix.shape[1]
        save(out_dir / "Mix.flac", mix)

        say(step="song", note="vocals, drums, bass, guitar, piano, the rest")
        first = runner.run(SONG_MODEL, mix)
        parts = {name: fit(first[name], length) for name in ("vocals", "drums", "bass", "guitar", "piano")}
        model_error = rms(mix - sum(parts.values()) - fit(first["other"], length)) / rms(mix)
        drums = parts["drums"]
        # Whatever is none of the five, so the stems add up to the song.
        other = mix - sum(parts.values())

        folded = []
        for name in ("guitar", "piano"):
            if db(rms(parts[name]) / rms(mix)) < ABSENT_DB:
                other = other + parts.pop(name)
                folded.append(name.capitalize())
        # A song without a piano keeps its keys in the rest: electric pianos,
        # synths and pads are not "piano" to the model.
        rest_name = "Keyboard" if "Piano" in folded else "Other"

        stems = []
        for name in ("vocals", "drums", "bass", "guitar", "piano"):
            if name not in parts:
                continue
            save(out_dir / f"{name.capitalize()}.flac", parts[name])
            stems.append({"name": name.capitalize(), "file": f"{name.capitalize()}.flac", "level_db": db(rms(parts[name]) / rms(mix))})
        if db(rms(other) / rms(mix)) >= -50:
            save(out_dir / f"{rest_name}.flac", other)
            stems.append({"name": rest_name, "file": f"{rest_name}.flac", "level_db": db(rms(other) / rms(mix))})

        split = split_drums(runner, drums, out_dir, "Drums.flac")
        result = {
            "rate": RATE,
            "duration": round(length / RATE, 3),
            "scale": round(scale, 6),
            "mix": "Mix.flac",
            "stems": stems,
            "drums": split,
            "folded": folded,
            "checks": {"model_sum_error_db": db(model_error)},
        }
    finally:
        runner.close()
    return result


def drums_only(source, out_dir, model_dir):
    runner = Runner(model_dir)
    try:
        say(step="load")
        drums = load(source)
        peak = float(np.max(np.abs(drums))) or 1.0
        drums = drums * min(1.0, 0.8 / peak)
        # The exact audio the pieces were cut from. The editor works on this
        # copy, so the pieces line up with it sample for sample.
        save(out_dir / "drums" / "base.flac", drums)
        split = split_drums(runner, drums, out_dir, "drums/base.flac")
        result = {"rate": RATE, "duration": round(drums.shape[1] / RATE, 3), "drums": split}
    finally:
        runner.close()
    return result


def main():
    mode = sys.argv[1]
    if mode == "rehits":
        out_dir = Path(sys.argv[2])
        result = rehits(out_dir)
    else:
        source, out_dir, model_dir = sys.argv[2], Path(sys.argv[3]), sys.argv[4]
        out_dir.mkdir(parents=True, exist_ok=True)
        result = full(source, out_dir, model_dir) if mode == "full" else drums_only(source, out_dir, model_dir)
    (out_dir / "result.json").write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
    say(step="done")


if __name__ == "__main__":
    main()
