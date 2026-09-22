"""Drop-in replacement for piper-sample-generator's generate_samples().

The original (github.com/rhasspy/piper-sample-generator) historically required
the Linux-only piper-phonemize package (recent versions have moved away from
that -- see README's "Alternative: WSL2 + the real generator" section -- but
this native-Windows version remains useful if you'd rather avoid WSL entirely).

This version produces the same kind of output (a folder of 16kHz mono WAV
clips of the given text) using the official piper-tts PyPI package (has a
prebuilt Windows wheel) across a pool of multi-speaker stock Piper voices,
for speaker diversity roughly matching or exceeding the original tool's
single multi-speaker checkpoint approach.

Usage (matches enough of the original signature to swap into openWakeWord's
train.py with a one-line import change):

    from generate_samples_piper import generate_samples
    generate_samples(text=["hey computer"], output_dir="...", max_samples=1000, ...)

Voice pool location defaults to <repo_root>/stock_multispeaker_voices/
(see README for the download list); override with the
OWW_VOICE_POOL_DIR environment variable if you keep it elsewhere.
"""

from __future__ import annotations

import os
import random
import uuid
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Union

import numpy as np
import soundfile as sf
import librosa
from tqdm import tqdm

from piper.voice import PiperVoice
from piper.config import SynthesisConfig

VOICE_POOL_DIR = Path(
    os.environ.get(
        "OWW_VOICE_POOL_DIR",
        str(Path(__file__).resolve().parent.parent / "stock_multispeaker_voices"),
    )
)
TARGET_SR = 16000

_VOICE_CACHE: dict[str, PiperVoice] = {}


def _discover_voices() -> List[Path]:
    return sorted(VOICE_POOL_DIR.glob("*.onnx"))


def _load_voice(onnx_path: Path) -> PiperVoice:
    key = str(onnx_path)
    if key not in _VOICE_CACHE:
        json_path = onnx_path.with_suffix(".onnx.json")
        _VOICE_CACHE[key] = PiperVoice.load(onnx_path, json_path, use_cuda=True)
    return _VOICE_CACHE[key]


def _random_synth_config(
    length_scales: Sequence[float],
    noise_scales: Sequence[float],
    noise_scale_ws: Sequence[float],
    num_speakers: int,
) -> SynthesisConfig:
    return SynthesisConfig(
        speaker_id=random.randrange(num_speakers) if num_speakers > 1 else None,
        length_scale=random.choice(length_scales),
        noise_scale=random.choice(noise_scales),
        noise_w_scale=random.choice(noise_scale_ws),
    )


def generate_samples(
    text: Union[str, Sequence[str]],
    output_dir: Union[str, Path],
    max_samples: Optional[int] = None,
    batch_size: int = 1,  # unused (no batching benefit with onnxruntime CPU/GPU single-session calls here); kept for signature compatibility
    length_scales: Sequence[float] = (0.75, 1.0, 1.25),
    noise_scales: Sequence[float] = (0.667,),
    noise_scale_ws: Sequence[float] = (0.8,),
    file_names: Optional[Iterable[str]] = None,
    auto_reduce_batch_size: bool = True,  # unused, kept for signature compatibility
    **_ignored_kwargs,
) -> None:
    """Generate `max_samples` synthetic WAV clips of `text` phrase(s) into `output_dir`.

    `text` may be a single phrase or a list of phrases (one is chosen at random
    per clip, matching the behavior openWakeWord's train.py relies on for its
    adversarial negative-phrase lists).
    """
    phrases = [text] if isinstance(text, str) else list(text)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    voice_paths = _discover_voices()
    if not voice_paths:
        raise RuntimeError(f"No .onnx voices found in {VOICE_POOL_DIR}")

    if max_samples is None:
        max_samples = len(phrases)

    names = list(file_names) if file_names is not None else None

    generated = 0
    with tqdm(total=max_samples, desc=f"  Generating clips ({output_dir.name})", unit="clip") as pbar:
        while generated < max_samples:
            voice_path = random.choice(voice_paths)
            voice = _load_voice(voice_path)
            num_speakers = voice.config.num_speakers or 1
            phrase = random.choice(phrases)
            cfg = _random_synth_config(length_scales, noise_scales, noise_scale_ws, num_speakers)

            chunks = list(voice.synthesize(phrase, cfg))
            if not chunks:
                continue
            audio = np.concatenate(
                [np.frombuffer(c.audio_int16_bytes, dtype=np.int16) for c in chunks]
            )
            sr = chunks[0].sample_rate

            if sr != TARGET_SR:
                audio_f = audio.astype(np.float32) / 32768.0
                audio_f = librosa.resample(audio_f, orig_sr=sr, target_sr=TARGET_SR)
                audio = np.clip(audio_f * 32768.0, -32768, 32767).astype(np.int16)

            if names is not None and generated < len(names):
                fname = names[generated]
            else:
                fname = f"{uuid.uuid4().hex}.wav"

            sf.write(str(output_dir / fname), audio, TARGET_SR, subtype="PCM_16")
            generated += 1
            pbar.update(1)
