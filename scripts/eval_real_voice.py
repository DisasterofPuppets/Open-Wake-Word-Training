"""Run one or more real human recordings of a wake phrase through a trained
.tflite model, using the genuine ai_edge_litert TFLite interpreter (NOT
openwakeword's Model() wrapper's silent onnx fallback), to get a hard
pass/fail score before deploying to Home Assistant.

Usage:
    python scripts/eval_real_voice.py <model.tflite> <recording1.wav> [recording2.wav ...]
    python scripts/eval_real_voice.py <model.tflite> --cutoff 0.6 *.wav

Each recording can contain multiple takes/utterances back to back — the
script reports the max score and top-5 scores across the whole file, not
just a single number, so you can see it fired on more than one utterance.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import librosa

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "training" / "openwakeword_repo"))

from openwakeword.model import Model  # noqa: E402

CHUNK = 1280  # 80ms @ 16kHz, openWakeWord's expected streaming chunk size


def load_16k_mono(path):
    audio, sr = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != 16000:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
    return np.clip(audio * 32768.0, -32768, 32767).astype(np.int16)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model", help="Path to the .tflite model to evaluate")
    parser.add_argument("recordings", nargs="+", help="One or more .wav files of real speech to test against")
    parser.add_argument("--cutoff", type=float, default=0.7, help="Detection threshold (default: 0.7, matching the openWakeWord .json probability_cutoff convention)")
    args = parser.parse_args()

    # Sanity-check the model's input shape before wasting time on a bad conversion.
    # openWakeWord classifier models must be (1, 16, 96) -- 16 timesteps of 96-dim
    # embeddings. A transposed (1, 96, 16) shape (or anything else) means the
    # onnx2tf conversion mangled the axes -- see WAKE_WORD_TRAINING.md section 7.
    import ai_edge_litert.interpreter as tflite
    interp = tflite.Interpreter(model_path=args.model, num_threads=1)
    interp.allocate_tensors()
    shape = tuple(interp.get_input_details()[0]["shape"])
    if shape != (1, 16, 96):
        print(f"WARNING: model input shape is {shape}, expected (1, 16, 96).")
        print("This almost certainly means the ONNX->TFLite conversion transposed the axes.")
        print("Re-run onnx2tf with '-kat <input_name>' to fix -- see WAKE_WORD_TRAINING.md section 7.\n")

    oww = Model(wakeword_models=[args.model], inference_framework="tflite")
    mdl_name = list(oww.models.keys())[0]
    print(f"Loaded model: {mdl_name}  (input shape: {shape})\n")

    any_failed = False
    for path in args.recordings:
        audio = load_16k_mono(path)
        oww.reset()
        scores = []
        for i in range(0, len(audio) - CHUNK + 1, CHUNK):
            chunk = audio[i:i + CHUNK]
            prediction = oww.predict(chunk)
            scores.append((i / 16000.0, prediction[mdl_name]))

        if not scores:
            print(f"=== {path} === (too short to evaluate, skipped)\n")
            continue

        max_t, max_score = max(scores, key=lambda x: x[1])
        fired = max_score >= args.cutoff
        any_failed = any_failed or not fired
        top5 = sorted(scores, key=lambda x: -x[1])[:5]
        print(f"=== {path} ===")
        print(f"  duration: {len(audio)/16000:.2f}s, chunks: {len(scores)}")
        print(f"  MAX score: {max_score:.4f} at t={max_t:.2f}s")
        print("  top 5 scores:", ", ".join(f"{t:.2f}s={s:.3f}" for t, s in top5))
        print(f"  cutoff is {args.cutoff} -> {'WOULD FIRE' if fired else 'would NOT fire'}\n")

    sys.exit(1 if any_failed else 0)


if __name__ == "__main__":
    main()
