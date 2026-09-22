#!/usr/bin/env python3
"""Generic pipeline: same stages as train_wake_word.py, but mixes real
recordings into the synthetic positive_train set before augmentation, and
holds a separate set of real recordings back to test the finished model --
never used in training.

On launch, opens a file picker (Tk) for you to choose which wake-word config
(config/<model_name>.yaml) to train -- so this works for any wake word you've
already set up via train_wake_word.bat (option 1), not just one hardcoded
model. Real recordings are read from/written to real_samples/<model_name>/
(train/ and eval_holdout/), auto-created on first run for that model -- see
real_samples/README.txt.

Requires install.py to have been run first, and stock_multispeaker_voices/
augmentation_data/features_cache to be in place (auto-linked from another
install, or downloaded fresh -- see README.md).

Prerequisites:
    - Everything train_wake_word.bat needs (this repo's .venv)
    - Tkinter (ships with the standard python.org Windows installer; if
      you're on a minimal/embeddable Python and the file picker fails to
      open, install the standard python.org build instead)

Run via:
    .venv\\Scripts\\python.exe scripts\\train_with_real_samples.py
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent

MAIN_VENV_PY = REPO_ROOT / ".venv" / "Scripts" / "python.exe"
CONVERT_VENV_DIR = REPO_ROOT / ".venv-convert"
CONVERT_VENV_PY = CONVERT_VENV_DIR / "Scripts" / "python.exe"

OPENWAKEWORD_TRAIN_PY = REPO_ROOT / "openWakeWord" / "openwakeword" / "train.py"
CONFIG_DIR = REPO_ROOT / "config"
EXAMPLE_CONFIG_NAME = "wake_word.example.yaml"

DEFAULT_CUTOFF = 0.7
TARGET_SR = 16000

ASSET_DIRS = [
    REPO_ROOT / "stock_multispeaker_voices",
    REPO_ROOT / "augmentation_data" / "room_impulse_responses",
    REPO_ROOT / "features_cache",
]

REAL_SAMPLES_README = (
    "Real recordings of the wake phrase, mixed into the synthetic positive\n"
    "training set. Copy your own WAV files here (one phrase per file -- use\n"
    "the WAV splitter tools in this repo's root if you recorded them\n"
    "back-to-back in one take).\n"
)
EVAL_HOLDOUT_README = (
    "A few real recordings held back from training entirely, used to test the\n"
    "finished model automatically at the end of train_with_real_samples.py.\n"
    "Different takes to the ones in train/.\n"
)


@dataclass
class Ctx:
    config_path: Path
    model_name: str
    display_name: str
    cutoff: float
    real_train_dir: Path
    real_eval_dir: Path


def step(msg: str) -> None:
    print(f"\n=== {msg} ===")


def fail(msg: str) -> None:
    print(f"\nERROR: {msg}")
    sys.exit(1)


def select_config() -> Path:
    step("Selecting wake-word config")
    candidates = sorted(p for p in CONFIG_DIR.glob("*.yaml") if p.name != EXAMPLE_CONFIG_NAME)
    if not candidates:
        fail(
            f"No wake-word config found in {CONFIG_DIR} (besides the template).\n"
            "Run train_wake_word.bat (option 1) first to create one, or copy\n"
            f"config\\{EXAMPLE_CONFIG_NAME} to config\\<your_wake_word>.yaml and fill it in."
        )

    try:
        import tkinter as tk
        from tkinter import filedialog
    except ImportError:
        fail(
            "Tkinter is not available in this Python install, so the file picker\n"
            "can't open. Install the standard python.org Windows build (Tkinter\n"
            "ships with it), or edit this script to hardcode CONFIG_PATH instead."
        )

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    chosen = filedialog.askopenfilename(
        title="Select the wake-word config to train",
        initialdir=str(CONFIG_DIR),
        filetypes=[("YAML config", "*.yaml")],
    )
    root.destroy()

    if not chosen:
        fail("No config selected.")
    chosen_path = Path(chosen)
    if chosen_path.name == EXAMPLE_CONFIG_NAME:
        fail(
            f"{EXAMPLE_CONFIG_NAME} is the template, not a real config.\n"
            f"Copy it to config\\<your_wake_word>.yaml, fill it in, and pick that instead."
        )
    print(f"  Using: {chosen_path}")
    return chosen_path


def load_ctx(config_path: Path) -> Ctx:
    data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    model_name = data.get("model_name")
    if not model_name:
        fail(f"{config_path} has no model_name field.")
    phrases = data.get("target_phrase") or []
    display_name = phrases[0].title() if phrases else model_name.replace("_", " ").title()

    raw = input(f"Probability cutoff for the deployed model (Enter to keep {DEFAULT_CUTOFF}): ").strip()
    if not raw:
        cutoff = DEFAULT_CUTOFF
    else:
        try:
            cutoff = float(raw)
        except ValueError:
            print(f"  Not a number, using default {DEFAULT_CUTOFF}.")
            cutoff = DEFAULT_CUTOFF

    real_dir = REPO_ROOT / "real_samples" / model_name
    return Ctx(
        config_path=config_path,
        model_name=model_name,
        display_name=display_name,
        cutoff=cutoff,
        real_train_dir=real_dir / "train",
        real_eval_dir=real_dir / "eval_holdout",
    )


def ensure_real_sample_dirs(ctx: Ctx) -> None:
    for d, readme_text in ((ctx.real_train_dir, REAL_SAMPLES_README), (ctx.real_eval_dir, EVAL_HOLDOUT_README)):
        d.mkdir(parents=True, exist_ok=True)
        readme = d / "README.txt"
        if not readme.exists():
            readme.write_text(readme_text)


def check_prereqs(ctx: Ctx) -> None:
    step("Checking prerequisites")
    if not MAIN_VENV_PY.exists():
        fail(f"Main venv not found at {MAIN_VENV_PY}. Run install.py / install.bat first.")
    if not OPENWAKEWORD_TRAIN_PY.exists():
        fail(f"openWakeWord not found at {OPENWAKEWORD_TRAIN_PY}. Run install.py / install.bat first.")
    if not ctx.config_path.exists():
        fail(f"Config not found at {ctx.config_path}.")
    for d in ASSET_DIRS:
        has_real_files = d.exists() and any(p.name != "README.txt" for p in d.glob("*"))
        if not has_real_files:
            fail(
                f"{d} looks empty.\n"
                "Run download_assets.bat -- it automatically finds and links an\n"
                "existing copy from another install (see KNOWN_INSTALL_ROOTS in\n"
                "scripts/download_assets.py), or downloads fresh. See README.md."
            )

    ensure_real_sample_dirs(ctx)
    real_train = list(ctx.real_train_dir.glob("*.wav"))
    real_eval = list(ctx.real_eval_dir.glob("*.wav"))
    if not real_train:
        fail(f"No real recordings found in {ctx.real_train_dir}.")
    print(f"  OK ({len(real_train)} real training clips, {len(real_eval)} held out for eval)")


def run_stage(ctx: Ctx, flags: list[str]) -> None:
    env = dict(os.environ)
    env["PYTHONUTF8"] = "1"
    cmd = [str(MAIN_VENV_PY), str(OPENWAKEWORD_TRAIN_PY), "--training_config", str(ctx.config_path), *flags]
    print(f"  $ {' '.join(cmd)}")
    subprocess.run(cmd, env=env)


def generate_synthetic_clips(ctx: Ctx) -> None:
    step("Stage 1: generating synthetic clips")
    run_stage(ctx, ["--generate_clips"])
    positive_train = REPO_ROOT / "training" / ctx.model_name / "positive_train"
    if not positive_train.exists():
        fail(f"Expected {positive_train} after --generate_clips but it's not there.")


def _write_at_target_sr(src: Path, dest: Path) -> None:
    """Copy a WAV into place, resampling to TARGET_SR mono 16-bit PCM if it
    isn't already -- train.py hard-fails on any clip whose sample rate
    doesn't match its config, and real recordings (from a mic/DAC) are
    rarely already at 16kHz the way our synthetic clips are."""
    audio, sr = sf.read(str(src), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)  # downmix to mono
    if sr != TARGET_SR:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=TARGET_SR)
    audio_i16 = np.clip(audio * 32768.0, -32768, 32767).astype(np.int16)
    sf.write(str(dest), audio_i16, TARGET_SR, subtype="PCM_16")


def mix_in_real_recordings(ctx: Ctx) -> None:
    step("Mixing real recordings into positive_train")
    positive_train = REPO_ROOT / "training" / ctx.model_name / "positive_train"
    copied = 0
    for wav in ctx.real_train_dir.glob("*.wav"):
        _write_at_target_sr(wav, positive_train / f"real_{wav.name}")
        copied += 1
    print(f"  Copied {copied} real clip(s) into {positive_train} (resampled to {TARGET_SR}Hz mono as needed)")
    print(f"  (eval_holdout's {len(list(ctx.real_eval_dir.glob('*.wav')))} clip(s) were NOT touched -- reserved for testing)")


def augment_and_train(ctx: Ctx) -> Path:
    step("Stage 2+3: augmenting and training (the slow part)")
    run_stage(ctx, ["--augment_clips", "--train_model"])
    onnx_path = REPO_ROOT / "training" / f"{ctx.model_name}.onnx"
    if not onnx_path.exists():
        fail(
            f"Training did not produce {onnx_path}. Scroll up for the actual error "
            "(ignore the expected-and-harmless onnx_tf crash right after export)."
        )
    print(f"  Got {onnx_path}")
    return onnx_path


def ensure_convert_venv() -> None:
    step("Convert venv (onnx2tf / tensorflow / ai-edge-litert)")
    if CONVERT_VENV_PY.exists():
        print(f"  Already exists: {CONVERT_VENV_DIR}")
    else:
        subprocess.run([sys.executable, "-m", "venv", str(CONVERT_VENV_DIR)], check=True)
        subprocess.run([str(CONVERT_VENV_PY), "-m", "pip", "install", "-q", "--upgrade", "pip"], check=True)
    # Always (re-)run this, even if the venv already existed -- pip install is a
    # no-op for packages already satisfied, so this is cheap, and it's what
    # picks up a package added here after the venv was first created (bit us
    # once already: tf_keras got added to this list but an existing venv
    # silently kept skipping the whole install step).
    # onnx2tf's own package declares ZERO hard dependencies (by design, so it
    # doesn't force a specific onnx/tensorflow version on you) -- every one of
    # these has to be listed here explicitly, or it'll ModuleNotFoundError
    # deep inside onnx2tf the first time that code path is hit.
    subprocess.run(
        [str(CONVERT_VENV_PY), "-m", "pip", "install", "-q", "onnx2tf", "onnx==1.23.0", "onnx-graphsurgeon", "sng4onnx", "psutil", "tensorflow==2.17.1", "tf_keras==2.17.0", "ai-edge-litert==2.2.0"],
        check=True,
    )


def convert_to_tflite(ctx: Ctx, onnx_path: Path) -> Path:
    step("Converting ONNX -> TFLite (-kat prevents the axis-transpose bug)")
    out_dir = REPO_ROOT / "training" / f"{ctx.model_name}_tflite_output"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    cmd = [str(CONVERT_VENV_PY), "-m", "onnx2tf", "-i", str(onnx_path), "-o", str(out_dir), "-kat", "x"]
    print(f"  $ {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    tflite_path = out_dir / f"{ctx.model_name}_float32.tflite"
    if not tflite_path.exists():
        fail(f"Expected {tflite_path} after conversion but it's not there.")
    return tflite_path


def verify_shape(tflite_path: Path) -> None:
    step("Verifying input tensor shape is (1, 16, 96)")
    code = (
        "import ai_edge_litert.interpreter as tflite\n"
        f"i = tflite.Interpreter(model_path=r'{tflite_path}', num_threads=1)\n"
        "i.allocate_tensors()\n"
        "print(tuple(int(v) for v in i.get_input_details()[0]['shape']))\n"
    )
    result = subprocess.run([str(MAIN_VENV_PY), "-c", code], capture_output=True, text=True)
    shape_str = result.stdout.strip().splitlines()[-1] if result.stdout.strip() else ""
    print(f"  Shape: {shape_str}")
    if shape_str != "(1, 16, 96)":
        fail(f"Input shape is {shape_str or '<unknown>'}, expected (1, 16, 96). Do not deploy this model.\n{result.stderr}")
    print("  Correct.")


def package_output(ctx: Ctx, tflite_path: Path) -> Path:
    step("Packaging deployable files")
    out_dir = REPO_ROOT / "model_output"
    out_dir.mkdir(exist_ok=True)
    final_tflite = out_dir / f"{ctx.model_name}.tflite"
    shutil.copy(tflite_path, final_tflite)
    manifest = {
        "type": "openWakeWord",
        "wake_word": ctx.display_name,
        "model": f"{ctx.model_name}.tflite",
        "trained_languages": ["en"],
        "openWakeWord": {"probability_cutoff": ctx.cutoff},
    }
    final_json = out_dir / f"{ctx.model_name}.json"
    final_json.write_text(json.dumps(manifest, indent=2))
    print(f"  {final_tflite}")
    print(f"  {final_json}")
    return final_tflite


def eval_against_holdout(ctx: Ctx, packaged_tflite: Path) -> None:
    step("Testing against held-out real recordings (never seen in training)")
    wavs = sorted(ctx.real_eval_dir.glob("*.wav"))
    if not wavs:
        print("  No eval_holdout recordings found, skipping.")
        return
    cmd = [str(MAIN_VENV_PY), str(REPO_ROOT / "scripts" / "eval_real_voice.py"), str(packaged_tflite), *[str(w) for w in wavs]]
    subprocess.run(cmd)


def main() -> None:
    config_path = select_config()
    ctx = load_ctx(config_path)
    print(f"\nopenWakeWord pipeline -- {ctx.display_name} ({ctx.model_name}), retrained with real recordings mixed in")
    check_prereqs(ctx)
    generate_synthetic_clips(ctx)
    mix_in_real_recordings(ctx)
    onnx_path = augment_and_train(ctx)
    ensure_convert_venv()
    tflite_path = convert_to_tflite(ctx, onnx_path)
    verify_shape(tflite_path)
    packaged = package_output(ctx, tflite_path)
    eval_against_holdout(ctx, packaged)

    print(
        f"\n=== Done ===\n"
        f"Deployable files are in: {REPO_ROOT / 'model_output'}\n"
        f"  {ctx.model_name}.tflite\n"
        f"  {ctx.model_name}.json\n\n"
        "Copy both to your openWakeWord add-on's model folder AND each Assist "
        "Satellite's local external_wake_words folder, then set it as the active "
        "wake word in Home Assistant -- see PIPELINE_README.md's 'Deploy to Home "
        "Assistant' section."
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nCancelled.")
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        fail(f"Command failed (exit {e.returncode}): {' '.join(str(c) for c in e.cmd)}")
