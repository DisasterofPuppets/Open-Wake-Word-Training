#!/usr/bin/env python3
"""End-to-end per-wake-word pipeline: config -> generate -> augment -> train
-> convert to TFLite (with the critical anti-transpose flag) -> verify shape
-> package the HA-ready .tflite + .json -> optionally test against real
recordings you supply.

Interactive by default (prompts for the wake phrase etc). Every prompt can
also be answered via a CLI flag instead -- see --help -- which is how the
GUI installer drives this script non-interactively.

Run via train_wake_word.bat (double-click), or directly:
    <repo>\\.venv\\Scripts\\python.exe scripts\\train_wake_word.py

Requires install.py (or install.bat) to have been run first, and the
one-time downloadable assets (voice pack / RIRs / feature cache) to already
be in place -- see README.md.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

MAIN_VENV_PY = REPO_ROOT / ".venv" / "Scripts" / "python.exe"
CONVERT_VENV_DIR = REPO_ROOT / ".venv-convert"
CONVERT_VENV_PY = CONVERT_VENV_DIR / "Scripts" / "python.exe"

OPENWAKEWORD_TRAIN_PY = REPO_ROOT / "openWakeWord" / "openwakeword" / "train.py"
CONFIG_TEMPLATE = REPO_ROOT / "config" / "wake_word.example.yaml"

ASSET_DIRS = [
    REPO_ROOT / "stock_multispeaker_voices",
    REPO_ROOT / "augmentation_data" / "room_impulse_responses",
    REPO_ROOT / "features_cache",
]


def step(msg: str) -> None:
    print(f"\n=== {msg} ===")


def ask(prompt: str, default: str | None = None) -> str:
    suffix = f" [{default}]" if default is not None else ""
    val = input(f"{prompt}{suffix}: ").strip()
    return val if val else (default or "")


def fail(msg: str) -> None:
    print(f"\nERROR: {msg}")
    sys.exit(1)


def check_prereqs() -> None:
    step("Checking prerequisites")
    if not MAIN_VENV_PY.exists():
        fail(f"Main venv not found at {MAIN_VENV_PY}. Run install.py / install.bat first.")
    if not OPENWAKEWORD_TRAIN_PY.exists():
        fail(f"openWakeWord not found at {OPENWAKEWORD_TRAIN_PY}. Run install.py / install.bat first.")
    for d in ASSET_DIRS:
        has_real_files = d.exists() and any(p.name != "README.txt" for p in d.glob("*"))
        if not has_real_files:
            fail(
                f"{d} looks empty (only the placeholder README.txt, or missing entirely).\n"
                "Run download_assets.bat / scripts/download_assets.py first -- see README.md."
            )
    print("  OK")


def sanitize_model_name(raw: str) -> str:
    return raw.strip().lower().replace(" ", "_").replace("-", "_")


def write_config(model_name: str, phrase: str, n_samples: int) -> Path:
    step("Writing training config")
    template_text = CONFIG_TEMPLATE.read_text()
    project_root_str = str(REPO_ROOT).replace("\\", "/")
    config_text = (
        template_text
        .replace('"<your_wake_word>"', f'"{model_name}"')
        .replace('"<your wake phrase>"', f'"{phrase}"')
        .replace("<PROJECT_ROOT>", project_root_str)
        .replace("n_samples: 10000", f"n_samples: {n_samples}")
    )
    config_path = REPO_ROOT / "config" / f"{model_name}.yaml"
    config_path.write_text(config_text)
    print(f"  Wrote {config_path}")
    return config_path


def run_training(config_path: Path, model_name: str) -> Path:
    step("Generating clips, augmenting, and training (this is the slow part)")
    env = dict(os.environ)
    env["PYTHONUTF8"] = "1"  # avoids a UnicodeEncodeError from torch.onnx's checkmark output
    cmd = [
        str(MAIN_VENV_PY), str(OPENWAKEWORD_TRAIN_PY),
        "--training_config", str(config_path),
        "--generate_clips", "--augment_clips", "--train_model",
    ]
    print(f"  $ {' '.join(cmd)}")
    # Not check=True: train.py unconditionally attempts its own broken onnx_tf
    # conversion after training and always exits non-zero because of it, even
    # though the .onnx file we actually need was already written successfully.
    subprocess.run(cmd, env=env)

    onnx_path = REPO_ROOT / "training" / f"{model_name}.onnx"
    if not onnx_path.exists():
        fail(
            f"Training did not produce {onnx_path}. Scroll up in the output above for "
            "the actual error (likely happened before the export step, not the "
            "expected-and-harmless onnx_tf crash after it)."
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


def convert_to_tflite(onnx_path: Path, model_name: str) -> Path:
    step("Converting ONNX -> TFLite (with -kat to prevent the axis-transpose bug)")
    out_dir = REPO_ROOT / "training" / f"{model_name}_tflite_output"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    cmd = [
        str(CONVERT_VENV_PY), "-m", "onnx2tf",
        "-i", str(onnx_path), "-o", str(out_dir), "-kat", "x",
    ]
    print(f"  $ {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

    tflite_path = out_dir / f"{model_name}_float32.tflite"
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
        fail(
            f"Input shape is {shape_str or '<unknown -- see stderr below>'}, expected (1, 16, 96).\n"
            f"{result.stderr}\n"
            "This means the conversion transposed the axes again -- do not deploy this model."
        )
    print("  Correct.")


def package_output(tflite_path: Path, model_name: str, display_name: str, cutoff: float) -> Path:
    step("Packaging deployable files")
    out_dir = REPO_ROOT / "model_output"
    out_dir.mkdir(exist_ok=True)
    final_tflite = out_dir / f"{model_name}.tflite"
    shutil.copy(tflite_path, final_tflite)

    manifest = {
        "type": "openWakeWord",
        "wake_word": display_name,
        "model": f"{model_name}.tflite",
        "trained_languages": ["en"],
        "openWakeWord": {"probability_cutoff": cutoff},
    }
    final_json = out_dir / f"{model_name}.json"
    final_json.write_text(json.dumps(manifest, indent=2))

    print(f"  {final_tflite}")
    print(f"  {final_json}")
    return out_dir


def maybe_eval_real_voice(model_name: str, test_wavs: list[str]) -> None:
    step("Optional: test against real recordings")
    paths = test_wavs
    if not paths:
        raw = input(
            "Paths to one or more WAV recordings of the phrase, comma-separated "
            "(or leave blank to skip): "
        ).strip()
        if not raw:
            print("  Skipped.")
            return
        paths = [p.strip().strip('"') for p in raw.split(",") if p.strip()]

    missing = [p for p in paths if not Path(p).exists()]
    if missing:
        print(f"  Skipping -- file(s) not found: {missing}")
        return

    packaged = REPO_ROOT / "model_output" / f"{model_name}.tflite"
    cmd = [str(MAIN_VENV_PY), str(REPO_ROOT / "scripts" / "eval_real_voice.py"), str(packaged), *paths]
    subprocess.run(cmd)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--name", default=None, help="Model name, short, no spaces (e.g. hey_computer)")
    parser.add_argument("--phrase", default=None, help="Wake phrase to train (what you'll actually say)")
    parser.add_argument("--display-name", default=None, help="Display name for Home Assistant")
    parser.add_argument("--samples", type=int, default=None, help="Number of synthetic training samples (default 10000)")
    parser.add_argument("--cutoff", type=float, default=None, help="Detection probability cutoff (default 0.7)")
    parser.add_argument("--test-wav", action="append", default=[], help="Path to a real recording to test against (repeatable); skips the prompt if given at least once")
    parser.add_argument("--skip-eval", action="store_true", help="Skip the real-voice test step entirely, no prompt")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    print("openWakeWord Windows Training Pipeline -- per-wake-word setup")
    check_prereqs()

    raw_name = args.name if args.name is not None else ask("Model name (short, no spaces -- e.g. hey_computer)")
    model_name = sanitize_model_name(raw_name)
    if not model_name:
        fail("Model name can't be empty.")

    phrase = args.phrase if args.phrase is not None else ask(
        "Wake phrase to train (what you'll actually say)", default=model_name.replace("_", " ")
    )
    display_name = args.display_name if args.display_name is not None else ask(
        "Display name for Home Assistant", default=phrase.title()
    )
    n_samples = args.samples if args.samples is not None else int(
        ask("Number of synthetic training samples (20,000+ recommended)", default="10000")
    )
    cutoff = args.cutoff if args.cutoff is not None else float(
        ask("Detection probability cutoff", default="0.7")
    )

    config_path = write_config(model_name, phrase, n_samples)
    onnx_path = run_training(config_path, model_name)
    ensure_convert_venv()
    tflite_path = convert_to_tflite(onnx_path, model_name)
    verify_shape(tflite_path)
    out_dir = package_output(tflite_path, model_name, display_name, cutoff)

    if args.skip_eval:
        print("\n--skip-eval given -- not testing against real recordings.")
    else:
        maybe_eval_real_voice(model_name, args.test_wav)

    print(
        f"\n=== Done ===\n"
        f"Deployable files are in: {out_dir}\n"
        f"  {model_name}.tflite\n"
        f"  {model_name}.json\n\n"
        "Remaining step (manual, paths are specific to your Home Assistant setup):\n"
        "copy both files to your openWakeWord add-on's model folder AND each Assist "
        "Satellite's local wake-word folder, then set it as the active wake word -- "
        "see README.md's 'Deploy to Home Assistant' section."
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nCancelled.")
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        fail(f"Command failed (exit {e.returncode}): {' '.join(str(c) for c in e.cmd)}")
