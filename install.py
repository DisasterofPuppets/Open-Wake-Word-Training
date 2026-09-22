#!/usr/bin/env python3
"""One-shot setup for this pipeline (Windows only).

Normally launched via install.ps1 / install.bat, which also make sure git
and Python itself are present (installing them via winget if not) before
getting here. If you already have both, you can also just run:

    python install.py [--gpu | --cpu]

What this does:
  1. Creates a venv (.venv/) if one doesn't exist yet
  2. pip installs requirements.txt into it
  3. Installs torch/torchaudio separately, as CUDA or CPU wheels depending
     on --gpu/--cpu (auto-detected from your GPU if neither is passed)
  4. Clones openWakeWord (if not already present) and applies the
     Windows-compatibility patch from patches/
  5. pip installs openWakeWord (editable) into the venv
  6. Downloads openWakeWord's own small required feature/VAD models
     (melspectrogram, embedding, silero_vad) -- NOT the same as the large
     downloadable assets below; these are required just to import/run
     openwakeword at all
  7. Drops patches/sitecustomize.py into the venv's site-packages
  8. Creates the folder skeleton for the large downloadable assets
     (voice pack / RIRs / feature cache) with a README.txt in each

What this does NOT do (see README.md for these):
  - Download the multi-speaker voice pack, RIR set, or ACAV100M features
    -- run download_assets.bat / scripts/download_assets.py for that
  - Pick a wake phrase for you -- run train_wake_word.bat /
    scripts/train_wake_word.py, which handles the config for you

Safe to re-run: every step is skip-if-already-done.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
VENV_DIR = REPO_ROOT / ".venv"
VENV_PY = VENV_DIR / "Scripts" / "python.exe"
OPENWAKEWORD_DIR = REPO_ROOT / "openWakeWord"

# CUDA build this pipeline was verified against; bump if PyTorch moves on.
CUDA_INDEX_URL = "https://download.pytorch.org/whl/cu128"
TORCH_VERSION = "2.9.1"

DATA_DIRS_WITH_README = {
    "stock_multispeaker_voices": (
        "Multi-speaker Piper voice pack (.onnx + .onnx.json pairs), ~430MB.\n"
        "Run download_assets.bat (or scripts/download_assets.py) to fetch this\n"
        "automatically -- see README.md's 'Download the large training assets'.\n"
    ),
    "augmentation_data/room_impulse_responses": (
        "Room impulse response WAVs for reverb augmentation, ~10MB.\n"
        "Run download_assets.bat (or scripts/download_assets.py) to fetch this\n"
        "automatically -- see README.md's 'Download the large training assets'.\n"
    ),
    "features_cache": (
        "Precomputed negative-audio features (ACAV100M, ~16GB) + false-positive\n"
        "validation set (~180MB). Run download_assets.bat (or\n"
        "scripts/download_assets.py) to fetch these -- it'll show you the total\n"
        "size and ask to confirm before starting. A free Hugging Face token\n"
        "(huggingface.co/settings/tokens) makes the big file much faster; the\n"
        "script will prompt for one.\n"
    ),
    "training": "Training output (checkpoints, generated clips, features) lands here.\n",
    "model_output": "Final .tflite + .json per trained wake word lands here.\n",
    "config": "Copy wake_word.example.yaml to <your_wake_word>.yaml here and edit it.\n",
    "real_samples": (
        "Real recordings, mixed into synthetic training by\n"
        "train_with_real_samples.py. Organised per wake word:\n"
        "  real_samples/<model_name>/train/         -- mixed into training\n"
        "  real_samples/<model_name>/eval_holdout/  -- held back to test the model\n"
        "<model_name> must match the model_name field in that wake word's\n"
        "config/<model_name>.yaml. Both subfolders are created automatically the\n"
        "first time you run train_with_real_samples.py and pick that config.\n"
    ),
}


def run(cmd: list[str], **kwargs) -> None:
    print(f"  $ {' '.join(str(c) for c in cmd)}")
    subprocess.run(cmd, check=True, **kwargs)


def step(msg: str) -> None:
    print(f"\n=== {msg} ===")


def ensure_venv() -> None:
    step("Python venv")
    if VENV_PY.exists():
        print(f"  Already exists: {VENV_DIR}")
        return
    run([sys.executable, "-m", "venv", str(VENV_DIR)])


def install_requirements() -> None:
    step("Installing requirements.txt")
    run([str(VENV_PY), "-m", "pip", "install", "-q", "--upgrade", "pip"])
    run([str(VENV_PY), "-m", "pip", "install", "-q", "-r", str(REPO_ROOT / "requirements.txt")])


def torch_already_installed(want_gpu: bool) -> bool:
    result = subprocess.run(
        [str(VENV_PY), "-c", "import torch; print(torch.__version__, torch.cuda.is_available())"],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        return False
    version, cuda_available = result.stdout.strip().split()
    return version.startswith(TORCH_VERSION) and (cuda_available == "True") == want_gpu


def install_torch(use_gpu: bool) -> None:
    step(f"Installing torch/torchaudio ({'GPU/CUDA' if use_gpu else 'CPU-only'})")
    if torch_already_installed(use_gpu):
        print("  Already installed with matching version and GPU/CPU mode.")
        return
    cmd = [str(VENV_PY), "-m", "pip", "install", "-q",
           f"torch=={TORCH_VERSION}", f"torchaudio=={TORCH_VERSION}"]
    if use_gpu:
        cmd += ["--index-url", CUDA_INDEX_URL]
    run(cmd)


def setup_openwakeword() -> None:
    step("openWakeWord clone + Windows patch")
    if not OPENWAKEWORD_DIR.exists():
        run(["git", "clone", "-q", "https://github.com/dscripka/openWakeWord", str(OPENWAKEWORD_DIR)])
    else:
        print(f"  Already cloned: {OPENWAKEWORD_DIR}")

    patch_path = REPO_ROOT / "patches" / "openwakeword-windows-fixes.patch"
    check = subprocess.run(
        ["git", "apply", "--check", "--reverse", str(patch_path)],
        cwd=OPENWAKEWORD_DIR, capture_output=True,
    )
    if check.returncode == 0:
        print("  Patch already applied, skipping.")
    else:
        run(["git", "apply", str(patch_path)], cwd=OPENWAKEWORD_DIR)

    run([str(VENV_PY), "-m", "pip", "install", "-q", "-e", str(OPENWAKEWORD_DIR)])


def download_feature_models() -> None:
    step("openWakeWord feature/VAD models (melspectrogram, embedding, silero_vad)")
    # download_models(model_names=[]) (the default) would ALSO fetch every
    # official pretrained wake-word model (hey_jarvis, alexa, ...) -- passing
    # a dummy non-matching name skips that and only gets the always-required
    # feature/VAD models train.py needs.
    run([
        str(VENV_PY), "-c",
        "import openwakeword.utils as u; u.download_models(model_names=['__none__'])",
    ])


def install_sitecustomize() -> None:
    step("sitecustomize.py runtime patches")
    site_packages = VENV_DIR / "Lib" / "site-packages"
    dest = site_packages / "sitecustomize.py"
    src = REPO_ROOT / "patches" / "sitecustomize.py"
    if dest.exists() and dest.read_text() == src.read_text():
        print(f"  Already in place: {dest}")
        return
    shutil.copy(src, dest)
    print(f"  Copied to: {dest}")


def create_data_skeleton() -> None:
    step("Folder skeleton for downloadable assets")
    for rel_path, readme_text in DATA_DIRS_WITH_README.items():
        d = REPO_ROOT / rel_path
        d.mkdir(parents=True, exist_ok=True)
        readme = d / "README.txt"
        if not readme.exists():
            readme.write_text(readme_text)
        print(f"  {rel_path}/")


def detect_gpu() -> bool:
    try:
        result = subprocess.run(["nvidia-smi"], capture_output=True)
        return result.returncode == 0
    except FileNotFoundError:
        return False


def main() -> None:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--gpu", action="store_true", help="Install CUDA-enabled torch")
    mode.add_argument("--cpu", action="store_true", help="Install CPU-only torch")
    args = parser.parse_args()

    if args.gpu:
        use_gpu = True
    elif args.cpu:
        use_gpu = False
    else:
        use_gpu = detect_gpu()
        print(f"No --gpu/--cpu given -- auto-detected {'an NVIDIA GPU' if use_gpu else 'no NVIDIA GPU'}, using {'GPU' if use_gpu else 'CPU'} torch.")

    print(f"Setting up in: {REPO_ROOT}")
    ensure_venv()
    install_requirements()
    install_torch(use_gpu)
    setup_openwakeword()
    download_feature_models()
    install_sitecustomize()
    create_data_skeleton()

    print(
        "\n=== Done ===\n"
        "Next steps:\n"
        "  1. Run download_assets.bat (or scripts/download_assets.py) to fetch the\n"
        "     voice pack / RIRs / feature cache -- it shows the total size (~16.5GB)\n"
        "     and asks to confirm before downloading anything.\n"
        "  2. Run train_wake_word.bat (or scripts/train_wake_word.py) and enter your\n"
        "     wake phrase when prompted -- it handles config, training, conversion,\n"
        "     verification, and packaging end to end.\n"
    )


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as e:
        print(f"\nSetup step failed (exit {e.returncode}): {' '.join(str(c) for c in e.cmd)}")
        sys.exit(1)
