#!/usr/bin/env python3
"""Download the large, one-time-only training assets this pipeline needs:
multi-speaker Piper voices (~430MB), MIT room impulse responses (~10MB), and
openWakeWord's precomputed ACAV100M negative-audio feature cache (~16GB +
~180MB validation set). Safe to re-run -- files that already exist at the
expected size are skipped.

No Hugging Face account or API key is required for any of this -- all three
sources are public repos. An optional free HF token only speeds up the one
large file; without one it still works, just slower.

Interactive by default (asks before downloading anything, and offers to use
an already-downloaded copy instead). All prompts can also be answered via
CLI flags instead -- see --help -- which is how the GUI installer drives
this script non-interactively.

Run via download_assets.bat (double-click), or directly:
    <repo>\\.venv\\Scripts\\python.exe scripts\\download_assets.py
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

VOICE_DIR = REPO_ROOT / "stock_multispeaker_voices"
RIR_DIR = REPO_ROOT / "augmentation_data" / "room_impulse_responses"
FEATURES_DIR = REPO_ROOT / "features_cache"

PIPER_VOICES_BASE = "https://huggingface.co/rhasspy/piper-voices/resolve/main"
# (HF path stem, approx .onnx size in bytes -- used only for the upfront estimate)
VOICES = [
    ("en/en_US/libritts_r/medium/en_US-libritts_r-medium", 75_000_000),
    ("en/en_US/libritts/high/en_US-libritts-high", 131_000_000),
    ("en/en_GB/vctk/medium/en_GB-vctk-medium", 74_000_000),
    ("en/en_US/l2arctic/medium/en_US-l2arctic-medium", 74_000_000),
    ("en/en_US/arctic/medium/en_US-arctic-medium", 74_000_000),
]

RIR_API_URL = "https://huggingface.co/api/datasets/davidscripka/MIT_environmental_impulse_responses/tree/main/16khz"
RIR_BASE = "https://huggingface.co/datasets/davidscripka/MIT_environmental_impulse_responses/resolve/main"
RIR_APPROX_TOTAL = 8_400_000

FEATURES_BASE = "https://huggingface.co/datasets/davidscripka/openwakeword_features/resolve/main"
FEATURE_FILES = [
    ("openwakeword_features_ACAV100M_2000_hrs_16bit.npy", 17_280_000_128),
    ("validation_set_features.npy", 184_836_608),
]

# >>> Other installs of this same pipeline to check for already-downloaded
# >>> assets before asking/downloading -- add your own paths here.
KNOWN_INSTALL_ROOTS = [
    Path(r"E:\Live Locally Hosted\Open Wake Word Training"),
]

# Placeholder files install.py creates in an empty asset folder -- safe to
# clear automatically when linking a real copy in over the top of one.
_PLACEHOLDER_ONLY_NAMES = {"README.txt"}


def find_in_known_roots(rel_path: str) -> Path | None:
    """Look for `rel_path` (e.g. "stock_multispeaker_voices") under this
    project's own parent folder and every KNOWN_INSTALL_ROOTS entry, skipping
    this project itself. Returns the first match that exists."""
    search_roots = [REPO_ROOT.parent] + KNOWN_INSTALL_ROOTS
    for root in search_roots:
        candidate = root / rel_path
        if candidate == (REPO_ROOT / rel_path):
            continue
        if candidate.exists():
            return candidate
    return None


def _clear_if_placeholder_only(dest: Path) -> None:
    if not dest.exists():
        return
    contents = list(dest.iterdir())
    if all(p.name in _PLACEHOLDER_ONLY_NAMES for p in contents):
        shutil.rmtree(dest)


def _remove_readme_once_populated(dest: Path) -> None:
    """Remove install.py's placeholder README.txt from `dest` once it has
    real files in it. Some downstream code (openwakeword's train.py, for
    room_impulse_responses) does a raw directory scan with no file-type
    filter, so a leftover README.txt sitting alongside real data gets picked
    up as if it were one of the real files and crashes."""
    readme = dest / "README.txt"
    if readme.exists() and any(p.name != "README.txt" for p in dest.glob("*")):
        readme.unlink()


def junction(dest: Path, src: Path) -> bool:
    """Create an NTFS junction dest -> src (no admin rights needed, unlike a
    symlink). Returns True if dest now points at (or already has) real data."""
    _clear_if_placeholder_only(dest)
    if dest.exists():
        return True
    dest.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        ["cmd", "/c", "mklink", "/J", str(dest), str(src)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"  Could not link to {src} ({result.stderr.strip()}) -- will fall back to asking/downloading.")
        return False
    print(f"  Found an existing copy at {src} -- linked instead of downloading (no extra disk used).")
    return True



def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}TB"


def ask(prompt: str) -> str:
    return input(prompt).strip()


def clean_path(raw: str) -> Path:
    return Path(raw.strip().strip('"').strip("'"))


def request_with_auth(url: str, token: str | None) -> urllib.request.Request:
    req = urllib.request.Request(url)
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    return req


def download_with_progress(url: str, dest: Path, token: str | None = None, label: str | None = None) -> None:
    label = label or dest.name
    req = request_with_auth(url, token)
    with urllib.request.urlopen(req) as resp:
        total = int(resp.headers.get("Content-Length", 0))
        if dest.exists() and total and dest.stat().st_size == total:
            print(f"  [skip] {label} already downloaded ({human(total)})")
            return
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(dest.suffix + ".part")
        downloaded = 0
        last_print = 0.0
        with open(tmp, "wb") as f:
            while True:
                chunk = resp.read(1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)
                now = time.time()
                if now - last_print > 1.0:
                    pct = f"{downloaded / total * 100:.0f}%" if total else "?"
                    print(f"\r  {label}: {human(downloaded)} / {human(total)} ({pct})", end="", flush=True)
                    last_print = now
        print(f"\r  {label}: {human(downloaded)} done" + " " * 20)
        tmp.replace(dest)


# ---------------------------------------------------------------------------
# "I already have this" support -- each has a CLI-arg override that skips
# the interactive prompt entirely (used by the GUI, which can't answer input()).
# ---------------------------------------------------------------------------

def resolve_existing_folder(prompt: str, override: str | None) -> Path | None:
    if override is not None:
        p = clean_path(override)
        if not p.is_dir():
            print(f"  --path '{p}' is not a folder that exists -- will download instead.")
            return None
        return p
    raw = ask(prompt)
    if not raw:
        return None
    p = clean_path(raw)
    if not p.is_dir():
        print(f"  '{p}' is not a folder that exists -- will download instead.")
        return None
    return p


def resolve_existing_file(prompt: str, expected_size: int, override: str | None, assume_yes: bool) -> Path | None:
    if override is not None:
        p = clean_path(override)
        if not p.is_file():
            print(f"  --path '{p}' is not a file that exists -- will download instead.")
            return None
        actual = p.stat().st_size
        if actual != expected_size and not assume_yes:
            print(f"  Size mismatch: expected {human(expected_size)}, found {human(actual)} -- will download instead.")
            return None
        return p
    raw = ask(prompt)
    if not raw:
        return None
    p = clean_path(raw)
    if not p.is_file():
        print(f"  '{p}' is not a file that exists -- will download instead.")
        return None
    actual = p.stat().st_size
    if actual != expected_size:
        print(f"  Size mismatch: expected {human(expected_size)}, found {human(actual)}.")
        if ask("  Use it anyway (Y/N)? ").lower() != "y":
            return None
    return p


# ---------------------------------------------------------------------------
# Voice pack
# ---------------------------------------------------------------------------

def download_voice_pack(existing_path: str | None) -> None:
    print("\n=== Multi-speaker Piper voice pack ===")
    expected = []
    for stem, _approx in VOICES:
        name = stem.rsplit("/", 1)[-1]
        expected += [f"{name}.onnx", f"{name}.onnx.json"]

    missing = [n for n in expected if not (VOICE_DIR / n).exists()]
    if not missing:
        _remove_readme_once_populated(VOICE_DIR)
        print("  Already fully present, skipping.")
        return

    auto_src = find_in_known_roots("stock_multispeaker_voices")
    if auto_src and all((auto_src / n).exists() for n in expected) and junction(VOICE_DIR, auto_src):
        return

    src = resolve_existing_folder(
        f"  {len(missing)} of {len(expected)} voice file(s) missing. If you already have\n"
        "  the Piper voice pack downloaded elsewhere, enter that folder's path\n"
        "  (or leave blank to download): ",
        existing_path,
    )
    if src:
        copied = 0
        for name in list(missing):
            candidate = src / name
            if candidate.exists():
                shutil.copy2(candidate, VOICE_DIR / name)
                missing.remove(name)
                copied += 1
        print(f"  Copied {copied} file(s) from {src}.")
        if not missing:
            _remove_readme_once_populated(VOICE_DIR)
            return
        print(f"  Still need to download: {missing}")

    for stem, _approx in VOICES:
        name = stem.rsplit("/", 1)[-1]
        for ext in (".onnx", ".onnx.json"):
            fname = f"{name}{ext}"
            if fname not in missing:
                continue
            url = f"{PIPER_VOICES_BASE}/{stem}{ext}"
            download_with_progress(url, VOICE_DIR / fname, label=fname)
    _remove_readme_once_populated(VOICE_DIR)


# ---------------------------------------------------------------------------
# Room impulse responses
# ---------------------------------------------------------------------------

def list_rir_files() -> list[tuple[str, int]]:
    req = urllib.request.Request(RIR_API_URL)
    with urllib.request.urlopen(req) as resp:
        data = json.loads(resp.read())
    return [(entry["path"].split("/", 1)[-1], entry.get("size", 0)) for entry in data if entry.get("type") == "file"]


def download_rirs(existing_path: str | None) -> None:
    print("\n=== Room impulse responses (reverb augmentation) ===")
    existing = list(RIR_DIR.glob("*.wav"))
    if existing:
        _remove_readme_once_populated(RIR_DIR)
        print(f"  Already have {len(existing)} RIR file(s), skipping.")
        return

    auto_src = find_in_known_roots("augmentation_data/room_impulse_responses")
    if auto_src and list(auto_src.glob("*.wav")) and junction(RIR_DIR, auto_src):
        return

    src = resolve_existing_folder(
        "  If you already have a folder of room-impulse-response WAVs (any set\n"
        "  works, not just this specific one), enter its path\n"
        "  (or leave blank to download the MIT IR set automatically): ",
        existing_path,
    )
    if src:
        wavs = list(src.glob("*.wav"))
        if wavs:
            for w in wavs:
                shutil.copy2(w, RIR_DIR / w.name)
            print(f"  Copied {len(wavs)} RIR file(s) from {src}.")
            _remove_readme_once_populated(RIR_DIR)
            return
        print(f"  No .wav files found in {src} -- will download instead.")

    try:
        files = list_rir_files()
    except Exception as e:
        print(f"  Could not list files ({e}) -- skipping. You can grab these manually, see README.md.")
        return
    for i, (name, _size) in enumerate(files, 1):
        url = f"{RIR_BASE}/16khz/{name}"
        print(f"  [{i}/{len(files)}]", end=" ")
        download_with_progress(url, RIR_DIR / name, label=name)
    _remove_readme_once_populated(RIR_DIR)


# ---------------------------------------------------------------------------
# ACAV100M feature cache (the big one)
# ---------------------------------------------------------------------------

def have_aria2c() -> bool:
    return shutil.which("aria2c") is not None


def download_one_feature_file(name: str, approx_size: int, token: str | None, use_aria2: bool) -> None:
    url = f"{FEATURES_BASE}/{name}"
    if use_aria2:
        print(f"  Downloading {name} ({human(approx_size)}) via aria2c (16 connections)...")
        cmd = ["aria2c", "-x16", "-s16", "-k10M", "-d", str(FEATURES_DIR), "-o", name, url]
        if token:
            cmd.insert(1, f"--header=Authorization: Bearer {token}")
        subprocess.run(cmd, check=True)
    else:
        download_with_progress(url, FEATURES_DIR / name, token=token, label=name)


def download_features(token: str | None, yes: bool, acav_path: str | None, validation_path: str | None, copy_mode: str) -> None:
    print("\n=== ACAV100M negative-audio feature cache (the big one) ===")

    # Already in place locally? Skip everything below, same as the voice
    # pack and RIRs -- no token prompt needed for files that are already here.
    still_needed = [
        (name, size) for name, size in FEATURE_FILES
        if not ((FEATURES_DIR / name).exists() and (FEATURES_DIR / name).stat().st_size == size)
    ]
    if not still_needed:
        for name, size in FEATURE_FILES:
            print(f"  [skip] {name} already in place ({human(size)})")
        _remove_readme_once_populated(FEATURES_DIR)
        return

    auto_src = find_in_known_roots("features_cache")
    if auto_src:
        sizes_ok = all((auto_src / name).exists() and (auto_src / name).stat().st_size == size for name, size in FEATURE_FILES)
        if sizes_ok and junction(FEATURES_DIR, auto_src):
            return

    use_aria2 = have_aria2c()
    if not use_aria2:
        print(
            "  NOTE: aria2c not found on PATH. A single-connection download of a ~16GB\n"
            "  file can throttle down over time even with a token. Installing aria2\n"
            "  (winget install aria2.aria2) and re-running is recommended if this stalls."
        )

    print(
        "\n  The ACAV100M feature cache is ~16GB. An optional free Hugging Face token\n"
        "  (huggingface.co/settings/tokens, 'read' scope) makes it noticeably faster --\n"
        "  not required, used only for this run, never saved anywhere."
    )
    if token is None and not yes:
        token = ask("  Paste your HF token (or leave blank to skip): ") or None
    if not yes and ask(f"  About to handle ~{human(sum(s for _, s in still_needed))} -- proceed (Y/N)? ").lower() != "y":
        print("  Skipped the feature cache -- training will not work until this is in place.")
        return

    overrides = {FEATURE_FILES[0][0]: acav_path, FEATURE_FILES[1][0]: validation_path}

    for name, approx_size in FEATURE_FILES:
        dest = FEATURES_DIR / name
        if dest.exists() and dest.stat().st_size == approx_size:
            print(f"  [skip] {name} already in place ({human(approx_size)})")
            continue

        src = resolve_existing_file(
            f"  Already have '{name}' ({human(approx_size)}) downloaded somewhere?\n"
            f"  Enter its full path (or leave blank to download it now): ",
            approx_size, overrides.get(name), assume_yes=(overrides.get(name) is not None),
        )
        if src:
            action = copy_mode if overrides.get(name) is not None else (ask("  Copy or Move it into place? (C/M) [C]: ").lower() or "c")
            dest.parent.mkdir(parents=True, exist_ok=True)
            if action in ("m", "move"):
                shutil.move(str(src), str(dest))
            else:
                print(f"  Copying {human(approx_size)}, this will take a while...")
                shutil.copy2(src, dest)
            print(f"  In place: {dest}")
            continue

        download_one_feature_file(name, approx_size, token, use_aria2)

    _remove_readme_once_populated(FEATURES_DIR)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--yes", action="store_true", help="Skip the two proceed-confirmation prompts")
    parser.add_argument("--hf-token", default=None, help="Hugging Face token (optional, speeds up the big file)")
    parser.add_argument("--voice-pack-path", default=None, help="Folder with an existing voice pack, instead of downloading")
    parser.add_argument("--rir-path", default=None, help="Folder with existing RIR WAVs, instead of downloading")
    parser.add_argument("--acav-path", default=None, help="Existing ACAV100M feature file, instead of downloading")
    parser.add_argument("--validation-path", default=None, help="Existing validation feature file, instead of downloading")
    parser.add_argument("--copy-mode", choices=["copy", "move"], default="copy", help="How to bring in --acav-path/--validation-path files (default: copy)")
    parser.add_argument("--skip-features", action="store_true", help="Skip the ACAV100M feature cache entirely (training won't work until it's added later)")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    print("This will set up the following one-time training assets:\n")
    print(f"  Multi-speaker voice pack     ~{human(sum(s for _, s in VOICES))}  (5 files)")
    print(f"  Room impulse responses       ~{human(RIR_APPROX_TOTAL)}   (270 files)")
    print(f"  ACAV100M feature cache       ~{human(FEATURE_FILES[0][1])}  (1 file)")
    print(f"  Validation set features      ~{human(FEATURE_FILES[1][1])}  (1 file)")
    total = sum(s for _, s in VOICES) + RIR_APPROX_TOTAL + sum(s for _, s in FEATURE_FILES)
    print(f"\n  TOTAL IF DOWNLOADING EVERYTHING: ~{human(total)}")
    print(
        "\n  No Hugging Face account or API key is needed for any of this -- all\n"
        "  three sources are public. Known install locations (see\n"
        "  KNOWN_INSTALL_ROOTS at the top of this script) are checked\n"
        "  automatically and linked in for free; anywhere else, you'll be\n"
        "  asked for a path instead of downloading it again.\n"
    )

    if not args.yes and ask("Proceed (Y/N)? ").lower() != "y":
        print("Cancelled.")
        sys.exit(0)

    download_voice_pack(args.voice_pack_path)
    download_rirs(args.rir_path)

    if args.skip_features:
        print("\n--skip-features given -- not fetching the ACAV100M feature cache.")
        print("\n=== Done ===")
        return

    download_features(args.hf_token, args.yes, args.acav_path, args.validation_path, args.copy_mode)

    print("\n=== Done ===")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nCancelled.")
        sys.exit(1)
