# openWakeWord Windows Training Pipeline

Windows-native training pipeline for openWakeWord wake words, with real
recordings optionally mixed into the positive training set alongside
synthetic TTS clips. Train as many wake words as you like from this one
install -- each gets its own `config/<model_name>.yaml` and (if you use real
recordings) its own `real_samples/<model_name>/`. Currently set up with
"Hey Holly" as the example/active model.

This file covers the scripts and layout in this folder. For the full generic
pipeline docs (install/download/train internals, Windows patches, Home
Assistant deploy steps), see [`PIPELINE_README.md`](PIPELINE_README.md).

## Disclaimer for AI use

Claude Code was used in part for the creation of these scripts. I have
tested them and run them on my own personal machine, however, as with any
code, results may vary, and it is the user's decision to run this without
prior review. I will not be held accountable for any damage or issues
arising from its use.

---

## Installation / Dependencies

1. `install.bat` -- sets up `.venv` (one-time, ~10-15 min)
2. Get the shared assets (voice pack, RIRs, feature cache, ~16.5GB):
   `download_assets.bat` -- automatically finds and links to an existing
   copy in another install (edit `KNOWN_INSTALL_ROOTS` in
   `scripts/download_assets.py` to add install locations), or downloads
   fresh if none is found.

See [`PIPELINE_README.md`](PIPELINE_README.md) for the full manual/troubleshooting
version of these two steps (exact pip installs, patches applied, why each one
is needed).

---

## Training

3. Train: double-click **`Train.bat`** -- asks which mode you want and
   hands off to the right script (see table below)

Output: `model_output\<model_name>.tflite` + `.json`. Copy both to the
openWakeWord add-on's model folder and each Assist Satellite's
`external_wake_words` folder, then set it as the active wake word in Home
Assistant.

---

## Scripts

| .bat | Runs | What it's for |
|---|---|---|
| `install.bat` | `install.py` | One-time setup: venv, torch, patched openWakeWord. Installs into whatever folder you run it from (see below) -- **not copied** to target folders. |
| `Train.bat` | -- | **Start here.** Asks single-mode vs bulk-real-samples, then calls one of the two below. |
| `train_wake_word.bat` | `scripts/train_wake_word.py` | Standard training: synthetic clips only. Prompts for wake phrase, model name, sample count, cutoff. Use for a brand new wake word with no real recordings. |
| `train_with_real_samples.bat` | `scripts/train_with_real_samples.py` | Opens a file picker to choose which `config/<model_name>.yaml` to train, mixes `real_samples/<model_name>/train/*.wav` into the synthetic positive set, trains, then auto-tests against `real_samples/<model_name>/eval_holdout/*.wav`. |
| `download_assets.bat` | `scripts/download_assets.py` | Downloads (or copies in, if you point it at an existing copy) the voice pack / RIRs / feature cache. |
| `eval_real_voice.bat` | `scripts/eval_real_voice.py` | Tests an already-trained `.tflite` against one or more real WAV recordings. `eval_real_voice.bat model_output\hey_holly.tflite real_samples\hey_holly\eval_holdout\*.wav` |
| `installer_gui.bat` | `installer_gui.ps1` | All-in-one GUI wrapping install / download / train. |

---

## Real recordings mix-in

Any wake word can optionally mix in real recordings alongside synthetic
clips, to improve real-world
generalisation. For example, "Hey Holly" here uses 24 real recordings:

| Folder | Contents |
|---|---|
| `real_samples/<model_name>/train/` | Real clips, copied into `positive_train` alongside the synthetic clips during training |
| `real_samples/<model_name>/eval_holdout/` | Real clips, **never used in training** -- automatically run against the finished model at the end of `train_with_real_samples.py` |

`scripts/train_with_real_samples.py` is a modified copy of the standard
`train_wake_word.py`: it opens a file picker for you to choose which
`config/<model_name>.yaml` to train, reads `model_name` from it, then runs
`--generate_clips`, copies `real_samples/<model_name>/train/*.wav` into
`training/<model_name>/positive_train/`, and runs `--augment_clips
--train_model` and the rest of the pipeline as normal.
`real_samples/<model_name>/train/` and `.../eval_holdout/` are created
automatically (with a `README.txt`) the first time you pick that config.

`config/hey_holly.yaml` is one example, pre-filled (10,000 synthetic
samples, 50,000 training steps). Set up a new wake word with
`train_wake_word.bat` (option 1 in `Train.bat`), which writes
`config/<model_name>.yaml` for you interactively.

---

## Folder layout

```
stock_multispeaker_voices\   -- shared Piper voice pack (auto-linked from another install, or downloaded)
augmentation_data\           -- shared RIRs (auto-linked from another install, or downloaded)
features_cache\              -- shared negative-audio features (auto-linked from another install, or downloaded)
training\                    -- generated clips, checkpoints (created by training)
model_output\                -- final .tflite + .json (created by training)
config\<model_name>.yaml          -- training config, one per wake word (e.g. hey_holly.yaml)
real_samples\<model_name>\train\        -- real recordings mixed into training (optional, per wake word)
real_samples\<model_name>\eval_holdout\ -- real recordings held back to test the finished model
scripts\                        -- pipeline scripts (see Scripts table above)
Split_Samples\                  -- optional scratch folder for your own split recordings
```

## Installing into a different project folder

This folder is the "repo" -- run `install.bat` by full path from inside any
other folder (e.g. `E:\Live Locally Hosted\Open Wake Word Training`) to set
up a fresh, independent instance there instead. On first run in an empty
folder it copies across everything in the table above
**except** `install.bat`/`install.ps1` themselves (a target folder doesn't
need to be able to re-run setup, only to download assets and train).

Real recordings are **not** copied across -- `real_samples/<model_name>/train/`
and `.../eval_holdout/` are created automatically (with a `README.txt`
explaining what goes there) the first time `train_with_real_samples.py`
picks that wake word's config. Record your own wake phrase, split it into
individual clips (whatever tool you like -- e.g. your DAC's own splitting),
and drop the files in yourself (most into
`train/`, a handful held back into `eval_holdout/`).

---

## Utilities (this folder only, not part of the pipeline itself)

- **`Bulk File rename.ps1`** -- renames a batch of recordings to
  `Hey_HollyN.wav` naming.

---

## Credits / Third-Party Projects

- **[openWakeWord](https://github.com/dscripka/openWakeWord)** (dscripka, Apache-2.0) -- the training pipeline and runtime model format this whole repo builds on.
- **[Piper](https://github.com/rhasspy/piper) / [piper-voices](https://huggingface.co/rhasspy/piper-voices)** (rhasspy, MIT) -- the TTS engine and stock multi-speaker voices used for synthetic sample generation.
- **[onnx2tf](https://github.com/PINTO0309/onnx2tf)** (PINTO0309, MIT) -- ONNX -> TFLite conversion.
- **[ai-edge-litert](https://github.com/google-ai-edge/LiteRT)** (Google, Apache-2.0) -- the TFLite interpreter used for shape verification and real-recording evaluation.
- **[MIT Acoustical Reverberation Scene Statistics Survey](https://mcdermottlab.mit.edu/Reverb/IR_Survey.html)** -- one option for the room-impulse-response WAVs used in reverb augmentation.
- **ACAV100M precomputed negative features** -- distributed via openWakeWord's own training docs/HuggingFace, for the negative-audio feature cache.

See [`PIPELINE_README.md`](PIPELINE_README.md) for the full license/attribution notes and detailed troubleshooting log.
