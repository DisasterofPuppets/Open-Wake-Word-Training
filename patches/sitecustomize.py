import scipy.special

# The `acoustics` package (old, unmaintained) imports scipy.special.sph_harm,
# which modern scipy removed in favor of sph_harm_y. We only use
# acoustics.generator.noise() (unrelated to spherical harmonics), but the
# import chain still crashes eagerly on acoustics.directivity unless this
# name exists. Aliasing is safe here since we never actually call it.
if not hasattr(scipy.special, "sph_harm"):
    scipy.special.sph_harm = scipy.special.sph_harm_y

# torchaudio.load() now always routes through torchcodec (the old
# soundfile/sox_io backend system was removed), but torchcodec's native
# libtorchcodec_image.dll fails to load here (version mismatch against our
# torch build). openWakeWord's data.py calls torchaudio.load() in several
# places just to read WAV files, so replace it with a plain soundfile-based
# implementation matching the same (tensor[channels, samples], sample_rate)
# return convention.
import torchaudio
import soundfile as _sf
import torch as _torch


def _load_via_soundfile(uri, frame_offset=0, num_frames=-1, normalize=True,
                         channels_first=True, format=None, buffer_size=4096, backend=None):
    data, sr = _sf.read(str(uri), dtype="float32" if normalize else "int16", always_2d=True)
    if frame_offset or num_frames != -1:
        end = None if num_frames == -1 else frame_offset + num_frames
        data = data[frame_offset:end]
    tensor = _torch.from_numpy(data)
    if channels_first:
        tensor = tensor.T
    return tensor, sr


torchaudio.load = _load_via_soundfile
