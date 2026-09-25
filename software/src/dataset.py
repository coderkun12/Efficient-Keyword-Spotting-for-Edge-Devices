"""
Speech Commands loading + preprocessing.

Follows the standard Google Speech Commands "10-keyword" benchmark setup used
in the original paper and in Hello Edge (Zhang et al., 2017):

    Core keywords (10): yes, no, up, down, left, right, on, off, stop, go
    + "unknown"  : a downsampled pool of the other ~25 spoken words
    + "silence"  : random 1-second crops of background noise

    => 12 output classes total.

Uses torchaudio's official train/val/test split (based on Google's
validation_list.txt / testing_list.txt), so results are comparable to
published numbers.
"""

import random
from pathlib import Path

import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader

# ---------------------------------------------------------------------------
# Label configuration
# ---------------------------------------------------------------------------

CORE_KEYWORDS = ["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"]
UNKNOWN_LABEL = "_unknown_"
SILENCE_LABEL = "_silence_"
BACKGROUND_NOISE_FOLDER = "_background_noise_"

# Final label list / index mapping (order matters -- index 0..11)
LABELS = CORE_KEYWORDS + [UNKNOWN_LABEL, SILENCE_LABEL]
LABEL_TO_INDEX = {label: i for i, label in enumerate(LABELS)}
NUM_CLASSES = len(LABELS)

# ---------------------------------------------------------------------------
# Audio / feature extraction config
# ---------------------------------------------------------------------------

SAMPLE_RATE = 16000
CLIP_LENGTH_SAMPLES = SAMPLE_RATE  # 1 second clips
N_MELS = 40
N_FFT = 400          # 25ms window at 16kHz
HOP_LENGTH = 160     # 10ms hop at 16kHz
# -> spectrogram shape per clip: (N_MELS, ~101 time frames)

# How many "unknown" and "silence" examples to include, per split, relative
# to the average per-class count of the 10 core keywords. 1.0 means the
# unknown/silence classes are roughly as large as one average core class,
# which keeps the dataset from being dominated by "unknown" while still
# teaching the model to reject non-keyword input.
UNKNOWN_RATIO = 1.0
SILENCE_RATIO = 1.0

# ---------------------------------------------------------------------------
# Training-time augmentation config (matches Google's reference preprocessing:
# random time shift + background noise injection). Only ever applied to the
# "training" split -- validation/testing must stay clean and deterministic.
# ---------------------------------------------------------------------------

TIME_SHIFT_MS = 100  # max shift, matches the ~100ms used in reference implementations
TIME_SHIFT_SAMPLES = int(SAMPLE_RATE * TIME_SHIFT_MS / 1000)
BACKGROUND_NOISE_PROB = 0.8       # chance a given training sample gets noise mixed in
BACKGROUND_NOISE_MAX_VOLUME = 0.1  # max relative volume of mixed-in noise

SEED = 42


def _time_shift(waveform: torch.Tensor, shift: int) -> torch.Tensor:
    """Shift waveform in time by `shift` samples, zero-filling the gap created."""
    if shift == 0:
        return waveform
    length = waveform.shape[-1]
    shifted = torch.zeros_like(waveform)
    if shift > 0:
        shifted[..., shift:] = waveform[..., : length - shift]
    else:
        s = -shift
        shifted[..., : length - s] = waveform[..., s:]
    return shifted


def _mix_background_noise(waveform: torch.Tensor, noise_waveforms, rng) -> torch.Tensor:
    """Randomly mix in a snippet of background noise at a random low volume."""
    if not noise_waveforms or rng.random() > BACKGROUND_NOISE_PROB:
        return waveform
    length = waveform.shape[-1]
    noise = noise_waveforms[rng.randrange(len(noise_waveforms))]
    total_len = noise.shape[-1]
    if total_len <= length:
        noise_crop = _pad_or_trim(noise, length)
    else:
        start = rng.randint(0, total_len - length)
        noise_crop = noise[..., start : start + length]
    volume = rng.uniform(0.0, BACKGROUND_NOISE_MAX_VOLUME)
    mixed = waveform + volume * noise_crop
    return torch.clamp(mixed, -1.0, 1.0)


def _pad_or_trim(waveform: torch.Tensor, length: int = CLIP_LENGTH_SAMPLES) -> torch.Tensor:
    """Ensure waveform is exactly `length` samples (pad with zeros or trim)."""
    num_samples = waveform.shape[-1]
    if num_samples == length:
        return waveform
    if num_samples > length:
        return waveform[..., :length]
    pad_amount = length - num_samples
    return torch.nn.functional.pad(waveform, (0, pad_amount))


def build_mel_transform():
    """Mel-spectrogram + log (dB) transform, used identically at train/val/test."""
    mel_spec = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
    )
    to_db = torchaudio.transforms.AmplitudeToDB()
    return torch.nn.Sequential(mel_spec, to_db)


class SpeechCommandsKWS(Dataset):
    """
    Wraps torchaudio.datasets.SPEECHCOMMANDS into a 12-class keyword-spotting
    dataset: 10 core keywords + unknown + silence.

    subset: "training" | "validation" | "testing"
    """

    def __init__(self, root: str = "data", subset: str = "training", download: bool = False):
        assert subset in ("training", "validation", "testing")
        self.subset = subset
        self.mel_transform = build_mel_transform()
        rng = random.Random(SEED + {"training": 0, "validation": 1, "testing": 2}[subset])

        # torchaudio handles the official split for us (based on Google's
        # validation_list.txt / testing_list.txt -- everything else is "training").
        base_dataset = torchaudio.datasets.SPEECHCOMMANDS(
            root=root, download=download, subset=subset
        )
        self._dataset_root = Path(base_dataset._path)

        # Walk the underlying file list once to bucket examples by label,
        # keeping direct (path) references rather than loading audio yet.
        core_examples = []      # list of (path, label_index)
        unknown_examples = []   # list of path (label assigned later)

        for i in range(len(base_dataset)):
            # base_dataset._walker entries are already full paths (built
            # internally by torchaudio via glob / list-file joins) -- NOT
            # paths relative to self._path. Use them as-is; do not re-join
            # with self._dataset_root or the path gets duplicated.
            full_path = Path(base_dataset._walker[i])
            label = full_path.parent.name

            if label == BACKGROUND_NOISE_FOLDER:
                continue  # handled separately for the silence class
            if label in CORE_KEYWORDS:
                core_examples.append((full_path, LABEL_TO_INDEX[label]))
            else:
                unknown_examples.append(full_path)

        # Downsample "unknown" so it doesn't dominate the dataset.
        avg_core_count = max(1, len(core_examples) // len(CORE_KEYWORDS))
        num_unknown = int(avg_core_count * UNKNOWN_RATIO)
        rng.shuffle(unknown_examples)
        unknown_examples = unknown_examples[:num_unknown]

        # Build "silence" examples as random 1-second crops of background noise.
        num_silence = int(avg_core_count * SILENCE_RATIO)
        noise_dir = self._dataset_root / BACKGROUND_NOISE_FOLDER
        noise_files = sorted(noise_dir.glob("*.wav")) if noise_dir.exists() else []

        # Preload all background noise waveforms once -- there are only a
        # handful of files (a few MB total), and both silence-crop generation
        # and training-time noise augmentation reuse this same cache instead
        # of re-reading from disk on every access.
        self._noise_waveforms = []
        for noise_path in noise_files:
            wav, sr = torchaudio.load(str(noise_path))
            assert sr == SAMPLE_RATE, f"Unexpected sample rate in {noise_path}: {sr}"
            self._noise_waveforms.append(wav)

        silence_crops = self._make_silence_crops(self._noise_waveforms, num_silence, rng)

        # Final flat example list: (path_or_None, label_index, silence_waveform_or_None)
        self.examples = (
            [(p, lbl, None) for p, lbl in core_examples]
            + [(p, LABEL_TO_INDEX[UNKNOWN_LABEL], None) for p in unknown_examples]
            + [(None, LABEL_TO_INDEX[SILENCE_LABEL], wav) for wav in silence_crops]
        )
        rng.shuffle(self.examples)

        # Augmentation only ever applies to the training split, and uses an
        # unseeded RNG on purpose -- we want genuinely different shifts/noise
        # on each epoch, not the same fixed augmentation every time.
        self.augment = subset == "training"
        self._aug_rng = random.Random()

    @staticmethod
    def _make_silence_crops(noise_waveforms, num_needed, rng):
        """Randomly crop `num_needed` 1-second clips from preloaded background noise waveforms."""
        crops = []
        if not noise_waveforms or num_needed == 0:
            return crops
        for _ in range(num_needed):
            waveform = noise_waveforms[rng.randrange(len(noise_waveforms))]
            total_len = waveform.shape[-1]
            if total_len <= CLIP_LENGTH_SAMPLES:
                crop = _pad_or_trim(waveform)
            else:
                start = rng.randint(0, total_len - CLIP_LENGTH_SAMPLES)
                crop = waveform[..., start:start + CLIP_LENGTH_SAMPLES]
            crops.append(crop)
        return crops

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        path, label_idx, precomputed_waveform = self.examples[idx]
        if precomputed_waveform is not None:
            waveform = precomputed_waveform
        else:
            waveform, sr = torchaudio.load(str(path))
            assert sr == SAMPLE_RATE, f"Unexpected sample rate in {path}: {sr}"
            waveform = _pad_or_trim(waveform)

        if self.augment:
            shift = self._aug_rng.randint(-TIME_SHIFT_SAMPLES, TIME_SHIFT_SAMPLES)
            waveform = _time_shift(waveform, shift)
            waveform = _mix_background_noise(waveform, self._noise_waveforms, self._aug_rng)

        # (1, n_mels, time) -> mel_transform expects (channel, time)
        mel_spec = self.mel_transform(waveform)  # shape: (1, N_MELS, T)
        return mel_spec, label_idx


def get_dataloaders(root: str = "data", batch_size: int = 128, num_workers: int = 4, download: bool = False):
    """Convenience helper: returns (train_loader, val_loader, test_loader)."""
    train_ds = SpeechCommandsKWS(root=root, subset="training", download=download)
    val_ds = SpeechCommandsKWS(root=root, subset="validation", download=download)
    test_ds = SpeechCommandsKWS(root=root, subset="testing", download=download)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Quick sanity check when run directly: python src/dataset.py
    train_loader, val_loader, test_loader = get_dataloaders(num_workers=0)
    print(f"Classes ({NUM_CLASSES}): {LABELS}")
    print(f"Train batches: {len(train_loader)} | Val: {len(val_loader)} | Test: {len(test_loader)}")
    x, y = next(iter(train_loader))
    print(f"Batch shape: {x.shape} (B, C, n_mels, time) | Labels shape: {y.shape}")