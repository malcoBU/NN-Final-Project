"""
augment.py
----------
Data augmentation techniques applied to raw waveforms (numpy arrays).
All functions receive a float32 numpy array at TARGET_SR and return
a new array of the same dtype.

Used exclusively during training — never applied to validation/test sets.
"""

import random
import numpy as np
import librosa

TARGET_SR = 16_000

# ── Individual transforms ─────────────────────────────────────────────────────

def add_gaussian_noise(y: np.ndarray, sigma: float = 0.005) -> np.ndarray:
    """
    Add zero-mean Gaussian noise to the waveform.

    Simulates: cheap microphones, background room noise.

    Parameters
    ----------
    y : np.ndarray
        Input waveform.
    sigma : float
        Standard deviation of the noise. 0.005 is subtle; 0.02 is clearly
        audible. Keep below 0.03 or the letter becomes hard to recognise.
    """
    noise = np.random.normal(loc=0.0, scale=sigma, size=len(y))
    return np.clip(y + noise, -1.0, 1.0).astype(np.float32)


def pitch_shift(
    y: np.ndarray,
    sr: int = TARGET_SR,
    n_steps: float | None = None,
) -> np.ndarray:
    """
    Shift the pitch without changing duration.

    Simulates: different voice types (deep/high-pitched speakers).

    Parameters
    ----------
    n_steps : float or None
        Semitones to shift. Positive = higher pitch, negative = lower.
        If None, samples uniformly from [-2, 2].
        Keep within ±3 semitones — beyond that, letters can sound like
        a different phoneme.
    """
    if n_steps is None:
        n_steps = random.uniform(-2.0, 2.0)
    return librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps).astype(np.float32)


def time_stretch(y: np.ndarray, rate: float | None = None) -> np.ndarray:
    """
    Speed up or slow down the waveform without changing pitch.

    Simulates: fast/slow speakers, different recording conditions.

    Parameters
    ----------
    rate : float or None
        Stretch factor. rate > 1 = faster (shorter), rate < 1 = slower (longer).
        If None, samples uniformly from [0.85, 1.15].
        Stay within [0.75, 1.25] — outside this range letters become
        unrecognisable.
    """
    if rate is None:
        rate = random.uniform(0.85, 1.15)
    return librosa.effects.time_stretch(y, rate=rate).astype(np.float32)


def change_volume(y: np.ndarray, gain: float | None = None) -> np.ndarray:
    """
    Multiply the waveform by a scalar gain factor.

    Simulates: microphone distance, speaker loudness variation.

    Parameters
    ----------
    gain : float or None
        Multiplier applied to every sample.
        If None, samples uniformly from [0.70, 1.30].
        np.clip keeps the result within [-1, 1] to avoid digital clipping.
    """
    if gain is None:
        gain = random.uniform(0.70, 1.30)
    return np.clip(y * gain, -1.0, 1.0).astype(np.float32)


def add_background_noise(
    y: np.ndarray,
    noise_array: np.ndarray,
    snr_db: float | None = None,
) -> np.ndarray:
    """
    Mix the signal with a background noise waveform at a given SNR.

    Simulates: realistic acoustic environments (street, office, cafe).

    Parameters
    ----------
    noise_array : np.ndarray
        Pre-loaded noise waveform (same sample rate as y).
        Tip: use freely available noise datasets such as ESC-50 or MUSAN.
    snr_db : float or None
        Signal-to-noise ratio in dB. If None, samples from [10, 30].
        10 dB = noisy but intelligible; 30 dB = barely perceptible noise.
    """
    if snr_db is None:
        snr_db = random.uniform(10.0, 30.0)

    # Loop or trim noise to match signal length
    if len(noise_array) < len(y):
        repeats = int(np.ceil(len(y) / len(noise_array)))
        noise_array = np.tile(noise_array, repeats)
    noise_array = noise_array[: len(y)]

    # Compute RMS of signal and noise, then scale noise to desired SNR
    rms_signal = np.sqrt(np.mean(y ** 2)) + 1e-8
    rms_noise  = np.sqrt(np.mean(noise_array ** 2)) + 1e-8
    desired_rms_noise = rms_signal / (10 ** (snr_db / 20))
    noise_scaled = noise_array * (desired_rms_noise / rms_noise)

    return np.clip(y + noise_scaled, -1.0, 1.0).astype(np.float32)


# ── Composed pipeline ─────────────────────────────────────────────────────────

# Maps name → (function, probability of being applied)
AUGMENTATION_REGISTRY: dict[str, tuple] = {
    "gaussian_noise": (add_gaussian_noise, 0.5),
    "pitch_shift":    (pitch_shift,         0.4),
    "time_stretch":   (time_stretch,         0.4),
    "volume_change":  (change_volume,        0.5),
}


def augment(
    y: np.ndarray,
    sr: int = TARGET_SR,
    p_apply: float = 0.8,
    enabled: list[str] | None = None,
) -> np.ndarray:
    """
    Apply a random subset of augmentation transforms to a waveform.

    Each registered transform is applied independently with its own
    probability. The whole pipeline is skipped with probability (1 - p_apply)
    to preserve some clean samples in every batch.

    Parameters
    ----------
    y : np.ndarray
        Input waveform (float32, normalised to [-1, 1]).
    sr : int
        Sample rate.
    p_apply : float
        Probability of running the augmentation pipeline at all.
        Set to 1.0 during offline pre-augmentation; keep < 1.0 for
        on-the-fly augmentation inside the DataLoader.
    enabled : list[str] or None
        Subset of AUGMENTATION_REGISTRY keys to use.
        If None, all registered transforms are candidates.

    Returns
    -------
    y_aug : np.ndarray
        Augmented (or original if not applied) waveform.
    """
    if random.random() > p_apply:
        return y  # return original with probability (1 - p_apply)

    registry = AUGMENTATION_REGISTRY
    if enabled is not None:
        registry = {k: v for k, v in registry.items() if k in enabled}

    y_aug = y.copy()
    for name, (fn, prob) in registry.items():
        if random.random() < prob:
            try:
                if name == "pitch_shift":
                    y_aug = fn(y_aug, sr=sr)
                else:
                    y_aug = fn(y_aug)
            except Exception:
                # If a transform fails (e.g. audio too short for stretch),
                # silently skip it and continue with the other transforms
                pass

    return y_aug


# ── CLI helper (offline pre-augmentation) ────────────────────────────────────
if __name__ == "__main__":
    """
    Quick smoke-test: load a .wav, apply augmentation, save output.

    Usage:
        python src/augment.py path/to/audio.wav
    """
    import sys
    from preprocess import load_and_preprocess
    import soundfile as sf

    if len(sys.argv) < 2:
        print("Usage: python src/augment.py <path_to_wav>")
        sys.exit(1)

    wav_path = sys.argv[1]
    y = load_and_preprocess(wav_path)
    y_aug = augment(y, p_apply=1.0)

    out_path = wav_path.replace(".wav", "_augmented.wav")
    sf.write(out_path, y_aug, TARGET_SR)
    print(f"Saved augmented audio to: {out_path}")