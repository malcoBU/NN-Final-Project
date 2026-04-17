"""
features.py
-----------
Extract fixed-size Mel-spectrogram tensors (and optionally MFCCs) from
preprocessed waveforms.

Output shape: (1, N_MELS, TIME_FRAMES)  — one channel, suitable for CNN input.
"""

import numpy as np
import librosa

# ── Hyperparameters ───────────────────────────────────────────────────────────
TARGET_SR    = 16_000   # must match preprocess.py
N_MELS       = 128      # number of Mel filter banks (frequency bins)
N_MFCC       = 40       # number of MFCC coefficients to keep
TIME_FRAMES  = 128      # fixed number of time columns after padding/cropping
HOP_LENGTH   = 512      # samples between consecutive STFT frames (~32 ms step)
N_FFT        = 1024     # FFT window size (~64 ms window at 16 kHz)
F_MIN        = 50.0     # lowest frequency included (Hz) — cuts sub-voice noise
F_MAX        = 8_000.0  # highest frequency included (Hz) — Nyquist for 16 kHz


# ── Core extraction ───────────────────────────────────────────────────────────

def extract_melspectrogram(
    y: np.ndarray,
    sr: int = TARGET_SR,
    n_mels: int = N_MELS,
    time_frames: int = TIME_FRAMES,
    hop_length: int = HOP_LENGTH,
    n_fft: int = N_FFT,
    f_min: float = F_MIN,
    f_max: float = F_MAX,
) -> np.ndarray:
    """
    Convert a waveform into a fixed-size log-Mel spectrogram tensor.

    Pipeline
    --------
    waveform  →  STFT  →  Mel filterbank  →  log (dB)  →  pad/crop  →  tensor

    Parameters
    ----------
    y : np.ndarray, shape (N,)
        Preprocessed float32 waveform.
    sr : int
        Sample rate of `y`.
    n_mels : int
        Number of Mel frequency bands (height of the output image).
    time_frames : int
        Fixed number of time steps (width of the output image).
        Audios shorter than this are zero-padded on the right;
        audios longer are cropped from the left so the ending
        (most informative part) is preserved.
    hop_length : int
        Number of samples between consecutive frames. Smaller = more
        temporal resolution but larger tensors.
    n_fft : int
        Length of the FFT window. Should be >= hop_length.
    f_min / f_max : float
        Frequency range to include in the Mel filterbank.

    Returns
    -------
    mel_tensor : np.ndarray, shape (1, n_mels, time_frames), float32
        Log-Mel spectrogram with a channel dimension prepended.
        Values are in dB relative to the maximum energy of the clip.
    """
    # 1 · Mel-spectrogram (power = amplitude²)
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        fmin=f_min,
        fmax=f_max,
    )
    # mel.shape == (n_mels, T) where T depends on audio length

    # 2 · Convert power to dB  (ref=np.max → peak = 0 dB, rest negative)
    mel_db = librosa.power_to_db(mel, ref=np.max)

    # 3 · Fix time axis to `time_frames` columns
    mel_db = _fix_length(mel_db, time_frames)

    # 4 · Add channel dimension: (n_mels, T) → (1, n_mels, T)
    return mel_db[np.newaxis, :, :].astype(np.float32)


def extract_mfcc(
    y: np.ndarray,
    sr: int = TARGET_SR,
    n_mfcc: int = N_MFCC,
    time_frames: int = TIME_FRAMES,
    hop_length: int = HOP_LENGTH,
    n_fft: int = N_FFT,
) -> np.ndarray:
    """
    Compute MFCC coefficients and their deltas as a fixed-size tensor.

    Concatenates [MFCC, Δ-MFCC, ΔΔ-MFCC] along the feature axis,
    giving a (3*n_mfcc, time_frames) representation that captures
    both static and dynamic spectral properties.

    Returns
    -------
    mfcc_tensor : np.ndarray, shape (1, 3*n_mfcc, time_frames), float32
    """
    mfcc = librosa.feature.mfcc(
        y=y, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length, n_fft=n_fft
    )
    delta1 = librosa.feature.delta(mfcc)   # first derivative  (velocity)
    delta2 = librosa.feature.delta(mfcc, order=2)  # second derivative (acceleration)

    # Stack along feature axis → (3*n_mfcc, T)
    combined = np.vstack([mfcc, delta1, delta2])
    combined = _fix_length(combined, time_frames)

    return combined[np.newaxis, :, :].astype(np.float32)


def extract_features(
    y: np.ndarray,
    sr: int = TARGET_SR,
    use_mfcc: bool = False,
) -> np.ndarray:
    """
    Main entry point for the DataLoader.

    By default returns the Mel-spectrogram only (best single representation
    for CNN-based letter classification). Set use_mfcc=True to return the
    MFCC tensor instead.

    Returns
    -------
    tensor : np.ndarray, shape (1, H, W), float32
    """
    if use_mfcc:
        return extract_mfcc(y, sr=sr)
    return extract_melspectrogram(y, sr=sr)


# ── Internal helpers ──────────────────────────────────────────────────────────

def _fix_length(matrix: np.ndarray, target_cols: int) -> np.ndarray:
    """
    Pad or crop a 2-D matrix along axis=1 to exactly `target_cols` columns.

    Padding: zeros appended to the right (silence).
    Cropping: columns removed from the LEFT so the ending of the audio
              (where the letter articulation usually completes) is kept.

    Parameters
    ----------
    matrix : np.ndarray, shape (rows, cols)
    target_cols : int

    Returns
    -------
    np.ndarray, shape (rows, target_cols)
    """
    cols = matrix.shape[1]
    if cols < target_cols:
        # Pad right with zeros
        pad = target_cols - cols
        matrix = np.pad(matrix, ((0, 0), (0, pad)), mode="constant")
    elif cols > target_cols:
        # Crop from the left
        matrix = matrix[:, cols - target_cols:]
    return matrix


def normalize_spectrogram(spec: np.ndarray) -> np.ndarray:
    """
    Per-instance standardisation: zero mean, unit variance.

    Applied optionally in the DataLoader after extract_features().
    Helps when dB values differ widely across speakers/microphones.

    Parameters
    ----------
    spec : np.ndarray, shape (1, H, W)

    Returns
    -------
    np.ndarray, shape (1, H, W), float32
    """
    mean = spec.mean()
    std  = spec.std() + 1e-8
    return ((spec - mean) / std).astype(np.float32)