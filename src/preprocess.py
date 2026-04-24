"""
preprocess.py
-------------
Audio loading, silence trimming, amplitude normalization and resampling.

Input formats accepted
----------------------
• .wav  — loaded directly with librosa / soundfile.
• .ogg  — converted to a temporary .wav file via pydub + ffmpeg, then
          processed identically to a native .wav.  The temporary file is
          deleted automatically after loading.

All public functions receive a file path and return a clean, fixed-sample-rate
numpy float32 array ready for feature extraction.

Requirements for .ogg conversion
----------------------------------
    pip install pydub
    # ffmpeg must also be installed and available on PATH:
    #   macOS  : brew install ffmpeg
    #   Ubuntu : sudo apt install ffmpeg
    #   Windows: https://ffmpeg.org/download.html
"""

import os
import tempfile
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path

# pydub is only required when .ogg files are present; imported lazily in
# ogg_to_wav() so the rest of the module works without it.
try:
    from pydub import AudioSegment
    _PYDUB_AVAILABLE = True
except ImportError:
    _PYDUB_AVAILABLE = False

# ── Constants ────────────────────────────────────────────────────────────────
TARGET_SR   = 16_000   # sample rate used throughout the whole project
TOP_DB      = 20       # dB threshold for silence trimming
MIN_SAMPLES = 1_600    # minimum audio length after trimming (0.1 s at 16 kHz)


# ── .ogg → .wav conversion ───────────────────────────────────────────────────

def ogg_to_wav(ogg_path: str, wav_path: str | None = None) -> str:
    """
    Convert a .ogg audio file to .wav using pydub + ffmpeg.

    Parameters
    ----------
    ogg_path : str
        Path to the source .ogg file.
    wav_path : str or None
        Destination path for the output .wav file.
        • If provided, the file is written there (permanent).
        • If None, a temporary file is created in the system's temp directory
          and its path is returned.  The caller is responsible for deleting it
          when done (load_and_preprocess handles this automatically).

    Returns
    -------
    str
        Absolute path to the resulting .wav file.

    Raises
    ------
    ImportError
        If pydub is not installed.
    RuntimeError
        If ffmpeg is not found or the conversion fails.
    FileNotFoundError
        If `ogg_path` does not exist.
    """
    if not _PYDUB_AVAILABLE:
        raise ImportError(
            "pydub is required to convert .ogg files.\n"
            "Install it with:  pip install pydub\n"
            "Also make sure ffmpeg is installed on your system."
        )

    ogg_path = str(ogg_path)
    if not os.path.exists(ogg_path):
        raise FileNotFoundError(f".ogg file not found: {ogg_path}")

    # pydub uses ffmpeg under the hood; format="ogg" tells it to use the OGG
    # demuxer explicitly instead of relying on extension-based auto-detection.
    try:
        audio = AudioSegment.from_file(ogg_path, format="ogg")
    except Exception as exc:
        raise RuntimeError(
            f"ffmpeg could not decode '{ogg_path}'.\n"
            f"Make sure ffmpeg is installed and the file is a valid OGG file.\n"
            f"Original error: {exc}"
        ) from exc

    # Determine output path
    if wav_path is None:
        # Create a named temporary file; delete=False so we can pass its path
        # to librosa — the OS keeps the file until we explicitly remove it.
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        wav_path = tmp.name
        tmp.close()

    audio.export(wav_path, format="wav")
    return wav_path


# ── Core function ─────────────────────────────────────────────────────────────
def load_and_preprocess(path: str, sr: int = TARGET_SR) -> np.ndarray:
    """
    Load a .wav or .ogg file and return a clean, normalised mono waveform.

    Steps
    -----
    0. If the file is a .ogg, convert it to a temporary .wav first (via
       pydub + ffmpeg) and delete the temp file after loading.
    1. Load audio and resample to `sr` Hz.
    2. Convert to mono (average channels if stereo).
    3. Trim leading/trailing silence.
    4. Normalise amplitude to [-1, 1].
    5. Pad with zeros if the result is shorter than MIN_SAMPLES.

    Parameters
    ----------
    path : str
        Absolute or relative path to a .wav or .ogg file.
    sr : int
        Target sample rate in Hz (default 16 000).

    Returns
    -------
    y : np.ndarray, shape (N,)
        Preprocessed waveform.

    Raises
    ------
    FileNotFoundError
        If `path` does not exist.
    ValueError
        If the file extension is not supported or the audio cannot be decoded.
    """
    path = str(path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Audio file not found: {path}")

    suffix = Path(path).suffix.lower()
    if suffix not in (".wav", ".ogg"):
        raise ValueError(
            f"Unsupported file extension '{suffix}'. "
            "Only .wav and .ogg files are accepted."
        )

    # ── Step 0 · Convert .ogg → temporary .wav ───────────────────────────────
    tmp_wav_path = None          # will hold the temp path if we create one
    load_path    = path          # path actually passed to librosa

    if suffix == ".ogg":
        tmp_wav_path = ogg_to_wav(path, wav_path=None)  # None → temp file
        load_path    = tmp_wav_path

    # ── Steps 1–5 · Standard preprocessing ───────────────────────────────────
    try:
        y, _ = librosa.load(load_path, sr=sr, mono=True)
    except Exception as exc:
        raise ValueError(
            f"Could not read audio file '{path}': {exc}"
        ) from exc
    finally:
        # Always delete the temp .wav, even if librosa raised an exception
        if tmp_wav_path is not None and os.path.exists(tmp_wav_path):
            os.remove(tmp_wav_path)

    # 2 · Trim silence
    y, _ = librosa.effects.trim(y, top_db=TOP_DB)

    # 3 · Normalise amplitude  (+ eps avoids division by zero on silent files)
    peak = np.max(np.abs(y))
    y = y / (peak + 1e-8)

    # 4 · Ensure minimum length (pad with zeros on the right)
    if len(y) < MIN_SAMPLES:
        y = np.pad(y, (0, MIN_SAMPLES - len(y)), mode="constant")

    return y.astype(np.float32)


# ── Batch helper ──────────────────────────────────────────────────────────────
def preprocess_directory(
    raw_dir: str,
    out_dir: str,
    sr: int = TARGET_SR,
    verbose: bool = True,
    ) -> dict:
    """
    Walk `raw_dir` recursively, preprocess every .ogg (or .wav) file and save
    the resulting numpy array as a .npy file in `out_dir`, preserving the
    relative folder structure.

    Expected raw_dir layout
    -----------------------
    raw_dir/
      english/a/speaker1_a_01.ogg
      english/b/speaker1_b_01.ogg
      spanish/a/speaker1_a_01.ogg
      ...

    Parameters
    ----------
    raw_dir : str
        Root folder containing the original audio files (.ogg or .wav).
    out_dir : str
        Destination folder for preprocessed .npy files.
    sr : int
        Target sample rate.
    verbose : bool
        Print progress to stdout.

    Returns
    -------
    stats : dict
        {"processed": int, "failed": int, "failed_files": list[str]}
    """
    raw_dir = Path(raw_dir)
    out_dir = Path(out_dir)
    stats = {"processed": 0, "failed": 0, "failed_files": []}

    # Collect both .ogg and .wav files so the function works with mixed datasets
    audio_files = sorted(
        list(raw_dir.rglob("*.ogg")) + list(raw_dir.rglob("*.wav"))
    )
    if verbose:
        n_ogg = sum(1 for f in audio_files if f.suffix == ".ogg")
        n_wav = sum(1 for f in audio_files if f.suffix == ".wav")
        print(f"Found {len(audio_files)} audio files in '{raw_dir}'  "
              f"({n_ogg} .ogg  +  {n_wav} .wav)")

    for audio_path in audio_files:
        try:
            y = load_and_preprocess(str(audio_path), sr=sr)

            # Mirror directory structure under out_dir, always save as .npy
            rel = audio_path.relative_to(raw_dir).with_suffix(".npy")
            dest = out_dir / rel
            dest.parent.mkdir(parents=True, exist_ok=True)

            np.save(str(dest), y)
            stats["processed"] += 1

            if verbose:
                print(f"  ✓  {rel}")

        except Exception as exc:
            stats["failed"] += 1
            stats["failed_files"].append(str(audio_path))
            if verbose:
                print(f"  ✗  {audio_path.name}  —  {exc}")

    if verbose:
        print(
            f"\nDone. Processed: {stats['processed']}  |  "
            f"Failed: {stats['failed']}"
        )
    return stats


# ── Utility ───────────────────────────────────────────────────────────────────
def get_duration(path: str, sr: int = TARGET_SR) -> float:
    """Return duration in seconds of a .wav file after resampling."""
    y, _ = librosa.load(path, sr=sr, mono=True)
    return len(y) / sr


def waveform_stats(y: np.ndarray, sr: int = TARGET_SR) -> dict:
    """Return basic stats about a waveform array (useful for EDA)."""
    return {
        "duration_s":  round(len(y) / sr, 3),
        "n_samples":   len(y),
        "sample_rate": sr,
        "min":         float(y.min()),
        "max":         float(y.max()),
        "rms":         float(np.sqrt(np.mean(y ** 2))),
    }


if __name__ == "__main__":
    preprocess_directory(
    raw_dir="./data/raw/english",
    out_dir="./data/processed/english",
    verbose=True)
    preprocess_directory(
    raw_dir="./data/raw/spanish",
    out_dir="./data/processed/spanish",
    verbose=True)