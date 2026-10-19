"""Audio loading, augmentation and MFCC feature extraction."""
import numpy as np
import librosa

SR = 22050
DURATION = 3.0
N_MFCC = 40
HOP = 512
N_FFT = 1024


def load_audio(path, sr=SR, duration=DURATION):
    y, _ = librosa.load(path, sr=sr)
    y, _ = librosa.effects.trim(y, top_db=25)           # remove leading/trailing silence
    return fix_length(y, int(sr * duration))


def fix_length(y, n):
    if len(y) >= n:
        start = (len(y) - n) // 2                          # centre crop
        return y[start:start + n]
    pad = n - len(y)
    return np.pad(y, (pad // 2, pad - pad // 2))


# ---------- augmentation (training data only) ----------
def add_noise(y, snr_db=None, rng=None):
    rng = rng or np.random.default_rng()
    snr_db = snr_db if snr_db is not None else rng.uniform(15, 30)
    p_sig = np.mean(y ** 2) + 1e-10
    p_noise = p_sig / (10 ** (snr_db / 10))
    return (y + rng.normal(0, np.sqrt(p_noise), len(y))).astype(np.float32)


def pitch_shift(y, sr=SR, steps=None, rng=None):
    rng = rng or np.random.default_rng()
    steps = steps if steps is not None else rng.uniform(-2, 2)
    return librosa.effects.pitch_shift(y, sr=sr, n_steps=steps)


def time_stretch(y, rate=None, rng=None):
    rng = rng or np.random.default_rng()
    rate = rate if rate is not None else rng.uniform(0.9, 1.1)
    return fix_length(librosa.effects.time_stretch(y, rate=rate), len(y))


def augment_variants(y, sr=SR, rng=None):
    """Return a list of augmented copies of waveform y."""
    return [add_noise(y, rng=rng), pitch_shift(y, sr, rng=rng), time_stretch(y, rng=rng)]


# ---------- features ----------
def extract_mfcc(y, sr=SR, n_mfcc=N_MFCC):
    """Return array of shape (3, n_mfcc, T): MFCC, delta, delta-delta."""
    m = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_fft=N_FFT, hop_length=HOP)
    d1 = librosa.feature.delta(m)
    d2 = librosa.feature.delta(m, order=2)
    return np.stack([m, d1, d2]).astype(np.float32)
