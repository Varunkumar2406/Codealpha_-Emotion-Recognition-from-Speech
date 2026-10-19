"""Dataset indexing for RAVDESS, TESS and EMO-DB.

Each loader walks a dataset folder and returns a DataFrame with columns:
    path, emotion, speaker, dataset
Emotions are mapped to one shared label set so the datasets can be combined.
No heavy dependencies here (only os / pandas) so it is easy to unit-test.
"""
import os
import pandas as pd

# Shared label set (7 classes)
EMOTIONS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
LABEL2ID = {e: i for i, e in enumerate(EMOTIONS)}

# RAVDESS: 03-01-06-01-02-01-12.wav -> modality-channel-emotion-intensity-statement-rep-actor
_RAVDESS = {"01": "neutral", "02": "neutral",  # 02 = calm, merged into neutral
            "03": "happy", "04": "sad", "05": "angry",
            "06": "fear", "07": "disgust", "08": "surprise"}

# TESS: OAF_back_angry.wav -> speaker_word_emotion
_TESS = {"angry": "angry", "disgust": "disgust", "fear": "fear", "happy": "happy",
         "neutral": "neutral", "sad": "sad", "ps": "surprise"}  # ps = pleasant surprise

# EMO-DB: 03a01Fa.wav -> speaker(2) text(3) emotion(1) version(1)
_EMODB = {"W": "angry", "E": "disgust", "A": "fear", "F": "happy",
          "N": "neutral", "T": "sad"}  # L (boredom) is dropped


def _walk_wavs(root):
    for dp, _, files in os.walk(root):
        for f in files:
            if f.lower().endswith(".wav"):
                yield os.path.join(dp, f), f


def parse_ravdess_name(fname):
    parts = os.path.splitext(fname)[0].split("-")
    if len(parts) != 7 or parts[1] != "01":  # keep speech only (01), drop song (02)
        return None
    emo = _RAVDESS.get(parts[2])
    return None if emo is None else (emo, f"rav_{parts[6]}")


def parse_tess_name(fname):
    parts = os.path.splitext(fname)[0].split("_")
    if len(parts) != 3:
        return None
    emo = _TESS.get(parts[2].lower())
    return None if emo is None else (emo, f"tess_{parts[0].upper()}")


def parse_emodb_name(fname):
    stem = os.path.splitext(fname)[0]
    if len(stem) < 6:
        return None
    emo = _EMODB.get(stem[5].upper())
    return None if emo is None else (emo, f"emodb_{stem[:2]}")


_PARSERS = {"ravdess": parse_ravdess_name, "tess": parse_tess_name, "emodb": parse_emodb_name}


def index_dataset(name, root):
    parser = _PARSERS[name]
    rows = []
    for path, fname in _walk_wavs(root):
        res = parser(fname)
        if res:
            rows.append({"path": path, "emotion": res[0], "speaker": res[1], "dataset": name})
    return pd.DataFrame(rows, columns=["path", "emotion", "speaker", "dataset"])


def build_index(ravdess=None, tess=None, emodb=None):
    frames = []
    for name, root in (("ravdess", ravdess), ("tess", tess), ("emodb", emodb)):
        if root:
            df = index_dataset(name, root)
            print(f"[data] {name}: {len(df)} files")
            frames.append(df)
    if not frames:
        raise ValueError("Provide at least one of --ravdess / --tess / --emodb")
    df = pd.concat(frames, ignore_index=True)
    if df.empty:
        raise RuntimeError("No audio files found - check the dataset paths.")
    df["label"] = df["emotion"].map(LABEL2ID)
    return df
