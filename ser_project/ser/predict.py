"""Predict the emotion of one audio file.

    python -m ser.predict --ckpt outputs/cnnlstm/model.pt --audio sample.wav
"""
import argparse
import torch
from ser.features import load_audio, extract_mfcc
from ser.models import build_model


def predict(ckpt_path, audio_path):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = build_model(ck["model"], len(ck["emotions"]), ck["n_mfcc"])
    model.load_state_dict(ck["state"]); model.eval()
    x = (extract_mfcc(load_audio(audio_path)) - ck["mean"]) / ck["std"]
    with torch.no_grad():
        logits = model(torch.tensor(x[None]))
        if "present" in ck:                                   # ignore classes never seen in training
            m = torch.full_like(logits, -1e9); m[:, ck["present"]] = 0; logits = logits + m
        probs = torch.softmax(logits, 1)[0].numpy()
    return {e: float(p) for e, p in sorted(zip(ck["emotions"], probs), key=lambda z: -z[1])}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True); ap.add_argument("--audio", required=True)
    a = ap.parse_args()
    res = predict(a.ckpt, a.audio)
    print(f"Predicted emotion: {next(iter(res))}")
    for e, p in res.items():
        print(f"  {e:9s} {p:6.1%}  {'#' * int(p * 40)}")
