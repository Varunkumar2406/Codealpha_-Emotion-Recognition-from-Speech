"""Train + evaluate a speech emotion recognition model.

Example:
    python -m ser.train --ravdess data/RAVDESS --tess data/TESS --model cnnlstm --epochs 60 --augment
"""
import argparse, json, os, random
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from ser.data import build_index, EMOTIONS
from ser.features import load_audio, extract_mfcc, augment_variants, N_MFCC
from ser.models import build_model


def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def split_data(df, seed, test_size=0.15, val_size=0.15):
    """Speaker-independent split when enough speakers exist, else stratified random."""
    if df["speaker"].nunique() >= 8:
        gss = GroupShuffleSplit(1, test_size=test_size, random_state=seed)
        tv, te = next(gss.split(df, groups=df["speaker"]))
        tv_df, te_df = df.iloc[tv], df.iloc[te]
        gss = GroupShuffleSplit(1, test_size=val_size / (1 - test_size), random_state=seed)
        tr, va = next(gss.split(tv_df, groups=tv_df["speaker"]))
        print("[split] speaker-independent")
        return tv_df.iloc[tr], tv_df.iloc[va], te_df
    print("[split] too few speakers -> stratified random split (results will be optimistic)")
    tv_df, te_df = train_test_split(df, test_size=test_size, stratify=df["label"], random_state=seed)
    tr_df, va_df = train_test_split(tv_df, test_size=val_size / (1 - test_size),
                                    stratify=tv_df["label"], random_state=seed)
    return tr_df, va_df, te_df


def featurize(df, augment, cache_dir, tag):
    cache = os.path.join(cache_dir, f"{tag}_aug{int(augment)}.npz")
    if os.path.exists(cache):
        z = np.load(cache); print(f"[cache] loaded {cache}"); return z["X"], z["y"]
    rng = np.random.default_rng(0)
    X, y = [], []
    for path, lab in tqdm(zip(df["path"], df["label"]), total=len(df), desc=f"features:{tag}"):
        wav = load_audio(path)
        X.append(extract_mfcc(wav)); y.append(lab)
        if augment:
            for aug in augment_variants(wav, rng=rng):
                X.append(extract_mfcc(aug)); y.append(lab)
    X, y = np.stack(X), np.array(y, dtype=np.int64)
    os.makedirs(cache_dir, exist_ok=True)
    np.savez_compressed(cache, X=X, y=y)
    return X, y


def evaluate(model, loader, device, crit=None, present=None):
    """present: list of class ids that exist in the training data (others can never be predicted)."""
    model.eval(); preds, trues, loss = [], [], 0.0
    mask = None
    if present is not None:
        mask = torch.zeros(len(EMOTIONS), dtype=torch.bool, device=device); mask[present] = True
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            out = model(xb)
            if crit is not None:
                loss += crit(out, yb).item() * len(yb)
            logits = out if mask is None else out.masked_fill(~mask, -1e9)
            preds.append(logits.argmax(1).cpu().numpy()); trues.append(yb.cpu().numpy())
    p, t = np.concatenate(preds), np.concatenate(trues)
    return (loss / len(t), accuracy_score(t, p),
            f1_score(t, p, average="macro", labels=present if present is not None else None, zero_division=0), p, t)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ravdess"); ap.add_argument("--tess"); ap.add_argument("--emodb")
    ap.add_argument("--model", default="cnnlstm", choices=["cnn", "lstm", "cnnlstm"])
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=12)
    ap.add_argument("--augment", action="store_true", help="noise / pitch / stretch on train set")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--cache", default="cache")
    a = ap.parse_args()

    seed_all(a.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = os.path.join(a.out, a.model); os.makedirs(run_dir, exist_ok=True)

    df = build_index(a.ravdess, a.tess, a.emodb)
    print(df["emotion"].value_counts().to_string())
    tr_df, va_df, te_df = split_data(df, a.seed)
    key = "_".join(sorted(df["dataset"].unique())) + f"_s{a.seed}"

    Xtr, ytr = featurize(tr_df, a.augment, a.cache, f"{key}_train")
    Xva, yva = featurize(va_df, False, a.cache, f"{key}_val")
    Xte, yte = featurize(te_df, False, a.cache, f"{key}_test")
    print(f"train {Xtr.shape}  val {Xva.shape}  test {Xte.shape}")

    # per-channel normalisation using TRAIN statistics only
    mean = Xtr.mean(axis=(0, 2, 3), keepdims=True)[0]
    std = Xtr.std(axis=(0, 2, 3), keepdims=True)[0] + 1e-6
    mk = lambda X, y, sh: DataLoader(TensorDataset(torch.tensor((X - mean) / std), torch.tensor(y)),
                                     batch_size=a.batch_size, shuffle=sh)
    dl_tr, dl_va, dl_te = mk(Xtr, ytr, True), mk(Xva, yva, False), mk(Xte, yte, False)

    model = build_model(a.model, len(EMOTIONS), N_MFCC).to(device)
    print(f"model={a.model} params={sum(p.numel() for p in model.parameters()):,} device={device}")
    counts = np.bincount(ytr, minlength=len(EMOTIONS)).astype(float)
    present = np.flatnonzero(counts > 0).tolist()          # e.g. EMO-DB has no 'surprise'
    print("classes in training data:", [EMOTIONS[i] for i in present])
    wv = np.zeros(len(EMOTIONS)); wv[present] = counts.sum() / (len(present) * counts[present])
    w = torch.tensor(wv, dtype=torch.float32, device=device)  # weight 0 for absent classes
    crit = nn.CrossEntropyLoss(weight=w, label_smoothing=0.1)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=4)

    hist = {k: [] for k in ["train_loss", "val_loss", "val_acc", "val_f1"]}
    best_f1, bad, ckpt = -1, 0, os.path.join(run_dir, "model.pt")
    for ep in range(1, a.epochs + 1):
        model.train(); tl = 0.0
        for xb, yb in dl_tr:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(); loss = crit(model(xb), yb); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
            tl += loss.item() * len(yb)
        tl /= len(ytr)
        vl, vacc, vf1, _, _ = evaluate(model, dl_va, device, crit, present)
        sched.step(vl)
        for k, v in zip(hist, [tl, vl, vacc, vf1]):
            hist[k].append(v)
        print(f"ep {ep:3d} | train_loss {tl:.4f} | val_loss {vl:.4f} | val_acc {vacc:.3f} | val_f1 {vf1:.3f}")
        if vf1 > best_f1:
            best_f1, bad = vf1, 0
            torch.save({"state": model.state_dict(), "model": a.model, "mean": mean, "std": std,
                        "emotions": EMOTIONS, "n_mfcc": N_MFCC, "present": present}, ckpt)
        else:
            bad += 1
            if bad >= a.patience:
                print("early stopping"); break

    # ---- final test evaluation with the best checkpoint ----
    model.load_state_dict(torch.load(ckpt, weights_only=False)["state"])
    _, acc, f1, p, t = evaluate(model, dl_te, device, None, present)
    labels = present
    names = [EMOTIONS[i] for i in labels]
    report = classification_report(t, p, labels=labels, target_names=names, zero_division=0, output_dict=True)
    print("\n" + classification_report(t, p, labels=labels, target_names=names, zero_division=0))
    print(f"TEST accuracy {acc:.4f}  macro-F1 {f1:.4f}")
    json.dump({"accuracy": acc, "macro_f1": f1, "report": report, "args": vars(a)},
              open(os.path.join(run_dir, "metrics.json"), "w"), indent=2)

    cm = confusion_matrix(t, p, labels=labels)
    cmn = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    plt.figure(figsize=(7, 6))
    sns.heatmap(cmn, annot=cm, fmt="d", cmap="Blues", xticklabels=names, yticklabels=names)
    plt.xlabel("Predicted"); plt.ylabel("True"); plt.title(f"{a.model} - test acc {acc:.3f}")
    plt.tight_layout(); plt.savefig(os.path.join(run_dir, "confusion_matrix.png"), dpi=150); plt.close()

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(hist["train_loss"], label="train"); ax[0].plot(hist["val_loss"], label="val")
    ax[0].set_title("Loss"); ax[0].set_xlabel("epoch"); ax[0].legend()
    ax[1].plot(hist["val_acc"], label="val acc"); ax[1].plot(hist["val_f1"], label="val macro-F1")
    ax[1].set_title("Validation"); ax[1].set_xlabel("epoch"); ax[1].legend()
    plt.tight_layout(); plt.savefig(os.path.join(run_dir, "training_curves.png"), dpi=150); plt.close()
    print(f"Saved results to {run_dir}/")


if __name__ == "__main__":
    main()
