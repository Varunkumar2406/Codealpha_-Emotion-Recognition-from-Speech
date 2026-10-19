# Emotion Recognition from Speech — Project Report

**Name:** ____________  **Date:** ____________

## 1. Objective
Recognise human emotions (angry, disgust, fear, happy, neutral, sad) from speech audio using MFCC features and deep learning.

## 2. Dataset
EMO-DB (Berlin Database of Emotional Speech): 535 German utterances, 10 actors, 7 emotions in the original set
(anger, boredom, disgust, fear, happiness, sadness, neutral). Boredom was removed, leaving 6 classes.
Add RAVDESS / TESS here if used.

## 3. Method
- **Preprocessing:** 22.05 kHz mono, silence trimmed, fixed to 3 s.
- **Features:** 40 MFCCs + delta + delta-delta (3 x 40 x 130).
- **Augmentation (train only):** Gaussian noise, pitch shift (±2 semitones), time stretch (0.9–1.1x).
- **Models:** CNN; 2-layer BiLSTM; CNN-LSTM hybrid.
- **Training:** AdamW, class-weighted cross-entropy with label smoothing 0.1, early stopping on validation macro-F1.
- **Split:** speaker-independent (train / validation / test contain different speakers).

## 4. Results
| Model | Test accuracy | Macro-F1 |
|---|---|---|
| CNN | | |
| BiLSTM | | |
| CNN-LSTM | | |

Insert `confusion_matrix.png` and `training_curves.png` for each model.

## 5. Analysis
- Best model and why: 
- Most confused emotions (see confusion matrix): 
- Effect of augmentation (with vs without): 
- Effect of seed / small test set (EMO-DB test set has only ~1–2 speakers): 

## 6. Limitations and future work
Acted (not spontaneous) speech; single language; small dataset; speaker-dependent variance.
Future work: add RAVDESS/TESS for cross-corpus testing, SpecAugment, attention, wav2vec2/HuBERT features, real-time microphone demo.

## 7. How to run
See README.md (`SER_Colab.ipynb` runs the entire pipeline on Google Colab).
