"""CNN, (Bi)LSTM and CNN-LSTM classifiers. Input: (B, 3, n_mfcc, T)."""
import torch
import torch.nn as nn


def conv_block(cin, cout, pool=(2, 2), p=0.2):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.MaxPool2d(pool), nn.Dropout2d(p))


class CNN(nn.Module):
    def __init__(self, n_classes, in_ch=3):
        super().__init__()
        self.features = nn.Sequential(conv_block(in_ch, 32), conv_block(32, 64),
                                      conv_block(64, 128), conv_block(128, 256))
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                  nn.Dropout(0.4), nn.Linear(256, n_classes))

    def forward(self, x):
        return self.head(self.features(x))


class LSTM(nn.Module):
    def __init__(self, n_classes, n_mfcc=40, in_ch=3, hidden=128, layers=2):
        super().__init__()
        self.lstm = nn.LSTM(n_mfcc * in_ch, hidden, layers, batch_first=True,
                            bidirectional=True, dropout=0.3)
        self.head = nn.Sequential(nn.Dropout(0.4), nn.Linear(hidden * 4, n_classes))

    def forward(self, x):
        b, c, f, t = x.shape
        x = x.permute(0, 3, 1, 2).reshape(b, t, c * f)       # (B, T, C*F)
        out, _ = self.lstm(x)
        pooled = torch.cat([out.mean(1), out.max(1).values], dim=1)  # mean+max over time
        return self.head(pooled)


class CNNLSTM(nn.Module):
    """CNN extracts local spectral patterns, BiLSTM models their temporal evolution."""
    def __init__(self, n_classes, n_mfcc=40, in_ch=3, hidden=128):
        super().__init__()
        self.cnn = nn.Sequential(conv_block(in_ch, 32, (2, 1)), conv_block(32, 64, (2, 1)),
                                 conv_block(64, 128, (2, 2)))
        freq_out = n_mfcc // 2 // 2 // 2
        self.lstm = nn.LSTM(128 * freq_out, hidden, batch_first=True, bidirectional=True)
        self.head = nn.Sequential(nn.Dropout(0.4), nn.Linear(hidden * 4, n_classes))

    def forward(self, x):
        x = self.cnn(x)                                      # (B, C, F', T')
        b, c, f, t = x.shape
        x = x.permute(0, 3, 1, 2).reshape(b, t, c * f)
        out, _ = self.lstm(x)
        return self.head(torch.cat([out.mean(1), out.max(1).values], dim=1))


def build_model(name, n_classes, n_mfcc=40):
    name = name.lower()
    if name == "cnn":
        return CNN(n_classes)
    if name == "lstm":
        return LSTM(n_classes, n_mfcc)
    if name in ("cnnlstm", "cnn_lstm"):
        return CNNLSTM(n_classes, n_mfcc)
    raise ValueError(f"Unknown model '{name}' (cnn | lstm | cnnlstm)")
