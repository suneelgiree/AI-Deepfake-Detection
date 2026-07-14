"""
Visual branch: EfficientNet (per-frame spatial features) + LSTM (temporal
modeling across the frame sequence). This is the full architecture your
report describes, unlike the earlier single-frame classifier.

Input: [batch, seq_len, C, H, W]
Output: [batch, num_classes]
"""
import torch
import torch.nn as nn
import timm


class CNNLSTMClassifier(nn.Module):
    def __init__(self, num_classes=2, lstm_hidden=256, lstm_layers=1, pretrained=True):
        super().__init__()
        self.cnn = timm.create_model(
            "efficientnet_b0", pretrained=pretrained, num_classes=0
        )
        feat_dim = self.cnn.num_features

        self.lstm = nn.LSTM(
            input_size=feat_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=False,
        )

        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        # x: [batch, seq_len, C, H, W]
        batch_size, seq_len, C, H, W = x.shape

        # Flatten batch and sequence dims to run all frames through CNN at once
        x = x.view(batch_size * seq_len, C, H, W)
        features = self.cnn(x)  # [batch*seq_len, feat_dim]

        # Reshape back into sequences for the LSTM
        features = features.view(batch_size, seq_len, -1)  # [batch, seq_len, feat_dim]

        lstm_out, (h_n, c_n) = self.lstm(features)
        # Use the final hidden state as the sequence summary
        final_hidden = h_n[-1]  # [batch, lstm_hidden]

        out = self.classifier(final_hidden)
        return out


if __name__ == "__main__":
    model = CNNLSTMClassifier(pretrained=False)
    dummy = torch.randn(2, 8, 3, 224, 224)  # batch=2, seq_len=8
    out = model(dummy)
    print("Output shape:", out.shape)  # expect [2, 2]
