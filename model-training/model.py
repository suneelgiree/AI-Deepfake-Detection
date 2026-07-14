"""
Visual branch: EfficientNet backbone fine-tuned for real/fake classification.

This is the CNN half of your report's "Visual Branch (EfficientNet + LSTM)".
The LSTM temporal component requires video-sequence input (frame sequences,
not independent images) — add it once you're training on FaceForensics++ or
DFDC video clips in Iteration 3. This single-frame classifier is still a
legitimate, functioning first version of the visual branch.
"""
import torch
import torch.nn as nn
import timm


class EfficientNetClassifier(nn.Module):
    def __init__(self, num_classes=2, pretrained=True):
        super().__init__()
        # efficientnet_b0 pretrained on ImageNet — fine-tuned here on deepfake data
        self.backbone = timm.create_model(
            "efficientnet_b0", pretrained=pretrained, num_classes=0  # remove classifier head
        )
        feat_dim = self.backbone.num_features

        self.classifier = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        features = self.backbone(x)
        out = self.classifier(features)
        return out


if __name__ == "__main__":
    # sanity check: run a dummy batch through the model
    model = EfficientNetClassifier()
    dummy = torch.randn(2, 3, 224, 224)
    out = model(dummy)
    print("Output shape:", out.shape)  # expect [2, 2]
