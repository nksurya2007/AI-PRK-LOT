"""
cnn_classifier.py
Defines the deep learning models for binary parking slot classification:
1. ParkingSlotCNN: Fast, lightweight custom CNN optimized for CPU/edge inference.
2. MobileNetV2SlotClassifier: Transfer-learning architecture for benchmark comparison.
"""

import torch
import torch.nn as nn
import torchvision.models as models

class ParkingSlotCNN(nn.Module):
    """
    Lightweight, high-accuracy Convolutional Neural Network for Parking Slot Occupancy.
    Input: (B, 3, 64, 64)
    Output: (B, 2) -> [logit_empty, logit_occupied]
    """
    def __init__(self, num_classes=2, dropout_rate=0.3):
        super(ParkingSlotCNN, self).__init__()
        
        # Feature Extractor Block 1
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2) # -> 32 x 32
        )
        
        # Feature Extractor Block 2
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2) # -> 16 x 16
        )
        
        # Feature Extractor Block 3
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2) # -> 8 x 8
        )

        # Feature Extractor Block 4
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)) # -> 4 x 4
        )
        
        # Classifier Head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        logits = self.classifier(x)
        return logits


class MobileNetV2SlotClassifier(nn.Module):
    """
    Transfer learning classifier using pretrained MobileNetV2 backbone.
    """
    def __init__(self, num_classes=2, pretrained=True):
        super(MobileNetV2SlotClassifier, self).__init__()
        weights = models.MobileNet_V2_Weights.DEFAULT if pretrained else None
        base = models.mobilenet_v2(weights=weights)
        self.features = base.features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        in_features = base.last_channel
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        logits = self.classifier(x)
        return logits


def get_model(model_name="custom_cnn", num_classes=2, pretrained=True):
    if model_name == "mobilenet_v2":
        return MobileNetV2SlotClassifier(num_classes=num_classes, pretrained=pretrained)
    else:
        return ParkingSlotCNN(num_classes=num_classes)


if __name__ == "__main__":
    model = ParkingSlotCNN()
    dummy = torch.randn(4, 3, 64, 64)
    out = model(dummy)
    print("ParkingSlotCNN test passed! Output shape:", out.shape)
