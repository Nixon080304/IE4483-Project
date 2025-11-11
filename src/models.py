import torch.nn as nn
from torchvision import models


def create_resnet18(
    num_classes: int = 2,
    pretrained: bool = True,
    freeze_backbone: bool = True
) -> nn.Module:
    """
    Creates a ResNet-18 model for binary classification (dog vs cat).
    """
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    # Replace final fully-connected layer
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    return model
