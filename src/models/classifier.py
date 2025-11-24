"""
Model definitions for MedVision.
Includes configurable backbones (ResNet, EfficientNet) and loss helpers.
This system is for research and educational purposes only, not for clinical use.
"""

from typing import Optional
import torch
import torch.nn as nn
from torchvision import models


class Classifier(nn.Module):
    """
    Generic classifier wrapper around torchvision backbones.
    Supports resnet34, resnet50, resnet101 and efficientnet_b0.
    """

    def __init__(
        self,
        architecture: str = 'resnet50',
        num_classes: int = 2,
        pretrained: bool = True,
        dropout: float = 0.3
    ):
        super().__init__()
        arch = architecture.lower()
        self.arch = arch

        if arch.startswith('resnet'):
            if arch == 'resnet34':
                backbone = models.resnet34(pretrained=pretrained)
                in_features = backbone.fc.in_features
            elif arch == 'resnet50':
                backbone = models.resnet50(pretrained=pretrained)
                in_features = backbone.fc.in_features
            elif arch == 'resnet101':
                backbone = models.resnet101(pretrained=pretrained)
                in_features = backbone.fc.in_features
            else:
                raise ValueError(f"Unsupported ResNet architecture: {architecture}")

            # Remove last fc
            modules = list(backbone.children())[:-2]  # keep conv layers
            self.features = nn.Sequential(*modules)
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(in_features, num_classes)
            )

            # For Grad-CAM target layer mapping
            self.target_layer = backbone.layer4

        elif arch.startswith('efficientnet'):
            try:
                backbone = getattr(models, arch)(pretrained=pretrained)
            except Exception:
                raise ValueError(f"Unsupported EfficientNet architecture: {architecture}")
            in_features = backbone.classifier[1].in_features
            modules = list(backbone.features)
            self.features = nn.Sequential(*modules)
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(in_features, num_classes)
            )
            self.target_layer = self.features[-1]

        else:
            raise ValueError(f"Unsupported architecture: {architecture}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


# Loss helpers

def get_loss_function(loss_type: str = 'weighted_ce', class_weights: Optional[torch.Tensor] = None,
                      focal_alpha: float = 0.25, focal_gamma: float = 2.0):
    """
    Return loss function configured for class imbalance.
    Supports: ce, weighted_ce, focal
    """
    loss_type = loss_type.lower()
    if loss_type == 'ce':
        return nn.CrossEntropyLoss()
    elif loss_type == 'weighted_ce':
        if class_weights is None:
            return nn.CrossEntropyLoss()
        else:
            return nn.CrossEntropyLoss(weight=class_weights)
    elif loss_type == 'focal':
        # Lightweight focal loss implementation
        class FocalLoss(nn.Module):
            def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
                super().__init__()
                self.alpha = alpha
                self.gamma = gamma
                self.reduction = reduction
                self.ce = nn.CrossEntropyLoss(reduction='none')

            def forward(self, inputs, targets):
                logp = -self.ce(inputs, targets)
                p = torch.exp(logp)
                loss = -((1 - p) ** self.gamma) * logp
                # Apply alpha weighting for binary only
                if inputs.shape[1] == 2:
                    alpha = torch.tensor([1 - self.alpha, self.alpha], device=inputs.device)
                    at = alpha[targets]
                    loss = at * loss
                if self.reduction == 'mean':
                    return loss.mean()
                elif self.reduction == 'sum':
                    return loss.sum()
                else:
                    return loss
        return FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
    else:
        raise ValueError(f"Unsupported loss type: {loss_type}")
