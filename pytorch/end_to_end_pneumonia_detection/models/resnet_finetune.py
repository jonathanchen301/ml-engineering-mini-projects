import torch.nn as nn
import torchvision.models as models

def freeze_backbone(model):
    for param in model.parameters():
        param.requires_grad = False
    for param in model.fc.parameters():
        param.requires_grad = True

def unfreeze_all(model):
    for param in model.parameters():
        param.requires_grad = True

def create_resnet_finetune(num_classes=2, weights="IMAGENET1K_V1", feature_extraction=False):
    model = models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    if feature_extraction:
        freeze_backbone(model)
    else:
        unfreeze_all(model)

    return model