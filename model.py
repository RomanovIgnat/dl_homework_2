from torchvision.models import resnext50_32x4d
import torch.nn as nn


class ClassifierModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = resnext50_32x4d(pretrained=False, num_classes=200)

    def forward(self, x):
        return self.model(x)
