from torchvision.models import resnet18
import torch.nn as nn


class ClassifierModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = resnet18(pretrained=False)
        self.model.fc = nn.Linear(512, 200)

    def forward(self, x):
        return self.model(x)
