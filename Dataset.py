import os

from torchvision.datasets import ImageFolder
from torchvision import transforms


TRANSFORMS = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])


class CustomDataset(ImageFolder):
    def __init__(self, path_to_dataset_folder):
        super().__init__(os.path.join(path_to_dataset_folder), transform=TRANSFORMS)
