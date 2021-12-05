import os

from torchvision.datasets import ImageFolder
from torchvision import transforms


TRAIN_TRANSFORMS = transforms.Compose([
    transforms.AutoAugment(transforms.AutoAugmentPolicy.IMAGENET),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])

TEST_TRANSFORMS = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])


class CustomDataset(ImageFolder):
    def __init__(self, path_to_dataset_folder, test=False):
        if test:
            super().__init__(os.path.join(path_to_dataset_folder), transform=TEST_TRANSFORMS)
        else:
            super().__init__(os.path.join(path_to_dataset_folder), transform=TRAIN_TRANSFORMS)
