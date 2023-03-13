import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import numpy as np

# Third-party packages

# Normalize with ImageNet mean and std
STD = np.array([0.229, 0.224, 0.225])
MEAN = np.array([0.485, 0.456, 0.406])

def train_transforms(img_size, rotate_degree):
    return A.Compose([
        #A.augmentations.crops.RandomResizedCrop(height=img_size, width=img_size),
        A.Resize(img_size, img_size),
        A.Rotate(rotate_degree),
        A.HorizontalFlip(),
        A.Normalize(mean=MEAN, std=STD),
        ToTensorV2(),
    ], p=1)

def val_transforms(img_size):
    return A.Compose([
    A.Resize(img_size, img_size),
    A.Normalize(mean=MEAN, std=STD),
    ToTensorV2(),
])

def get_transform(train, img_size, rotate_degree):
    return train_transforms(img_size, rotate_degree) if train else val_transforms(img_size)