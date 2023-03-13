import numpy as np
from torch.utils import data
from sklearn.model_selection import train_test_split

# Third-party packages
from dataset import CXRDataset


def get_dataloaders(batch_size, train_transform, val_transform, train_df, val_df, test_df, normalize=True):

    train_dataset = CXRDataset(
        df=train_df,
        transforms=train_transform,
        normalize=normalize,
    )

    train_dataset2, _ = train_test_split(train_dataset, test_size= 3875/5216, shuffle=False)
    train_dataset = data.ConcatDataset([train_dataset, train_dataset2])

    val_dataset = CXRDataset(
        df=val_df,
        transforms=val_transform,
        normalize=normalize,
    )

    test_dataset = CXRDataset(
        df=test_df,
        transforms=val_transform,
        normalize=normalize,
    )
    train_dataloader = data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,
    )

    val_dataloader = data.DataLoader(
        dataset=val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )

    test_dataloader = data.DataLoader(
        dataset=test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )

    return train_dataloader, val_dataloader, test_dataloader
