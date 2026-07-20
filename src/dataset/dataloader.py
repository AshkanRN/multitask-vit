from datasets import load_dataset
from torch.utils.data import DataLoader

from .fairface_dataset import FairFaceDataset


def get_dataset(train_transform=None, age_transform=None, val_transform=None):
    dataset = load_dataset("HuggingFaceM4/FairFace", "0.25")

    train_dataset = FairFaceDataset(
        dataset["train"],
        transforms=train_transform,
        age_transforms=age_transform
    )

    val_dataset = FairFaceDataset(
        dataset["validation"],
        transforms=val_transform
    )

    return train_dataset, val_dataset



def get_dataloaders(train_dataset, val_dataset, batch_size=32, num_workers=0):
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False
    )


    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )


    return train_loader, val_loader
