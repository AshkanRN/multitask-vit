from datasets import load_dataset
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from .fairface_dataset import FairFaceDataset


def get_dataset(train_transform=None, age_transform=None, val_transform=None,
                 val_size=0.5, seed=42, stratify_col="age"):
    """
        Original HuggingFace FairFace split:
            train      -> 86,744 images
            validation -> 10,954 images
        
        We keep the original training split untouched for model training.
        The original validation split is divided into two equal parts:
            - validation set -> used for model selection, early stopping, and tuning
            - test set       -> used only once for final evaluation
        
        Resulting split:
            train      : 86,744 images
            validation : ~5,477 images
            test       : ~5,477 images             
    """    
    import time

    t0 = time.perf_counter()
    dataset = load_dataset("HuggingFaceM4/FairFace", "0.25")
    print(f"load_dataset: {time.perf_counter() - t0:.2f}s")

    t0 = time.perf_counter()
    train_set = dataset["train"]
    hf_val_set = dataset["validation"]
    print(f"extract splits: {time.perf_counter() - t0:.2f}s")

    t0 = time.perf_counter()
    indices = list(range(len(hf_val_set)))
    stratify_labels = hf_val_set["age"]
    print(f"prepare indices: {time.perf_counter() - t0:.2f}s")

    t0 = time.perf_counter()
    val_idx, test_idx = train_test_split(
        indices,
        test_size=0.5,
        random_state=42,
        stratify=stratify_labels
    )
    print(f"train_test_split: {time.perf_counter() - t0:.2f}s")

    t0 = time.perf_counter()
    val_split = hf_val_set.select(val_idx)
    test_split = hf_val_set.select(test_idx)
    print(f"select(): {time.perf_counter() - t0:.2f}s")
    
    train_dataset = FairFaceDataset(
        train_set,
        transforms=train_transform,
        age_transforms=age_transform
    )

    val_dataset = FairFaceDataset(
        val_split,
        transforms=val_transform
    )

    test_dataset = FairFaceDataset(
        test_split,
        transforms=val_transform
    )

    return train_dataset, val_dataset, test_dataset



def get_dataloaders(train_dataset, val_dataset, test_dataset, batch_size=20, num_workers=0):
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

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader
