from torchvision.transforms import v2
import torch

def base_transform(processor):
    return [
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(
            mean=processor.image_mean,
            std=processor.image_std,
        ),
    ]

def get_train_transform(processor):
    return v2.Compose([
        v2.RandomHorizontalFlip(p=0.5),

        v2.RandomRotation(5, fill=(128, 128, 128)),

        v2.ColorJitter(
            brightness=0.15,
            contrast=0.15,
            saturation=0.10,
            hue=0.02,
        ),

        *base_transform(processor)
    ])

def get_age_transform(processor):
    return v2.Compose([
        v2.RandomHorizontalFlip(p=0.7),

        v2.RandomRotation(5, fill=(128, 128, 128)),

        v2.ColorJitter(
            brightness=0.20,
            contrast=0.20,
            saturation=0.15,
            hue=0.03,
        ),

        *base_transform(processor)
    ])

def get_val_transform(processor):
    return v2.Compose([
        *base_transform(processor),
    ])