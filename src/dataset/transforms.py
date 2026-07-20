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

        v2.RandomRotation(8, fill=(128, 128, 128)),

        v2.ColorJitter(
            brightness=0.1,
            contrast=0.1,
            saturation=0.1,
            hue=0.02,
        ),

        v2.RandomApply(
            [
                v2.GaussianBlur(
                    kernel_size=3,
                    sigma=(0.1,0.5)
                )
            ],
            p=0.1
        ),

        *base_transform(processor)
    ])

def get_age_transform(processor):
    return v2.Compose([
        v2.RandomHorizontalFlip(p=0.5),

        v2.RandomRotation(8, fill=(128, 128, 128)),

        # v2.RandomResizedCrop(
        #     (224, 224),
        #     scale=(0.9, 1.0),
        #     ratio=(0.95, 1.05)
        # ),

        v2.ColorJitter(
            brightness=0.15,
            contrast=0.15,
            saturation=0.10,
            hue=0.03,
        ),

        *base_transform(processor)
    ])

def get_val_transform(processor):
    return v2.Compose([
        *base_transform(processor),
    ])