from torchvision.transforms import v2

def get_train_transform():
    return v2.Compose([
        v2.RandomHorizontalFlip(p=0.5),

        v2.RandomRotation(5, fill=(128, 128, 128)),

        v2.ColorJitter(
            brightness=0.15,
            contrast=0.15,
            saturation=0.10,
            hue=0.02,
        ),
    ])

def get_age_transform():
    return v2.Compose([
        v2.RandomHorizontalFlip(p=0.7),

        v2.RandomRotation(5, fill=(128, 128, 128)),

        v2.ColorJitter(
            brightness=0.20,
            contrast=0.20,
            saturation=0.15,
            hue=0.03,
        ),
    ])