from torch.utils.data import Dataset

class FairFaceDataset(Dataset):
    def __init__(
        self,
        dataset,
        transforms=None,
        age_transforms=None
    ):
        self.dataset = dataset
        self.transforms = transforms
        self.age_transforms = age_transforms



    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]

        image = sample['image']

        gender = sample['gender']
        age = sample['age']
        race = sample['race']

        if age in [0,7,8] and self.age_transforms is not None:
            image = self.age_transforms(image)

        elif self.transforms is not None:
            image = self.transforms(image)

        return image,{
            'gender': gender,
            'age': age,
            'race': race
        }