from torch.utils.data import Dataset

class FairFaceDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]

        image = sample['image']

        gender = sample['gender']
        age = sample['age']
        race = sample['race']

        if self.transform:
            image = self.transform(image)

        return image,{
            'gender': gender,
            'age': age,
            'race': race
        }