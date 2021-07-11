import os
import cv2

import albumentations as A
from albumentations.pytorch import ToTensorV2

from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split


__all__ = ['CatDogDataset', 'fetch_dataloader']


class CatDogDataset(Dataset):
    
    def __init__(self, file_paths, labels, transform=None):

        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        label = self.labels[idx]
        file_path = self.file_paths[idx]
        
        image = cv2.imread(file_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed["image"]
            
        return image, label


def fetch_dataloader(types, data_dir, batch_size, num_workers):
    dataloaders = {}

    train_dir = os.path.join(data_dir, "train")
    train_files = sorted(os.listdir(train_dir))
    train_labels = []
    for file in train_files:
        if "cat" in file:
            train_labels.append(0)
        else:
            train_labels.append(1)

    train_file_paths = [os.path.join(train_dir, path) for path in train_files]

    train_file_paths, val_file_paths, train_labels, val_labels = train_test_split(
        train_file_paths, train_labels, stratify=train_labels, random_state=42
    )

    train_transform = A.Compose([
        A.SmallestMaxSize(max_size=256),
        A.HorizontalFlip(p=0.5),
        A.RandomCrop(224, 224),
        A.Normalize(),
        ToTensorV2()
    ])

    eval_transform = A.Compose([
        A.SmallestMaxSize(max_size=256),
        A.CenterCrop(224, 224),
        A.Normalize(),
        ToTensorV2()
    ])

    for split in ['train', 'val', 'test']:
        if split in types:
            if split == 'train':
                dl = DataLoader(CatDogDataset(train_file_paths,
                                              train_labels,
                                              train_transform),
                                batch_size, shuffle=True, num_workers=num_workers)
            elif split == "val":
                dl = DataLoader(CatDogDataset(val_file_paths,
                                              val_labels,
                                              eval_transform),
                                batch_size, shuffle=False, num_workers=num_workers)

            dataloaders[split] = dl

    return dataloaders