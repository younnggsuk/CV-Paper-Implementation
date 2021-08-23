import os
import cv2

import albumentations as A
from albumentations.pytorch import ToTensorV2

from torch.utils.data import Dataset, DataLoader


__all__ = ["PascalVOC2012", "PascalVOC2012Inference", "fetch_dataloader"]


class PascalVOC2012(Dataset):
    
    def __init__(self, image_paths, label_paths, transform=None):
        self.image_paths = image_paths
        self.label_paths = label_paths
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label_path = self.label_paths[idx]
        
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
        
        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]
            
        return image, mask


class PascalVOC2012Inference(Dataset):
    
    def __init__(self, image_paths, label_paths, transform=None):
        self.image_paths = image_paths
        self.label_paths = label_paths
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label_path = self.label_paths[idx]
        
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
        origin_size = image.shape[:-1]

        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]
            
        return image, mask, origin_size


def fetch_dataloader(types, data_dir, batch_size, num_workers):
    dataloaders = {}

    image_dir = os.path.join(data_dir, "VOCdevkit", "VOC2012", "JPEGImages") # *.jpg
    label_dir = os.path.join(data_dir, "VOCdevkit", "VOC2012", "SegmentationClassAug") # *.png

    train_files = []
    with open(os.path.join(data_dir, "VOCdevkit", "VOC2012", "ImageSets", "Segmentation", "trainaug.txt"), "rt") as f:
        for line in f.readlines():
            train_files.append(line.replace('\n', ''))

    valid_files = []
    with open(os.path.join(data_dir, "VOCdevkit", "VOC2012", "ImageSets", "Segmentation", "val.txt"), "rt") as f:
        for line in f.readlines():
            valid_files.append(line.replace('\n', ''))

    train_image_paths = [os.path.join(image_dir, file_name + ".jpg") for file_name in train_files]
    train_label_paths = [os.path.join(label_dir, file_name + ".png") for file_name in train_files]

    valid_image_paths = [os.path.join(image_dir, file_name + ".jpg") for file_name in valid_files]
    valid_label_paths = [os.path.join(label_dir, file_name + ".png") for file_name in valid_files]

    transform = A.Compose([
        A.LongestMaxSize(max_size=448),
        A.PadIfNeeded(min_height=448, min_width=448, border_mode=cv2.BORDER_CONSTANT),
        A.Normalize(),
        ToTensorV2()
    ])

    for split in ["train", "val", "eval"]:
        if split in types:
            if split == "train":
                dl = DataLoader(PascalVOC2012(train_image_paths,
                                              train_label_paths,
                                              transform=transform),
                                batch_size, shuffle=True, num_workers=num_workers)
                                
            elif split == "val":
                dl = DataLoader(PascalVOC2012(valid_image_paths,
                                              valid_label_paths,
                                              transform=transform),
                                              batch_size, shuffle=False, num_workers=num_workers)

            elif split == "eval":
                dl = DataLoader(PascalVOC2012Inference(valid_image_paths,
                                                       valid_label_paths,
                                                       transform=transform),
                                batch_size, shuffle=False, num_workers=num_workers)

            dataloaders[split] = dl

    return dataloaders