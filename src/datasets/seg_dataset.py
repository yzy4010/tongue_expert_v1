import os
import cv2
import torch
from torch.utils.data import Dataset
import pandas as pd

class TongueSegDataset(Dataset):
    def __init__(self, image_dir, mask_dir, split_file, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform

        with open(split_file) as f:
            self.ids = [line.strip() for line in f]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        sid = self.ids[idx]

        img = cv2.imread(os.path.join(self.image_dir, sid + ".jpg"))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(os.path.join(self.mask_dir, sid + ".png"), 0)
        mask = (mask > 0).astype("float32")

        if self.transform:
            augmented = self.transform(image=img, mask=mask)
            img = augmented["image"]
            mask = augmented["mask"]

        img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        mask = torch.from_numpy(mask).unsqueeze(0)

        return img, mask
