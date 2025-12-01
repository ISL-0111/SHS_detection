import os
import cv2
import numpy as np
from torch.utils.data import Dataset

class InferenceDataset(Dataset):
    def __init__(self, image_dir, augmentation=None):
        self.image_paths = sorted([
            os.path.join(image_dir, f)
            for f in os.listdir(image_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp'))
        ])
        self.augmentation = augmentation

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = cv2.imread(self.image_paths[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.augmentation:
            augmented = self.augmentation(image=image)
            image = augmented["image"]
        return image
