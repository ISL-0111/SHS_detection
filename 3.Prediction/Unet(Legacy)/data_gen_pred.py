import os
import cv2
import numpy as np
from torch.utils.data import Dataset as BaseDataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image


class Dataset(BaseDataset):
    CLASSES = ['background', 'PV_normal', 'PV_heater', 'PV_pool']
    
    def __init__(self, images_dir, masks_dir=None, classes=None, augmentation=None):
        self.ids = os.listdir(images_dir)
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        
        if masks_dir is not None:
            self.masks_fps = [
                os.path.join(masks_dir, f"m_{os.path.splitext(image_id)[0][2:]}.png")
                for image_id in self.ids
            ]
        else:
            self.masks_fps = None  # Inference mode (no masks)

        self.background_class = 0

        if classes:
            self.class_values = [self.CLASSES.index(cls) for cls in classes]
        else:
            self.class_values = list(range(len(self.CLASSES)))

        self.class_map = {i: i for i in self.class_values}
        self.augmentation = augmentation


    def __getitem__(self, i):
        try:
            image = cv2.imread(self.images_fps[i])
            if image is None or image.size == 0:
                raise ValueError(f"Corrupted or unreadable image: {self.images_fps[i]}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Handle mask loading if masks exist
            if self.masks_fps:
                mask = np.array(Image.open(self.masks_fps[i]).convert('L'))
                if mask is None or mask.size == 0:
                    raise ValueError(f"Corrupted or unreadable mask: {self.masks_fps[i]}")
                if mask.shape != (320, 320):
                    raise ValueError(f"Invalid mask shape: {mask.shape}, expected (320, 320)")
                if len(self.class_values) < len(self.CLASSES):
                    mask_remap = np.zeros_like(mask)
                    for idx, class_value in enumerate(self.class_values):
                        mask_remap[mask == class_value] = idx
                    mask = mask_remap
            else:
                mask = None  # Inference mode

            if image.shape[:2] != (320, 320):
                image = cv2.resize(image, (320, 320))
                if mask is not None:
                    mask = cv2.resize(mask, (320, 320), interpolation=cv2.INTER_NEAREST)

            if self.augmentation:
                if mask is not None:
                    sample = self.augmentation(image=image, mask=mask)
                    image, mask = sample["image"], sample["mask"]
                else:
                    sample = self.augmentation(image=image)
                    image = sample["image"]

            image = image.transpose(2, 0, 1)

            return (image, mask) if mask is not None else (image, None)

        except Exception as e:
            print(f"[Warning] Skipping index {i} due to error: {e}")
            return self.__getitem__((i + 1) % len(self))

    def __len__(self):
        return len(self.ids)
# Training set images augmentation - optimized for 320x320 PV images
def get_training_augmentation():
    train_transform = [
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),  # Added vertical flip for PV panels (useful for aerial views)
        A.Affine(scale=(0.9, 1.1), translate_percent=(0.1, 0.1), rotate=(-15, 15), p=0.5)
,
        # Since images are already 320x320, we don't need padding/cropping
        # But keep these for safety in case of size variations
        A.PadIfNeeded(min_height=320, min_width=320, border_mode=0),
        A.RandomCrop(height=320, width=320),
        
        # Noise and blur - useful for making model robust to image quality variations
        A.GaussNoise(scale=(10, 50), p=0.3),
        A.OneOf([
            A.Blur(blur_limit=3, p=1),
            A.MotionBlur(blur_limit=3, p=1),
        ], p=0.3),
        
        # Color augmentations - important for PV panel detection under different lighting
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1),
            A.RandomGamma(gamma_limit=(80, 120), p=1),
            A.CLAHE(clip_limit=2.0, p=1),
        ], p=0.8),
        
        # Hue/Saturation changes for different weather/lighting conditions
        A.HueSaturationValue(
            hue_shift_limit=10, 
            sat_shift_limit=20, 
            val_shift_limit=10, 
            p=0.5
        ),
        
        # Shadow simulation (useful for PV panels)
        A.RandomShadow(p=0.3),
    ]
    return A.Compose(train_transform)

def get_validation_augmentation():
    test_transform = [
        A.PadIfNeeded(min_height=320, min_width=320, border_mode=0),
        A.CenterCrop(height=320, width=320),
        ToTensorV2(),  # <- convert to tensor here
    ]
    return A.Compose(test_transform)

