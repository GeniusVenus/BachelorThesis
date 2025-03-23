import albumentations as A
from albumentations.pytorch import ToTensorV2

class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor

def get_transforms(config, split):
    if split == 'train':
        return A.Compose([
            # Spatial transformations for viewpoint robustness
            # A.RandomRotate90(p=0.5),
            # A.HorizontalFlip(p=0.5),
            # A.VerticalFlip(p=0.3),  # Aerial view makes vertical flips valid
            #
            # # Scale/perspective variations
            # A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.5),
            #
            # # Color augmentations for lighting/weather robustness
            # A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            # A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10, p=0.3),
            # A.CLAHE(clip_limit=2.0, p=0.2),  # Helps with shadow/highlight issues common in aerial imagery
            #
            # # Weather simulation augmentations
            # A.RandomShadow(p=0.1),
            # A.RandomFog(p=0.1),

            # Normalize and convert
            A.Normalize(mean=config['mean'], std=config['std'], max_pixel_value=255.0),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.Normalize(mean=config['mean'], std=config['std'], max_pixel_value=255.0),
            ToTensorV2(),
        ])
